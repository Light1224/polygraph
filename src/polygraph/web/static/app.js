const $ = (sel) => document.querySelector(sel);

let network = null;
let currentSeed = null;
let lastSubgraph = null;
let lastDetail = null;
let searchTimer = null;
let nodeSliderTimer = null;
let activeResultIdx = -1;
let physicsFrozen = false;
let edgeColors = {};
const hiddenEdgeTypes = new Set();

const EDGE_LABELS = {
  EXCLUDES: "Mutually exclusive",
  CO_EVENT: "Same event",
  SHARED_TAG: "Shared topic",
  SUBEVENT: "Nested deadline",
  TEMPORAL: "Before / gated on",
  RELATED: "Related belief",
  RESOLVES_IF: "Conditional resolve",
  LEADS: "Price leads",
  COMOVES: "Moves together",
  IMPLIES: "Inferred link",
};

const WEAK_EDGE_TYPES = new Set(["RELATED", "COMOVES", "SHARED_TAG"]);

const PRESETS = {
  local: { mode: "explore", depth: 1, maxNodes: 120 },
  near: { mode: "explore", depth: 2, maxNodes: 500 },
  wide: { mode: "explore", depth: 4, maxNodes: 1200 },
  deep: { mode: "explore", depth: 8, maxNodes: 2500 },
};

function probColor(p) {
  if (p == null || isNaN(p)) return { background: "#334155", border: "#475569" };
  const t = Math.max(0, Math.min(1, p));
  if (t < 0.5) {
    const u = t * 2;
    const r = Math.round(220 - u * 80);
    const g = Math.round(80 + u * 60);
    return { background: `rgb(${r},${g},100)`, border: "#64748b" };
  }
  const u = (t - 0.5) * 2;
  const r = Math.round(140 - u * 100);
  const g = Math.round(140 + u * 60);
  return { background: `rgb(${r},${g},100)`, border: "#64748b" };
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function fmtVolume(v) {
  if (!v) return "—";
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}

function fmtProb(p) {
  if (p == null || isNaN(p)) return null;
  return `${(p * 100).toFixed(1)}%`;
}

function fmtDelta(d, label) {
  if (d == null || isNaN(d)) return "";
  const sign = d >= 0 ? "+" : "";
  const cls = d >= 0 ? "delta-up" : "delta-down";
  return `<span class="delta ${cls}" title="${label}">${label} ${sign}${(d * 100).toFixed(1)}pp</span>`;
}

function probBarHtml(probYes, probNo) {
  const py = probYes != null && !isNaN(probYes) ? Math.max(0, Math.min(1, probYes)) : null;
  const pn = probNo != null && !isNaN(probNo) ? probNo : py != null ? 1 - py : null;
  const pyLabel = fmtProb(py) || "—";
  const pnLabel = fmtProb(pn) || "—";
  const width = py != null ? (py * 100).toFixed(1) : "0";
  return `
    <div class="prob-bar-wrap">
      <div class="prob-bar"><div class="prob-yes" style="width:${width}%"></div></div>
      <div class="prob-labels">
        <span class="yes">Yes ${pyLabel}</span>
        <span class="no">No ${pnLabel}</span>
      </div>
    </div>`;
}

function setLoading(on) {
  $("#loading").classList.toggle("show", on);
}

function queryParams() {
  const mode = $("#view-mode").value;
  const depth = $("#depth").value;
  const maxNodes = $("#max-nodes").value;
  const ex = $("#show-excludes").checked ? "&include_excludes=true" : "";
  return `mode=${mode}&depth=${depth}&max_nodes=${maxNodes}${ex}`;
}

function initDepthSelect() {
  const sel = $("#depth");
  sel.innerHTML = "";
  for (let i = 1; i <= 20; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = String(i);
    if (i === 2) opt.selected = true;
    sel.appendChild(opt);
  }
}

function renderLegend() {
  const legend = $("#legend");
  legend.innerHTML = "";
  for (const [type, color] of Object.entries(edgeColors)) {
    const div = document.createElement("div");
    div.className = "legend-item";
    div.dataset.type = type;
    if (hiddenEdgeTypes.has(type)) div.classList.add("off");
    div.innerHTML = `<span class="legend-swatch" style="background:${color}"></span><span class="legend-label">${EDGE_LABELS[type] || type}</span>`;
    div.addEventListener("click", () => {
      if (hiddenEdgeTypes.has(type)) hiddenEdgeTypes.delete(type);
      else hiddenEdgeTypes.add(type);
      div.classList.toggle("off");
      if (lastSubgraph) renderGraph(lastSubgraph);
    });
    legend.appendChild(div);
  }
}

async function loadStats() {
  try {
    const s = await api("/api/stats");
    let pill = `${s.nodes.toLocaleString()} markets · ${s.edges.toLocaleString()} edges`;
    if (s.markets_with_prices) {
      pill += ` · ${s.markets_with_prices.toLocaleString()} w/ prices`;
    }
    $("#stats-pill").textContent = pill;
    edgeColors = s.edge_colors || {};
    renderLegend();
  } catch (e) {
    $("#stats-pill").textContent = "no graph — run polygraph build";
  }
}

function physicsOptions(subgraph) {
  const n = subgraph.node_count;
  const focus = subgraph.mode === "focus";
  const compact = n <= 60;

  return {
    enabled: !physicsFrozen,
    solver: "forceAtlas2Based",
    forceAtlas2Based: {
      gravitationalConstant: compact ? -40 : -28,
      centralGravity: 0.008,
      springLength: focus ? 110 : compact ? 95 : 115,
      springConstant: 0.045,
      damping: 0.88,
      avoidOverlap: compact ? 0.35 : 0.2,
    },
    stabilization: {
      iterations: compact ? 100 : Math.min(200, 80 + n * 0.35),
      fit: true,
      updateInterval: 25,
    },
    maxVelocity: 35,
    minVelocity: 0.75,
    timestep: 0.45,
  };
}

function stabilizeAndCalm(net) {
  if (physicsFrozen) return;
  const handler = () => {
    net.off("stabilizationIterationsDone", handler);
    calmPhysics(net);
  };
  net.on("stabilizationIterationsDone", handler);
  net.stabilize();
}

function calmPhysics(net) {
  if (physicsFrozen) {
    net.setOptions({ physics: { enabled: false } });
    return;
  }
  net.setOptions({
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -18,
        centralGravity: 0.004,
        springLength: 100,
        springConstant: 0.028,
        damping: 0.94,
        avoidOverlap: 0.15,
      },
      stabilization: false,
      maxVelocity: 12,
      minVelocity: 0.4,
      timestep: 0.35,
    },
  });
}

function edgeVisStyle(e) {
  const weak = WEAK_EDGE_TYPES.has(e.relation) || e.tier === "RELATED" || e.tier === "EMPIRICAL";
  const conf = e.confidence || 0.5;
  return {
    dashes: weak ? [6, 4] : false,
    width: Math.max(0.6, Math.min(3, conf * 2)),
    color: {
      color: e.color,
      highlight: "#e2e8f0",
      opacity: weak ? 0.55 : 0.88,
    },
  };
}

function buildVisData(subgraph) {
  const large = subgraph.node_count > 250;
  const visibleEdges = subgraph.edges.filter((e) => !hiddenEdgeTypes.has(e.relation));
  const linkedIds = new Set();
  for (const e of visibleEdges) {
    linkedIds.add(e.from);
    linkedIds.add(e.to);
  }
  if (currentSeed) linkedIds.add(currentSeed);

  const nodes = new vis.DataSet(
    subgraph.nodes
      .filter((n) => linkedIds.has(n.id))
      .map((n) => {
        const pc = probColor(n.prob_yes);
        return {
          id: n.id,
          label: large ? "" : n.label,
          title: `${n.question}\nP(Yes): ${((n.prob_yes ?? 0) * 100).toFixed(0)}% · vol ${fmtVolume(n.volume)}`,
          value: Math.max(3, Math.min(24, 3 + Math.log10((n.volume || 1) + 1) * 2.5)),
          color: {
            background: n.is_seed ? "#fbbf24" : pc.background,
            border: n.is_seed ? "#f59e0b" : pc.border,
            highlight: { background: "#6366f1", border: "#818cf8" },
          },
          font: { color: "#e8edf5", size: n.is_seed ? 12 : large ? 0 : 10 },
          borderWidth: n.is_seed ? 3 : 1,
        };
      })
  );

  const showEdgeLabels = visibleEdges.length < 80;
  const edges = new vis.DataSet(
    visibleEdges.map((e, i) => {
      const style = edgeVisStyle(e);
      return {
        id: i,
        from: e.from,
        to: e.to,
        arrows: e.arrows || (e.direction === "forward" ? "to" : undefined),
        title: [e.relation, e.mechanism, e.evidence_quote].filter(Boolean).join("\n"),
        label: showEdgeLabels ? e.relation : undefined,
        font: { size: 9, color: "#94a3b8", strokeWidth: 0 },
        ...style,
      };
    })
  );

  return { nodes, edges };
}

function renderGraph(subgraph) {
  const container = $("#graph");
  const focus = subgraph.mode === "focus";
  const { nodes, edges } = buildVisData(subgraph);

  $("#graph-badge").textContent = focus
    ? `${nodes.length} nodes · ${edges.length} links · focus`
    : `${nodes.length} nodes · ${edges.length} edges · ${subgraph.depth} hop${subgraph.depth === 1 ? "" : "s"}`;

  const options = {
    nodes: { shape: "dot", shadow: false, borderWidth: 1, font: { face: "IBM Plex Sans" } },
    edges: { smooth: { type: "dynamic", roundness: 0.35 }, shadow: false },
    physics: physicsOptions(subgraph),
    interaction: {
      hover: true,
      tooltipDelay: 60,
      navigationButtons: false,
      keyboard: true,
      hideEdgesOnDrag: true,
      hideEdgesOnZoom: false,
      dragNodes: true,
      dragView: true,
      zoomView: true,
    },
  };

  if (network) {
    network.setData({ nodes, edges });
    network.setOptions(options);
    stabilizeAndCalm(network);
  } else {
    network = new vis.Network(container, { nodes, edges }, options);
    stabilizeAndCalm(network);
    network.on("click", (params) => {
      if (params.nodes.length) selectMarket(params.nodes[0], false);
    });
  }
}

async function selectMarket(id, updateSearch = true) {
  currentSeed = id;
  setLoading(true);
  try {
    const [detail, subgraph, prices] = await Promise.all([
      api(`/api/market/${id}`),
      api(`/api/subgraph/${id}?${queryParams()}`),
      api(`/api/market/${id}/prices?limit=60`).catch(() => ({ points: [] })),
    ]);
    lastSubgraph = subgraph;
    lastDetail = detail;
    renderGraph(subgraph);
    renderDetail(detail, subgraph);
    renderPriceChart(prices, detail);
    if (updateSearch) $("#search").value = detail.question || "";
    closeSearch();
  } catch (e) {
    console.error(e);
    alert(e.message || "Failed to load graph");
  } finally {
    setLoading(false);
  }
}

function shortMech(mech) {
  if (!mech) return "";
  const s = mech.replace(/^[^:]+:\s*/, "").replace(/\[⚠.*$/, "").trim();
  return s.length > 72 ? s.slice(0, 71) + "…" : s;
}

function renderPriceChart(prices, detail) {
  const wrap = $("#price-chart-wrap");
  const svg = $("#price-chart");
  const meta = $("#price-chart-meta");
  const pts = prices?.points || [];
  wrap.classList.remove("snapshot");

  if (!pts.length) {
    const snap = detail?.prob_yes;
    if (snap == null || isNaN(snap)) {
      wrap.hidden = true;
      svg.innerHTML = "";
      meta.textContent = "";
      return;
    }
    wrap.hidden = false;
    wrap.classList.add("snapshot");
    const w = 280;
    const h = 56;
    const pad = 4;
    const y = h - pad - snap * (h - pad * 2);
    const color = "#6366f1";
    svg.innerHTML = `
      <line x1="${pad}" y1="${y.toFixed(1)}" x2="${w - pad}" y2="${y.toFixed(1)}"
        stroke="${color}" stroke-width="1.8" stroke-dasharray="4 3" />
    `;
    meta.textContent = `Snapshot P(Yes) ${(snap * 100).toFixed(1)}%`;
    const hint = wrap.querySelector(".price-chart-empty");
    if (hint) hint.remove();
    const empty = document.createElement("div");
    empty.className = "price-chart-empty";
    empty.textContent = "No CLOB history for this market — run polygraph prices for sparklines.";
    meta.after(empty);
    return;
  }

  wrap.querySelector(".price-chart-empty")?.remove();
  wrap.hidden = false;
  const vals = pts.map((p) => p.price);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 0.01;
  const w = 280;
  const h = 56;
  const pad = 4;
  const coords = vals.map((v, i) => {
    const x = pad + (i / Math.max(1, vals.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = vals[vals.length - 1];
  const first = vals[0];
  const delta = last - first;
  const color = delta >= 0 ? "#22c55e" : "#ef4444";
  svg.innerHTML = `
    <polyline fill="none" stroke="${color}" stroke-width="1.8" points="${coords.join(" ")}" />
    <polyline fill="${color}" fill-opacity="0.12" stroke="none"
      points="${pad},${h - pad} ${coords.join(" ")} ${w - pad},${h - pad}" />
  `;
  meta.textContent = `P(Yes) ${(last * 100).toFixed(1)}% · ${pts.length} pts · Δ ${(delta * 100).toFixed(1)}pp`;
}

function renderDetail(detail, subgraph) {
  $("#detail-empty").hidden = true;
  const el = $("#detail");
  el.hidden = false;

  const tags = (detail.tag_slugs || []).map((t) => `<span class="tag">${esc(t)}</span>`).join("");
  const entities = (detail.entities || []).map((t) => `<span class="tag entity">${esc(t)}</span>`).join("");
  const events = (detail.events || [])
    .map((e) => `<div class="meta-line">${esc(e.title || e.id)}</div>`)
    .join("");

  const deltas = [fmtDelta(detail.delta_1d, "1d"), fmtDelta(detail.delta_7d, "7d")]
    .filter(Boolean)
    .join("");

  const degree = detail.graph_degree ?? 0;
  const degreeNote =
    degree === 0
      ? `<div class="hint-box">No visual links — try <strong>explore</strong> mode, more hops, or enable weak edge types in the legend.</div>`
      : "";

  el.innerHTML = `
    <div class="question">${esc(detail.question)}</div>
    ${probBarHtml(detail.prob_yes, detail.prob_no)}
    ${deltas ? `<div class="delta-row">${deltas}</div>` : ""}
    <div class="field"><div class="label">Volume</div>${fmtVolume(detail.volume)}</div>
    <div class="field"><div class="label">Domain</div><span class="domain-pill domain-${esc(detail.domain || "general")}">${esc(detail.domain || "general")}</span></div>
    <div class="field"><div class="label">In view</div>${subgraph.node_count} nodes · ${subgraph.edge_count} edges</div>
    <div class="field"><div class="label">Graph degree</div>${degree}</div>
    ${degreeNote}
    ${detail.event_title ? `<div class="field"><div class="label">Event</div>${esc(detail.event_title)}</div>` : ""}
    ${events ? `<div class="field"><div class="label">Event context</div>${events}</div>` : ""}
    ${entities ? `<div class="field"><div class="label">Entities</div><div class="tags">${entities}</div></div>` : ""}
    ${tags ? `<div class="field"><div class="label">Topics</div><div class="tags">${tags}</div></div>` : ""}
    ${detail.end_date ? `<div class="field"><div class="label">End date</div>${esc(String(detail.end_date).slice(0, 10))}</div>` : ""}
    <div class="field" style="margin-top:0.75rem">
      <a class="ext" href="${detail.polymarket_url}" target="_blank" rel="noopener">View on Polymarket ↗</a>
    </div>
  `;

  const neighbors = $("#neighbors");
  const panel = $("#neighbors-panel");
  const direct = subgraph.edges
    .filter((e) => e.from === currentSeed || e.to === currentSeed)
    .filter((e) => !hiddenEdgeTypes.has(e.relation))
    .map((e) => {
      const otherId = e.from === currentSeed ? e.to : e.from;
      const node = subgraph.nodes.find((n) => n.id === otherId);
      const dir = e.from === currentSeed ? "→" : "←";
      return {
        id: otherId,
        question: node?.question || otherId,
        rel: e.relation,
        dir,
        mechanism: shortMech(e.mechanism),
        tier: e.tier,
      };
    });

  if (direct.length) {
    panel.hidden = false;
    neighbors.innerHTML = direct
      .slice(0, 30)
      .map(
        (n) => `
      <div class="neighbor" data-id="${n.id}">
        <div class="neighbor-q">${esc(n.question)}</div>
        <div class="edge-types">${n.dir} <span class="rel-${n.rel}">${n.rel}</span>${n.tier === "EMPIRICAL" || n.tier === "RELATED" ? " · weak" : ""}</div>
        ${n.mechanism ? `<div class="neighbor-mech">${esc(n.mechanism)}</div>` : ""}
      </div>`
      )
      .join("");
    neighbors.querySelectorAll(".neighbor").forEach((el) => {
      el.addEventListener("click", () => selectMarket(el.dataset.id));
    });
  } else {
    panel.hidden = true;
  }
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s || "";
  return d.innerHTML;
}

function searchItemHtml(r) {
  const prob = r.prob_yes != null ? `${(r.prob_yes * 100).toFixed(0)}% · ` : "";
  return `
      <div class="search-item" data-id="${r.id}">
        <div class="q">${esc(r.question)}</div>
        <div class="meta">${prob}${fmtVolume(r.volume)}${r.neg_risk ? " · neg-risk" : ""}</div>
      </div>`;
}

function renderSearchResults(results) {
  const box = $("#search-results");
  if (!results.length) {
    box.innerHTML = `<div class="search-item"><span class="q" style="color:var(--muted)">No matches</span></div>`;
    box.classList.add("open");
    return;
  }

  const grouped = new Map();
  for (const r of results) {
    const key = (r.group_item_title || "").trim();
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(r);
  }
  const multiGroup = [...grouped.keys()].filter((k) => k).length > 1;

  let html = "";
  for (const [key, items] of grouped) {
    if (multiGroup && key) {
      html += `<div class="search-group-label">${esc(key)}</div>`;
    }
    html += items.map((r) => searchItemHtml(r)).join("");
  }
  box.innerHTML = html;
  box.classList.add("open");
  activeResultIdx = -1;
  box.querySelectorAll(".search-item").forEach((el) => {
    el.addEventListener("click", () => selectMarket(el.dataset.id));
  });
}

async function doSearch(q) {
  if (!q.trim()) {
    closeSearch();
    return;
  }
  try {
    const { results } = await api(`/api/search?q=${encodeURIComponent(q)}&limit=15`);
    renderSearchResults(results);
  } catch (e) {
    console.error(e);
  }
}

function closeSearch() {
  $("#search-results").classList.remove("open");
  activeResultIdx = -1;
}

function syncViewControls() {
  const focus = $("#view-mode").value === "focus";
  $("#depth").disabled = focus;
  $("#max-nodes").disabled = focus;
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.disabled = focus;
  });
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  $("#view-mode").value = p.mode;
  $("#depth").value = String(p.depth);
  $("#max-nodes").value = String(p.maxNodes);
  $("#max-nodes-val").textContent = String(p.maxNodes);
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.preset === name);
  });
  syncViewControls();
  if (currentSeed) selectMarket(currentSeed, false);
}

function syncPresetHighlight() {
  const depth = Number($("#depth").value);
  const nodes = Number($("#max-nodes").value);
  const mode = $("#view-mode").value;
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    const p = PRESETS[btn.dataset.preset];
    btn.classList.toggle(
      "active",
      mode === p.mode && depth === p.depth && nodes === p.maxNodes
    );
  });
}

$("#search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => doSearch(e.target.value), 220);
});

$("#search").addEventListener("keydown", (e) => {
  const items = [...$("#search-results").querySelectorAll(".search-item")];
  if (e.key === "ArrowDown" && items.length) {
    e.preventDefault();
    activeResultIdx = Math.min(activeResultIdx + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle("active", i === activeResultIdx));
  } else if (e.key === "ArrowUp" && items.length) {
    e.preventDefault();
    activeResultIdx = Math.max(activeResultIdx - 1, 0);
    items.forEach((el, i) => el.classList.toggle("active", i === activeResultIdx));
  } else if (e.key === "Enter" && activeResultIdx >= 0) {
    e.preventDefault();
    selectMarket(items[activeResultIdx].dataset.id);
  } else if (e.key === "Escape") closeSearch();
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) closeSearch();
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select")) return;
  if (e.key === "/") {
    e.preventDefault();
    $("#search").focus();
    $("#search").select();
  } else if (e.key === "f" && currentSeed) {
    const sel = $("#view-mode");
    sel.value = sel.value === "focus" ? "explore" : "focus";
    syncViewControls();
    selectMarket(currentSeed, false);
  }
});

$("#view-mode").addEventListener("change", () => {
  syncViewControls();
  if (currentSeed) selectMarket(currentSeed, false);
});

$("#depth").addEventListener("change", () => {
  syncPresetHighlight();
  if (currentSeed) selectMarket(currentSeed, false);
});

function reloadSubgraph() {
  syncPresetHighlight();
  if (currentSeed) selectMarket(currentSeed, false);
}

$("#max-nodes").addEventListener("input", (e) => {
  $("#max-nodes-val").textContent = e.target.value;
  clearTimeout(nodeSliderTimer);
  nodeSliderTimer = setTimeout(reloadSubgraph, 350);
});
$("#max-nodes").addEventListener("change", reloadSubgraph);
$("#show-excludes").addEventListener("change", () => {
  if (currentSeed) selectMarket(currentSeed, false);
});

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
});

$("#btn-reset-layout").addEventListener("click", () => {
  if (network && lastSubgraph) {
    physicsFrozen = false;
    $("#btn-freeze").classList.remove("active");
    renderGraph(lastSubgraph);
  }
});

$("#btn-freeze").addEventListener("click", () => {
  physicsFrozen = !physicsFrozen;
  $("#btn-freeze").classList.toggle("active", physicsFrozen);
  if (network) {
    network.setOptions({ physics: { enabled: !physicsFrozen } });
    if (!physicsFrozen && lastSubgraph) calmPhysics(network);
  }
});

$("#btn-fit").addEventListener("click", () => {
  if (network) network.fit({ animation: { duration: 400, easingFunction: "easeInOutQuad" } });
});

async function boot() {
  initDepthSelect();
  syncViewControls();
  await loadStats();
  applyPreset("near");
  try {
    const { results } = await api("/api/search?q=GTA&limit=1");
    if (results.length) await selectMarket(results[0].id, false);
  } catch (_) {}
}

boot();
