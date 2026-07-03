# Polygraph

**Polygraph** maps [Polymarket](https://polymarket.com) as a graph of interacting prediction markets. It pulls the public catalog, infers how markets relate to each other, and lets you explore those relationships in a local web UI.

Read-only by design — no wallet, no order placement, no trading.

---

## What it does

1. **Ingest** ~45k markets, events, and tags from Polymarket's public Gamma and Combo APIs into SQLite.
2. **Embed** market text locally with `all-MiniLM-L6-v2` for search and anchor matching.
3. **Infer** typed edges — structural, temporal, and empirical — with provenance and confidence on every link.
4. **Build** a NetworkX graph and export GraphML.
5. **Explore** ego-subgraphs interactively: search, hop presets, edge filters, price sparklines.

The graph is a **belief landscape**, not a claim about ground truth. Nodes are markets with live-implied P(Yes). Edges say how resolution rules or observed prices connect one market to another.

---

## Edge types

Edges are grouped by tier — how much to trust them in the default view.

| Relation | Tier | What it means |
|----------|------|---------------|
| `CO_EVENT` | GROUND | Same multi-outcome event (star topology from highest-volume market) |
| `SUBEVENT` | GROUND | Earlier `end_date` → later deadline within one event |
| `TEMPORAL` | GROUND | Dependent question ("before X?") linked to an anchor market via text + entities + embeddings |
| `EXCLUDES` | GROUND | Neg-risk siblings — mutually exclusive (stored in DB, hidden in default UI to reduce clutter) |
| `SHARED_TAG` | RELATED | Shared rare topic tag (ultra-common tags skipped) |
| `RELATED` | RELATED | Entity overlap or semantic similarity — **only for isolated high-volume markets** |
| `COMOVES` | EMPIRICAL | Correlated daily returns (or recent Δ snapshots) within the same event |

**GROUND** edges come from Polymarket structure and resolution text. **EMPIRICAL** edges come from price co-movement. **RELATED** edges are weak hypotheses — dashed in the UI.

After inference, run `polygraph stats` or inspect `data/validation_report.json` for connectivity and edge counts.

---

## Quick start

**Requirements:** Python 3.11+, conda (recommended).

```bash
git clone https://github.com/YOUR_USER/polygraph.git
cd polygraph

conda env create -f environment.yml
conda activate polygraph
pip install -e ".[web,embeddings]"
```

### First run

```bash
polygraph fetch --scope active     # ~2–3 min — markets, events, tags
polygraph embed                    # ~4 min first run — local embedding index
polygraph prices --limit 3000      # optional — CLOB history for sparklines + COMOVES
polygraph infer                    # ~15 s — write edges to SQLite
polygraph build --skip-infer       # export GraphML
polygraph serve                    # http://127.0.0.1:8080
```

`polygraph build` alone runs inference and exports GraphML in one step (use `--skip-infer` if you already ran `infer`).

### Daily refresh

```bash
polygraph update                   # fetch → embed → prices → infer → build
polygraph update --loop              # every 24 h until Ctrl+C
polygraph update --no-prices         # skip CLOB fetch for a faster cycle
```

Timestamps and per-step results are saved in SQLite meta (`last_update_at`, `last_update_result`).

---

## Web UI

| Feature | Description |
|---------|-------------|
| Search | FTS5 + embedding search; `/` to focus |
| Presets | Local / Near / Wide / Deep — hop depth + node cap |
| Modes | **Explore** (BFS neighborhood) or **Focus** (direct neighbors) |
| Legend | Click edge types to filter |
| Detail panel | P(Yes) bar, 1d/7d deltas, neighbors with mechanism text |
| Sparklines | CLOB price history when available |

Keyboard: `/` search · `f` toggle focus/explore.

---

## CLI reference

| Command | Description |
|---------|-------------|
| `fetch` | Download catalog (`--scope active\|all`, `--fresh` to wipe DB) |
| `embed` | Build or refresh the local vector index |
| `prices` | Fetch CLOB daily price history for liquid markets |
| `infer` | Run the edge inference pipeline |
| `build` | Infer (unless `--skip-infer`) + write `market_graph.graphml` |
| `update` | Full refresh cycle; `--loop` for scheduled runs |
| `serve` | Start the FastAPI explorer |
| `stats` | Row counts and fetch scope |
| `neighbors` | CLI subgraph for a market ID |
| `edge-types` | Print edge type glossary |

---

## Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Gamma API  │     │  Combo API  │     │  CLOB API   │
│  markets    │     │  parlays    │     │  prices     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                    fetch.py / prices.py
                           │
                           ▼
                   SQLite (store.py)
                     markets · events
                     edges · price_history
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           embed       infer/       FTS5 search
         (MiniLM)    Tier A/B/R
              │            │
              └────────────┘
                           ▼
                      graph.py
                           │
                           ▼
                  market_graph.graphml
                           │
                           ▼
              FastAPI + vis-network UI
```

**Infer stages**

- **Tier A (GROUND)** — `EXCLUDES`, `CO_EVENT`, `SHARED_TAG`, `SUBEVENT`, `TEMPORAL`
- **Tier R (RELATED)** — semantic rescue for markets with no structural links
- **Tier B (EMPIRICAL)** — `COMOVES` from return correlation within events
- **Validate** — probability constraint checks, cross-domain filtering, redundant-edge deactivation

---

## Generated files

All large artifacts are **gitignored** — rebuild after clone.

| Path | Contents |
|------|----------|
| `data/polygraph.db` | SQLite cache (~800 MB with raw JSON; serving needs less) |
| `data/market_graph.graphml` | Graph export for NetworkX / Gephi |
| `data/vectors/` | Mmap-backed embedding index |
| `data/validation_report.json` | Edge counts, connectivity, constraint signals |

---

## Project layout

```
src/polygraph/
  fetch.py          API ingestion
  store.py          SQLite schema + search
  prices.py         CLOB price history
  graph.py          NetworkX build + GraphML export
  update.py         Daily refresh orchestration
  infer/            Edge inference pipeline
  vector/           Embedding index
  web/              FastAPI server + static UI
```

---

## Stack

- [httpx](https://www.python-httpx.org/) — HTTP clients
- [SQLite](https://www.sqlite.org/) + FTS5 — catalog and full-text search
- [sentence-transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2`
- [NetworkX](https://networkx.org/) — graph assembly
- [FastAPI](https://fastapi.tiangolo.com/) + [vis-network](https://visjs.github.io/vis-network/) — local explorer

---

## Notes

- Uses **public endpoints only** — no API keys.
- **No trading code** — CLOB is read-only price history.
- `fetch --scope active` for open markets; `--scope all` for the full historical crawl.
- Embeddings are built once with `embed` and mmap-loaded during infer (no re-encode on each run).

---

## License

MIT — see [LICENSE](LICENSE).

## References

- [Polymarket Gamma API](https://docs.polymarket.com/api-reference/introduction)
- [Neg-risk markets](https://docs.polymarket.com/advanced/neg-risk)
