# Polygraph

Map **Polymarket** as an interactive graph of interacting prediction markets — read-only, no wallet, no trading.

Ingests ~45k markets from public APIs, infers typed relationship edges (structural, temporal, empirical), and explores them in a local web UI.

## Edge types

Markets are **nodes**. **Edges** are tiered by how much to trust them:

| Relation | Tier | Meaning |
|----------|------|---------|
| `EXCLUDES` | GROUND | Neg-risk siblings — at most one Yes (stored, hidden in default UI) |
| `CO_EVENT` | GROUND | Same multi-outcome event |
| `SUBEVENT` | GROUND | Earlier deadline ⊆ later deadline within an event |
| `TEMPORAL` | GROUND | Dependent market gated on an anchor event |
| `SHARED_TAG` | RELATED | Shared rare topic tag |
| `RELATED` | RELATED | Entity / embedding similarity (isolated markets only) |
| `COMOVES` | EMPIRICAL | Correlated price moves within an event |

Structural edges come from Polymarket's data model and resolution text. `COMOVES` uses CLOB price history when available.

## Quick start

Requires **Python 3.11+** and a **conda** env (recommended).

```bash
git clone https://github.com/YOUR_USER/polygraph.git
cd polygraph

conda env create -f environment.yml
conda activate polygraph
pip install -e ".[web,embeddings]"
```

### First-time build

```bash
# 1. Download active markets & events (~2–3 min)
polygraph fetch --scope active

# 2. Local embeddings for search + TEMPORAL anchors (~4 min first run)
polygraph embed

# 3. Optional: price history for sparklines + richer COMOVES edges
polygraph prices --limit 3000

# 4. Infer edges + write graph (~15s on ~45k markets)
polygraph infer
polygraph build --skip-infer

# 5. Explore
polygraph serve
# → http://127.0.0.1:8080
```

Or run inference and GraphML export in one step: `polygraph build`.

### Daily refresh

```bash
polygraph update              # fetch → embed → prices → infer → build
polygraph update --loop       # repeat every 24h (Ctrl+C to stop)
```

Update metadata is stored in SQLite (`last_update_at`, `last_update_result`).

## CLI

| Command | Description |
|---------|-------------|
| `fetch` | Ingest markets, events, tags from Gamma / Combo APIs |
| `embed` | Build local MiniLM embedding index |
| `prices` | Fetch CLOB daily price history |
| `infer` | Run edge inference pipeline |
| `build` | Infer (unless `--skip-infer`) + export GraphML |
| `update` | Full daily refresh cycle |
| `serve` | Launch interactive graph UI |
| `stats` | Show cache counts |

## Outputs (gitignored, generated locally)

| Path | Contents |
|------|----------|
| `data/polygraph.db` | SQLite — markets, events, inferred edges, price history |
| `data/market_graph.graphml` | NetworkX graph export |
| `data/vectors/` | Embedding index (mmap-backed) |
| `data/validation_report.json` | Post-infer audit metrics |

## Architecture

```
Gamma API ──► fetch.py ──► SQLite (store.py)
CLOB API  ──► prices.py ──┘
                              │
Combo API ────────────────────┘
                              │
                    embed (MiniLM vectors)
                              │
                              ▼
                    infer/ pipeline ──► edges table
                              │
                              ▼
                         graph.py ──► GraphML
                              │
                              ▼
                    FastAPI + vis.js web UI
```

## Stack

- **httpx** — Gamma, Combo, CLOB clients
- **SQLite + FTS5** — catalog cache and search
- **sentence-transformers** — `all-MiniLM-L6-v2` embeddings
- **NetworkX** — graph build and export
- **FastAPI + vis-network** — local explorer

## Notes

- No authentication required — public catalog and price-history endpoints only.
- No trading code — never touches order placement.
- `polygraph fetch --scope active` for open markets; `--scope all` for full historical crawl.
- Large artifacts stay local (`data/` is gitignored). Clone + run the commands above to rebuild.

## License

MIT — see [LICENSE](LICENSE).

## References

- [Polymarket Gamma API](https://docs.polymarket.com/api-reference/introduction)
- [Neg-risk markets](https://docs.polymarket.com/advanced/neg-risk)
