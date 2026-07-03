"""FastAPI web server for interactive graph exploration."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from polygraph.config import DEFAULT_DB_PATH, DEFAULT_GRAPH_PATH
from polygraph.store import Store
from polygraph.vector.index import try_load
from polygraph.web.graph_service import ego_subgraph, graph_summary, load_graph

STATIC_DIR = Path(__file__).parent / "static"


def _embedding_index():
    return try_load()


def create_app(
    db_path: Path = DEFAULT_DB_PATH,
    graph_path: Path = DEFAULT_GRAPH_PATH,
) -> FastAPI:
    app = FastAPI(title="Polygraph", description="Polymarket market interaction graph")
    gpath = str(graph_path.resolve())
    dpath = db_path.resolve()

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "db": dpath.exists(),
            "graph": graph_path.exists(),
        }

    @app.get("/api/stats")
    def stats():
        if not graph_path.exists():
            raise HTTPException(404, "Graph not built. Run `polygraph build` first.")
        g = load_graph(gpath)
        summary = graph_summary(g, db_path=dpath)
        with Store(dpath) as store:
            summary["fetch_scope"] = store.get_meta("fetch_scope")
            summary["markets_in_db"] = store.market_count()
            summary["markets_with_prices"] = store.markets_with_price_history_count()
        return summary

    @app.get("/api/search")
    def search(
        q: str = Query("", min_length=0),
        limit: int = Query(20, ge=1, le=50),
    ):
        if not dpath.exists():
            raise HTTPException(404, "Database not found. Run `polygraph fetch` first.")
        with Store(dpath) as store:
            results = store.search_markets(q, limit=limit, embedding_index=_embedding_index())
        return {"query": q, "results": results}

    @app.get("/api/market/{market_id}")
    def market(market_id: str):
        if not dpath.exists():
            raise HTTPException(404, "Database not found.")
        with Store(dpath) as store:
            m = store.get_market(market_id)
        if not m:
            raise HTTPException(404, f"Market {market_id} not found")
        if graph_path.exists():
            g = load_graph(gpath)
            m["graph_degree"] = g.degree(market_id) if market_id in g else 0
        return m

    @app.get("/api/market/{market_id}/prices")
    def market_prices(market_id: str, limit: int = Query(90, ge=1, le=365)):
        if not dpath.exists():
            raise HTTPException(404, "Database not found.")
        with Store(dpath) as store:
            series = store.get_market_price_series(market_id)
        if not series:
            return {"market_id": market_id, "points": [], "source": "none"}
        points = series[-limit:]
        return {
            "market_id": market_id,
            "points": points,
            "source": "clob_history",
            "count": len(points),
        }

    @app.get("/api/subgraph/{market_id}")
    def subgraph(
        market_id: str,
        depth: int = Query(2, ge=1, le=20),
        max_nodes: int = Query(500, ge=10, le=3000),
        include_excludes: bool = Query(False),
        mode: str = Query("explore", pattern="^(explore|focus)$"),
    ):
        if not graph_path.exists():
            raise HTTPException(404, "Graph not built. Run `polygraph build` first.")
        g = load_graph(gpath)
        result = ego_subgraph(
            g,
            market_id,
            db_path=dpath,
            depth=depth,
            max_nodes=max_nodes,
            include_excludes=include_excludes,
            mode=mode,
        )
        if result.get("error"):
            raise HTTPException(404, result["error"])
        return result

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
