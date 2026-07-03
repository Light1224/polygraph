"""SQLite cache for fetched Polymarket entities."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from polygraph.client import parse_json_field

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    condition_id TEXT,
    slug TEXT,
    question TEXT,
    description TEXT,
    active INTEGER,
    closed INTEGER,
    neg_risk INTEGER,
    combo_status TEXT,
    volume REAL,
    liquidity REAL,
    end_date TEXT,
    event_ids TEXT,
    tag_slugs TEXT,
    series_slug TEXT,
    group_item_title TEXT,
    raw_json TEXT NOT NULL,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_markets_condition ON markets(condition_id);
CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active, closed);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    slug TEXT,
    title TEXT,
    neg_risk INTEGER,
    enable_neg_risk INTEGER,
    neg_risk_augmented INTEGER,
    series_slug TEXT,
    tag_slugs TEXT,
    market_ids TEXT,
    raw_json TEXT NOT NULL,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tags (
    id TEXT PRIMARY KEY,
    slug TEXT,
    label TEXT,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS combo_eligible (
    condition_id TEXT PRIMARY KEY,
    title TEXT,
    tags TEXT,
    volume REAL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    tier TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    mechanism TEXT,
    evidence_quote TEXT,
    evidence_json TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_active ON edges(active);

CREATE TABLE IF NOT EXISTS price_history (
    token_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    price REAL NOT NULL,
    PRIMARY KEY (token_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_price_history_token ON price_history(token_id);
"""


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(markets)")}
        extra = {
            "prob_yes": "REAL",
            "prob_no": "REAL",
            "delta_1d": "REAL",
            "delta_7d": "REAL",
            "token_id_yes": "TEXT",
            "spread": "REAL",
        }
        for name, typ in extra.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE markets ADD COLUMN {name} {typ}")

    def _prob_fields(self, market: dict[str, Any]) -> tuple:
        prices = parse_json_field(market.get("outcomePrices"), [])
        token_ids = parse_json_field(market.get("clobTokenIds"), [])
        prob_yes = float(prices[0]) if len(prices) > 0 and prices[0] else None
        prob_no = float(prices[1]) if len(prices) > 1 and prices[1] else None
        token_yes = str(token_ids[0]) if token_ids else None
        return (
            prob_yes,
            prob_no,
            market.get("oneDayPriceChange"),
            market.get("oneWeekPriceChange"),
            token_yes,
            market.get("spread"),
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def upsert_market(self, market: dict[str, Any]) -> None:
        events = market.get("events") or []
        event_ids = [str(e["id"]) for e in events if e.get("id") is not None]
        tags = market.get("tags") or []
        tag_slugs = [t.get("slug") or t.get("label") or str(t.get("id", "")) for t in tags]
        # propagate event-level tags when market has none
        if not tag_slugs:
            for e in events:
                for t in e.get("tags") or []:
                    slug = t.get("slug") or t.get("label")
                    if slug:
                        tag_slugs.append(slug)

        prob_yes, prob_no, d1, d7, token_yes, spread = self._prob_fields(market)

        self.conn.execute(
            """
            INSERT INTO markets (
                id, condition_id, slug, question, description,
                active, closed, neg_risk, combo_status,
                volume, liquidity, end_date,
                event_ids, tag_slugs, series_slug, group_item_title,
                prob_yes, prob_no, delta_1d, delta_7d, token_id_yes, spread,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                condition_id=excluded.condition_id,
                slug=excluded.slug,
                question=excluded.question,
                description=excluded.description,
                active=excluded.active,
                closed=excluded.closed,
                neg_risk=excluded.neg_risk,
                combo_status=excluded.combo_status,
                volume=excluded.volume,
                liquidity=excluded.liquidity,
                end_date=excluded.end_date,
                event_ids=excluded.event_ids,
                tag_slugs=excluded.tag_slugs,
                series_slug=excluded.series_slug,
                group_item_title=excluded.group_item_title,
                prob_yes=excluded.prob_yes,
                prob_no=excluded.prob_no,
                delta_1d=excluded.delta_1d,
                delta_7d=excluded.delta_7d,
                token_id_yes=excluded.token_id_yes,
                spread=excluded.spread,
                raw_json=excluded.raw_json,
                fetched_at=datetime('now')
            """,
            (
                str(market["id"]),
                market.get("conditionId"),
                market.get("slug"),
                market.get("question"),
                market.get("description"),
                int(bool(market.get("active"))),
                int(bool(market.get("closed"))),
                int(bool(market.get("negRisk"))),
                market.get("comboStatus"),
                market.get("volumeNum") or market.get("volume"),
                market.get("liquidityNum") or market.get("liquidity"),
                market.get("endDate"),
                json.dumps(event_ids),
                json.dumps(list(dict.fromkeys(tag_slugs))),
                market.get("seriesSlug"),
                market.get("groupItemTitle"),
                prob_yes,
                prob_no,
                d1,
                d7,
                token_yes,
                spread,
                json.dumps(market),
            ),
        )

    def upsert_event(self, event: dict[str, Any]) -> None:
        markets = event.get("markets") or []
        market_ids = [str(m["id"]) for m in markets if m.get("id") is not None]
        tags = event.get("tags") or []
        tag_slugs = [t.get("slug") or t.get("label") or str(t.get("id", "")) for t in tags]

        self.conn.execute(
            """
            INSERT INTO events (
                id, slug, title, neg_risk, enable_neg_risk, neg_risk_augmented,
                series_slug, tag_slugs, market_ids, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                slug=excluded.slug,
                title=excluded.title,
                neg_risk=excluded.neg_risk,
                enable_neg_risk=excluded.enable_neg_risk,
                neg_risk_augmented=excluded.neg_risk_augmented,
                series_slug=excluded.series_slug,
                tag_slugs=excluded.tag_slugs,
                market_ids=excluded.market_ids,
                raw_json=excluded.raw_json,
                fetched_at=datetime('now')
            """,
            (
                str(event["id"]),
                event.get("slug"),
                event.get("title"),
                int(bool(event.get("negRisk"))),
                int(bool(event.get("enableNegRisk"))),
                int(bool(event.get("negRiskAugmented"))),
                event.get("seriesSlug"),
                json.dumps(tag_slugs),
                json.dumps(market_ids),
                json.dumps(event),
            ),
        )

    def upsert_tag(self, tag: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO tags (id, slug, label, raw_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                slug=excluded.slug,
                label=excluded.label,
                raw_json=excluded.raw_json
            """,
            (
                str(tag["id"]),
                tag.get("slug"),
                tag.get("label"),
                json.dumps(tag),
            ),
        )

    def upsert_combo(self, combo: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO combo_eligible (condition_id, title, tags, volume)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(condition_id) DO UPDATE SET
                title=excluded.title,
                tags=excluded.tags,
                volume=excluded.volume
            """,
            (
                combo.get("condition_id"),
                combo.get("title"),
                json.dumps(combo.get("tags") or []),
                combo.get("volume"),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def market_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM markets").fetchone()
        return int(row[0])

    def iter_markets(self, *, active_only: bool = False) -> Iterator[sqlite3.Row]:
        if active_only:
            query = "SELECT * FROM markets WHERE active=1 AND closed=0"
        else:
            query = "SELECT * FROM markets"
        yield from self.conn.execute(query)

    _INFER_MARKET_SQL = """
        SELECT id, slug, question, description, active, closed, neg_risk,
               volume, liquidity, end_date, event_ids, tag_slugs, series_slug,
               group_item_title, prob_yes, prob_no, delta_1d, delta_7d
        FROM markets
    """

    def iter_markets_for_infer(self) -> Iterator[dict[str, Any]]:
        """Lightweight rows for inference (no raw_json blob)."""
        for row in self.conn.execute(self._INFER_MARKET_SQL):
            yield {
                "id": row[0],
                "slug": row[1],
                "question": row[2],
                "description": row[3],
                "active": row[4],
                "closed": row[5],
                "neg_risk": row[6],
                "volume": row[7],
                "liquidity": row[8],
                "end_date": row[9],
                "event_ids": row[10],
                "tag_slugs": row[11],
                "series_slug": row[12],
                "group_item_title": row[13],
                "prob_yes": row[14],
                "prob_no": row[15],
                "delta_1d": row[16],
                "delta_7d": row[17],
            }

    _INFER_EVENT_SQL = """
        SELECT id, slug, title, neg_risk, enable_neg_risk, series_slug,
               tag_slugs, market_ids
        FROM events
    """

    def iter_events_for_infer(self) -> Iterator[dict[str, Any]]:
        for row in self.conn.execute(self._INFER_EVENT_SQL):
            yield {
                "id": row[0],
                "slug": row[1],
                "title": row[2],
                "neg_risk": row[3],
                "enable_neg_risk": row[4],
                "series_slug": row[5],
                "tag_slugs": row[6],
                "market_ids": row[7],
            }

    def iter_events(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute("SELECT * FROM events")

    def combo_condition_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT condition_id FROM combo_eligible")
        return {row[0] for row in rows if row[0]}

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def price_history_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM price_history").fetchone()
        return int(row[0])

    def markets_with_price_history_count(self) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(DISTINCT m.id)
            FROM markets m
            INNER JOIN price_history ph ON ph.token_id = m.token_id_yes
            """
        ).fetchone()
        return int(row[0])

    def has_price_history(self) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM price_history LIMIT 1"
        ).fetchone()
        return row is not None

    def iter_markets_for_prices(
        self,
        *,
        min_volume: float = 0,
        limit: int | None = None,
        skip_fetched: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Markets eligible for CLOB history fetch (need token_id_yes)."""
        sql = """
            SELECT m.id, m.token_id_yes, m.volume, m.question
            FROM markets m
            WHERE m.token_id_yes IS NOT NULL AND m.token_id_yes != ''
              AND COALESCE(m.volume, 0) >= ?
        """
        params: list[Any] = [min_volume]
        if skip_fetched:
            sql += """
              AND NOT EXISTS (
                SELECT 1 FROM price_history ph WHERE ph.token_id = m.token_id_yes
              )
            """
        sql += " ORDER BY m.volume DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self.conn.execute(sql, params):
            yield {
                "id": row[0],
                "token_id_yes": row[1],
                "volume": row[2],
                "question": row[3],
            }

    def upsert_price_points(self, token_id: str, points: list[tuple[int, float]]) -> int:
        if not points:
            return 0
        self.conn.executemany(
            """
            INSERT INTO price_history (token_id, ts, price) VALUES (?, ?, ?)
            ON CONFLICT(token_id, ts) DO UPDATE SET price=excluded.price
            """,
            [(token_id, ts, price) for ts, price in points],
        )
        return len(points)

    def get_price_series(self, token_id: str) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT ts, price FROM price_history WHERE token_id=? ORDER BY ts",
            (token_id,),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def get_market_price_series(self, market_id: str) -> list[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT token_id_yes FROM markets WHERE id=?", (market_id,)
        ).fetchone()
        if not row or not row[0]:
            return []
        return [
            {"ts": ts, "price": price}
            for ts, price in self.get_price_series(str(row[0]))
        ]

    def price_series_by_market_ids(
        self, market_ids: list[str]
    ) -> dict[str, list[tuple[int, float]]]:
        """Batch load price series for inference (market_id → sorted points)."""
        if not market_ids:
            return {}
        placeholders = ",".join("?" * len(market_ids))
        rows = self.conn.execute(
            f"SELECT id, token_id_yes FROM markets WHERE id IN ({placeholders})",
            market_ids,
        ).fetchall()
        token_to_market = {str(r[1]): str(r[0]) for r in rows if r[1]}
        if not token_to_market:
            return {}
        tokens = list(token_to_market.keys())
        ph = ",".join("?" * len(tokens))
        hist = self.conn.execute(
            f"SELECT token_id, ts, price FROM price_history WHERE token_id IN ({ph}) ORDER BY ts",
            tokens,
        ).fetchall()
        out: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for token_id, ts, price in hist:
            mid = token_to_market.get(str(token_id))
            if mid:
                out[mid].append((int(ts), float(price)))
        return dict(out)

    def ensure_search_index(self) -> None:
        """FTS5 index for fast market search."""
        self.conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS markets_fts USING fts5(
                question, slug, group_item_title,
                content='markets',
                content_rowid='rowid'
            );
            """
        )
        fts_count = self.conn.execute("SELECT COUNT(*) FROM markets_fts").fetchone()[0]
        market_count = self.market_count()
        if fts_count < market_count:
            self.conn.execute("INSERT INTO markets_fts(markets_fts) VALUES('rebuild')")
            self.conn.commit()

    @staticmethod
    def _fts_query(raw: str) -> str | None:
        """Build safe FTS5 prefix query from user input."""
        tokens = re.findall(r"[\w']+", raw.lower())
        tokens = [t for t in tokens if len(t) > 1]
        if not tokens:
            return None
        return " ".join(f'"{t}"*' for t in tokens[:8])

    def search_markets(
        self,
        query: str,
        *,
        limit: int = 20,
        embedding_index=None,
    ) -> list[dict[str, Any]]:
        self.ensure_search_index()
        q = query.strip()
        if not q:
            return []

        rows: list = []
        fts_q = self._fts_query(q)
        if fts_q:
            try:
                rows = self.conn.execute(
                    """
                    SELECT m.id, m.question, m.slug, m.volume, m.neg_risk, m.group_item_title, m.prob_yes
                    FROM markets_fts fts
                    JOIN markets m ON m.rowid = fts.rowid
                    WHERE markets_fts MATCH ?
                    ORDER BY bm25(markets_fts), m.volume DESC
                    LIMIT ?
                    """,
                    (fts_q, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            tokens = re.findall(r"[\w']+", q.lower())
            tokens = [t for t in tokens if len(t) > 1][:6]
            if tokens:
                clauses = " AND ".join(
                    "(question LIKE ? OR slug LIKE ? OR group_item_title LIKE ?)"
                    for _ in tokens
                )
                params: list = []
                for t in tokens:
                    like = f"%{t}%"
                    params.extend([like, like, like])
                params.append(limit)
                rows = self.conn.execute(
                    f"""
                    SELECT id, question, slug, volume, neg_risk, group_item_title, prob_yes
                    FROM markets
                    WHERE {clauses}
                    ORDER BY volume DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()

        if not rows and embedding_index is not None:
            try:
                hits = embedding_index.search(q, k=limit)
                if hits:
                    ids = [h[0] for h in hits]
                    placeholders = ",".join("?" * len(ids))
                    rows = self.conn.execute(
                        f"""
                        SELECT id, question, slug, volume, neg_risk, group_item_title, prob_yes
                        FROM markets WHERE id IN ({placeholders})
                        """,
                        ids,
                    ).fetchall()
                    order = {mid: i for i, (mid, _) in enumerate(hits)}
                    rows = sorted(rows, key=lambda r: order.get(r[0], 999))
            except Exception:
                pass

        return [
            {
                "id": r[0],
                "question": r[1],
                "slug": r[2],
                "volume": r[3],
                "neg_risk": bool(r[4]),
                "group_item_title": r[5],
                "prob_yes": r[6],
            }
            for r in rows
        ]

    def get_market(self, market_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, question, slug, description, volume, liquidity, neg_risk, "
            "combo_status, end_date, tag_slugs, event_ids, group_item_title, active, closed, "
            "prob_yes, prob_no, delta_1d, delta_7d "
            "FROM markets WHERE id = ?",
            (market_id,),
        ).fetchone()
        if not row:
            return None
        event_ids = json.loads(row[10] or "[]")
        events_meta: list[dict[str, Any]] = []
        for eid in event_ids[:3]:
            ev = self.conn.execute(
                "SELECT id, title, tag_slugs, series_slug FROM events WHERE id=?",
                (str(eid),),
            ).fetchone()
            if ev:
                events_meta.append(
                    {
                        "id": ev[0],
                        "title": ev[1],
                        "tag_slugs": json.loads(ev[2] or "[]"),
                        "series_slug": ev[3],
                    }
                )
        tag_slugs = json.loads(row[9] or "[]")
        if not tag_slugs and events_meta:
            for ev in events_meta:
                tag_slugs.extend(ev.get("tag_slugs") or [])
            tag_slugs = list(dict.fromkeys(tag_slugs))

        base = {
            "id": row[0],
            "question": row[1],
            "slug": row[2],
            "description": row[3],
            "volume": row[4],
            "liquidity": row[5],
            "neg_risk": bool(row[6]),
            "combo_status": row[7],
            "end_date": row[8],
            "tag_slugs": tag_slugs,
            "event_ids": event_ids,
            "events": events_meta,
            "event_title": " | ".join(e["title"] for e in events_meta if e.get("title")),
            "group_item_title": row[11],
            "active": bool(row[12]),
            "closed": bool(row[13]),
            "prob_yes": row[14],
            "prob_no": row[15],
            "delta_1d": row[16],
            "delta_7d": row[17],
            "polymarket_url": f"https://polymarket.com/event/{row[2]}",
        }
        from polygraph.infer.context import build_context_text
        from polygraph.infer.domain import market_domain
        from polygraph.infer.entities import extract_proper_spans

        base["event_tags"] = tag_slugs
        base["context_text"] = build_context_text(
            {**base, "event_tags": tag_slugs, "event_description": row[3]}
        )
        base["entities"] = extract_proper_spans(base["question"])
        if base["event_title"]:
            base["entities"] = list(
                dict.fromkeys(base["entities"] + extract_proper_spans(base["event_title"]))
            )
        base["domain"] = market_domain(base)
        return base

    def enrich_from_raw(self) -> None:
        """Backfill prob/delta columns from raw_json for existing rows."""
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM markets WHERE prob_yes IS NULL OR prob_yes = ''"
        ).fetchone()[0]
        if not pending:
            return
        rows = self.conn.execute(
            "SELECT id, raw_json FROM markets WHERE prob_yes IS NULL OR prob_yes = ''"
        ).fetchall()
        for row in rows:
            m = json.loads(row[1])
            prob_yes, prob_no, d1, d7, token_yes, spread = self._prob_fields(m)
            self.conn.execute(
                """UPDATE markets SET prob_yes=?, prob_no=?, delta_1d=?, delta_7d=?,
                   token_id_yes=?, spread=? WHERE id=?""",
                (prob_yes, prob_no, d1, d7, token_yes, spread, row[0]),
            )
        self.conn.commit()

    def clear_edges(self) -> None:
        self.conn.execute("DELETE FROM edges")
        self.conn.commit()

    def upsert_edge(self, edge: Any) -> None:
        self._upsert_edge_row(edge)

    def upsert_edges_batch(self, edges: list[Any]) -> None:
        if not edges:
            return
        self.conn.executemany(
            """
            INSERT INTO edges (
                source_id, target_id, relation, tier, direction,
                confidence, mechanism, evidence_quote, evidence_json, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                tier=excluded.tier,
                direction=excluded.direction,
                confidence=excluded.confidence,
                mechanism=excluded.mechanism,
                evidence_quote=excluded.evidence_quote,
                evidence_json=excluded.evidence_json,
                active=excluded.active
            """,
            [self._edge_params(e) for e in edges],
        )

    def _edge_params(self, edge: Any) -> tuple:
        return (
            edge.source_id,
            edge.target_id,
            edge.relation,
            edge.tier,
            edge.direction,
            edge.confidence,
            edge.mechanism,
            edge.evidence_quote,
            json.dumps(edge.evidence),
            edge.active,
        )

    def _upsert_edge_row(self, edge: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO edges (
                source_id, target_id, relation, tier, direction,
                confidence, mechanism, evidence_quote, evidence_json, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                tier=excluded.tier,
                direction=excluded.direction,
                confidence=excluded.confidence,
                mechanism=excluded.mechanism,
                evidence_quote=excluded.evidence_quote,
                evidence_json=excluded.evidence_json,
                active=excluded.active
            """,
            self._edge_params(edge),
        )

    def iter_edges(self, *, active_only: bool = True) -> Iterator[sqlite3.Row]:
        if active_only:
            q = "SELECT * FROM edges WHERE active=1"
        else:
            q = "SELECT * FROM edges"
        yield from self.conn.execute(q)

    def edge_count(self, *, active_only: bool = True) -> int:
        if active_only:
            row = self.conn.execute("SELECT COUNT(*) FROM edges WHERE active=1").fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return int(row[0])
