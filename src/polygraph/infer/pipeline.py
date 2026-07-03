"""Run full edge inference pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from polygraph.infer.context import attach_event_context
from polygraph.infer.corpus import CorpusIndex
from polygraph.infer.empirical import infer_comoves
from polygraph.infer.entities import EntityIndex
from polygraph.infer.models import EdgeRecord
from polygraph.infer.text_rules import infer_ground_edges
from polygraph.infer.audit import audit_edges
from polygraph.infer.validate import neg_risk_probability_check, validate_edges
from polygraph.store import Store
from polygraph.vector.index import try_load

console = Console()


def _markets_as_dicts(store: Store) -> list[dict[str, Any]]:
    return list(store.iter_markets_for_infer())


def _events_as_dicts(store: Store) -> list[dict[str, Any]]:
    return list(store.iter_events_for_infer())


def load_embedding_index():
    """Load persisted index only — never encode on infer (use `polygraph embed`)."""
    try:
        index = try_load()
        if index is None:
            console.print(
                "[yellow]No embedding index — run `polygraph embed` for TEMPORAL/RELATED.[/yellow]"
            )
        return index
    except ImportError as exc:
        console.print(f"[yellow]Skipping embeddings:[/yellow] {exc}")
        return None
    except Exception as exc:
        console.print(f"[yellow]Could not load embeddings:[/yellow] {exc}")
        return None


def run_inference(store: Store, *, empirical: bool = True, use_embeddings: bool = True) -> list[EdgeRecord]:
    t0 = time.perf_counter()
    store.enrich_from_raw()
    markets = _markets_as_dicts(store)
    events = _events_as_dicts(store)
    attach_event_context(markets, events)
    markets_by_id = {str(m["id"]): m for m in markets}
    console.print(f"[dim]Loaded {len(markets):,} markets in {time.perf_counter() - t0:.1f}s[/dim]")

    embedding_index = None
    corpus: CorpusIndex | None = None
    entity_index: EntityIndex | None = None
    if use_embeddings:
        console.print("[bold]Embeddings[/bold] — mmap load (no re-encode)…")
        embedding_index = load_embedding_index()
        if embedding_index is not None:
            t_idx = time.perf_counter()
            corpus = CorpusIndex(markets)
            entity_index = EntityIndex(markets)
            console.print(f"[dim]Indices built in {time.perf_counter() - t_idx:.1f}s[/dim]")

    console.print("[bold]Tier A[/bold] — GROUND edges (text + structure)…")
    t_ground = time.perf_counter()
    ground = infer_ground_edges(
        markets,
        events,
        embedding_index=embedding_index,
        corpus=corpus,
        entity_index=entity_index,
    )
    console.print(f"  {len(ground)} ground edges ({time.perf_counter() - t_ground:.1f}s)")

    related_edges: list[EdgeRecord] = []
    if use_embeddings and embedding_index is not None:
        console.print("[bold]Tier R[/bold] — RELATED (isolated markets only)…")
        from polygraph.infer.semantic import infer_related

        existing = {tuple(sorted((e.source_id, e.target_id))) for e in ground}
        t_rel = time.perf_counter()
        related_edges = infer_related(
            markets,
            markets_by_id,
            embedding_index,
            existing_pairs=existing,
            ground_edges=ground,
            entity_index=entity_index,
        )
        console.print(f"  {len(related_edges)} related edges ({time.perf_counter() - t_rel:.1f}s)")

    empirical_edges: list[EdgeRecord] = []
    if empirical:
        console.print("[bold]Tier B[/bold] — EMPIRICAL edges (co-movement)…")
        t_emp = time.perf_counter()
        price_series: dict[str, list[tuple[int, float]]] = {}
        if store.has_price_history():
            from polygraph.infer.empirical import MIN_VOLUME

            eligible = [str(m["id"]) for m in markets if float(m.get("volume") or 0) >= MIN_VOLUME]
            price_series = store.price_series_by_market_ids(eligible)
            console.print(f"[dim]  price history: {len(price_series):,} markets[/dim]")
        empirical_edges = infer_comoves(
            markets,
            price_series=price_series or None,
        )
        console.print(f"  {len(empirical_edges)} comove edges ({time.perf_counter() - t_emp:.1f}s)")

    all_edges = ground + related_edges + empirical_edges
    all_edges, violations = validate_edges(all_edges, markets_by_id)
    violations.extend(neg_risk_probability_check(events, markets_by_id))
    audit = audit_edges(all_edges, markets_by_id)

    console.print(f"[bold]Saving[/bold] {len(all_edges):,} edges…")
    t_save = time.perf_counter()
    store.clear_edges()
    store.upsert_edges_batch(all_edges)
    store.commit()
    console.print(f"[dim]Saved in {time.perf_counter() - t_save:.1f}s[/dim]")

    report = {
        "edge_counts": _count_by_relation(all_edges),
        "active_counts": _count_by_relation([e for e in all_edges if e.active]),
        "audit": audit,
        "violations": violations,
        "violation_count": len(violations),
    }
    report_path = store.path.parent / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    table = Table(title="Inferred edges")
    table.add_column("Relation")
    table.add_column("Tier")
    table.add_column("Count", justify="right")
    table.add_column("Active", justify="right")
    for rel in sorted(_count_by_relation(all_edges).keys()):
        tier = next((e.tier for e in all_edges if e.relation == rel), "")
        table.add_row(
            rel,
            tier,
            str(_count_by_relation(all_edges)[rel]),
            str(_count_by_relation([e for e in all_edges if e.active]).get(rel, 0)),
        )
    console.print(table)
    console.print(
        f"[yellow]{len(violations)} constraint signals[/yellow] → {report_path}"
    )
    console.print(
        f"[dim]Total infer time: {time.perf_counter() - t0:.1f}s · "
        f"Connected: {audit['connected_markets']:,} · "
        f"Isolated: {audit['isolated_markets']:,} · "
        f"Avg degree: {audit['avg_visual_degree']}[/dim]"
    )
    return all_edges


def _count_by_relation(edges: list[EdgeRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in edges:
        counts[e.relation] = counts.get(e.relation, 0) + 1
    return counts


def run_inference_file(
    db_path: Path, *, empirical: bool = True, use_embeddings: bool = True
) -> list[EdgeRecord]:
    with Store(db_path) as store:
        return run_inference(store, empirical=empirical, use_embeddings=use_embeddings)
