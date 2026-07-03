"""Read-only HTTP clients for Polymarket public APIs."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import httpx

from polygraph.config import COMBO_API, CLOB_API, GAMMA_API, PAGE_SIZE


class GammaClient:
    """Gamma API — markets, events, tags. Fully public, no auth."""

    def __init__(self, base_url: str = GAMMA_API, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GammaClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def paginate_offset(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        page_size: int = PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """Offset pagination (works for small collections like tags)."""
        base = dict(params or {})
        offset = 0
        while True:
            page = self._get(
                path,
                params={**base, "limit": page_size, "offset": offset},
            )
            if not page:
                break
            yield from page
            if len(page) < page_size:
                break
            offset += page_size
            time.sleep(0.05)

    def paginate_keyset(
        self,
        path: str,
        *,
        collection_key: str,
        params: dict[str, Any] | None = None,
        page_size: int = PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """
        Cursor pagination for large collections (markets, events).
        Gamma rejects offset > ~2000 on /markets and /events.
        """
        base = dict(params or {})
        cursor: str | None = None
        while True:
            req: dict[str, Any] = {**base, "limit": page_size}
            if cursor:
                req["after_cursor"] = cursor
            payload = self._get(path, params=req)
            items = payload.get(collection_key, [])
            yield from items
            cursor = payload.get("next_cursor")
            if not cursor or not items:
                break
            time.sleep(0.05)

    def iter_markets(self, **filters: Any) -> Iterator[dict[str, Any]]:
        yield from self.paginate_keyset(
            "/markets/keyset",
            collection_key="markets",
            params=filters,
        )

    def iter_events(self, **filters: Any) -> Iterator[dict[str, Any]]:
        yield from self.paginate_keyset(
            "/events/keyset",
            collection_key="events",
            params=filters,
        )

    def iter_tags(self) -> Iterator[dict[str, Any]]:
        yield from self.paginate_offset("/tags")


class ClobClient:
    """CLOB API — historical YES token prices (public, no auth)."""

    def __init__(self, base_url: str = CLOB_API, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "polygraph/0.1"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ClobClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_prices_history(
        self,
        token_id: str,
        *,
        interval: str = "1d",
        fidelity: int = 60,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[tuple[int, float]]:
        """Daily (or configured) price series for a CLOB outcome token."""
        params: dict[str, Any] = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        response = self._client.get("/prices-history", params=params)
        response.raise_for_status()
        payload = response.json()
        out: list[tuple[int, float]] = []
        for pt in payload.get("history") or []:
            try:
                out.append((int(pt["t"]), float(pt["p"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out


class ComboClient:
    """Combo RFQ catalog — which markets can be combined (public)."""

    def __init__(self, base_url: str = COMBO_API, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ComboClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def iter_combo_markets(self) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            response = self._client.get("/v1/rfq/combo-markets", params=params)
            response.raise_for_status()
            payload = response.json()
            yield from payload.get("markets", [])
            cursor = payload.get("next_cursor")
            if not cursor:
                break
            time.sleep(0.05)


def parse_json_field(value: Any, default: Any = None) -> Any:
    """Gamma often returns JSON-encoded strings for list fields."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default
