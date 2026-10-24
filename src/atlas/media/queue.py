"""The client work queue `social.media_requests`: clients only say WHICH entity they want (a contract key). Rows are read with the
publishable key, never trusted beyond the key itself, and the pipeline re-resolves every asset from the authoritative chain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import MediaQueueCfg
from ..http import HttpError, PoliteClient
from . import keys as K

log = logging.getLogger("atlas.media.queue")

CURSOR_META = "media_queue_cursor"


class QueueUnavailable(RuntimeError):
    """The queue could not be read (table not created yet, bad key, outage). The run goes on without it."""


@dataclass(frozen=True)
class QueueRow:
    id: int
    kind: str
    key: str

    @property
    def valid(self) -> bool:
        return self.kind in ("author", "institution") and K.valid_key(self.kind, self.key)


def fetch_rows(cfg: MediaQueueCfg, after_id: int, limit: int, client: PoliteClient) -> list[QueueRow]:
    """Rows with id > after_id, oldest first. Malformed rows come back with valid=False so the cursor can step over them."""
    if not (cfg.url and cfg.publishable_key):
        raise QueueUnavailable("media.queue.url / publishable_key not configured")
    headers = {"apikey": cfg.publishable_key, "Accept-Profile": cfg.schema_name, "Accept": "application/json"}
    if not cfg.publishable_key.startswith("sb_publishable_"):   # legacy anon JWT keys also travel as a bearer token
        headers["Authorization"] = f"Bearer {cfg.publishable_key}"
    params = {"select": "id,entity_kind,entity_key,created_at", "id": f"gt.{after_id}", "order": "id.asc", "limit": str(limit)}
    try:
        resp = client.get(f"{cfg.url.rstrip('/')}/rest/v1/{cfg.table}", params=params, headers=headers)
    except HttpError as exc:
        raise QueueUnavailable(str(exc)) from exc
    if resp.status_code != 200:
        raise QueueUnavailable(f"HTTP {resp.status_code}")   # the body can echo request details; keep it out of logs
    try:
        data: Any = resp.json()
    except ValueError as exc:
        raise QueueUnavailable("not JSON") from exc
    rows: list[QueueRow] = []
    for r in data if isinstance(data, list) else []:
        if isinstance(r, dict) and isinstance(r.get("id"), int):
            rows.append(QueueRow(r["id"], str(r.get("entity_kind")), str(r.get("entity_key"))))
    return sorted(rows, key=lambda r: r.id)


def advance_cursor(rows: list[QueueRow], handled: set[int], cursor: int) -> int:
    """Highest row id such that every row up to it was handled: rows we could not get to (budget) stay for the next run."""
    for r in sorted(rows, key=lambda r: r.id):
        if r.id not in handled:
            break
        cursor = r.id
    return cursor
