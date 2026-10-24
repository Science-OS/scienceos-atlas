"""The media stage of `atlas run`: decide which entities to resolve (feed + client queue + failed retries), spend the per-run
budget, honour the TTLs kept in the SQLite state, and write the store. Nothing here ever deletes anything."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import zip_longest
from typing import Any

from ..config import Config
from ..models import ts
from ..store import Store
from . import keys as K
from .cascade import Cascade, Resolved
from .queue import CURSOR_META, QueueUnavailable, QueueRow, advance_cursor, fetch_rows
from .sources import MediaHttp, SourceError
from .store import MediaStore

log = logging.getLogger("atlas.media.stage")


@dataclass
class Target:
    kind: str                                   # author | institution
    ids: dict[str, str]                         # bare ids: orcid / wikidata / openalex, or ror / wikidata
    name: str | None = None                     # used only to sanity-check a Wikidata label found by id
    alt_names: list[str] = field(default_factory=list)
    website: str | None = None
    row_id: int | None = None                   # queue row this target came from

    @property
    def keys(self) -> list[str]:
        return K.entity_keys(self.kind, self.ids)


def target_from_key(kind: str, key: str, row_id: int | None = None) -> Target:
    return Target(kind, K.ids_from_keys(kind, [key]), row_id=row_id)


# ------------------------------------------------------------------ targets from the SQLite state


def author_target(card: dict[str, Any]) -> Target:
    return Target("author", {k: v for k, v in {"openalex": card.get("id"), "orcid": card.get("orcid")}.items() if v},
                  name=card.get("name"), alt_names=list(card.get("alternatives") or [])[:20])


def institution_target(inst: dict[str, Any]) -> Target:
    return Target("institution", {"ror": inst.get("ror")} if inst.get("ror") else {}, name=inst.get("name"))


def state_targets(state: Store, max_authors: int, order: list[str] | None = None, include_orphans: bool = False) -> list[Target]:
    """Authors (any of orcid/openalex) and institutions (ROR) of the works in `order` (arxiv ids, most important first;
    default: every work with an OpenAlex record). Each paper's authors come first, then the institutions on and behind them."""
    cards = {oid: c for oid, (c, _) in state.author_rows().items() if not c.get("missing")}
    works = state.oa_work_rows()
    ids = [a for a in (order if order is not None else sorted(works, reverse=True)) if works.get(a, {}).get("status") == "found"]
    authors: dict[str, Target] = {}
    insts: dict[str, Target] = {}
    out: list[Target] = []
    for aid in ids:
        paper_insts: list[dict[str, Any]] = []
        for au in (works[aid]["work"] or {}).get("authorships", [])[:max_authors]:
            card = cards.get(au.get("id") or "")
            if card and card["id"] not in authors:
                authors[card["id"]] = author_target(card)
                out.append(authors[card["id"]])
            paper_insts += au.get("institutions") or []
            paper_insts += (card or {}).get("institutions") or []
        for inst in paper_insts:
            if inst.get("ror") and inst["ror"] not in insts:
                insts[inst["ror"]] = institution_target(inst)
                out.append(insts[inst["ror"]])
    if include_orphans:
        for oid, card in cards.items():
            if oid not in authors:
                authors[oid] = author_target(card)
                out.append(authors[oid])
            for inst in card.get("institutions") or []:
                if inst.get("ror") and inst["ror"] not in insts:
                    insts[inst["ror"]] = institution_target(inst)
                    out.append(insts[inst["ror"]])
    return [t for t in out if t.keys]


# ------------------------------------------------------------------ processing


@dataclass
class Processed:
    stats: Counter = field(default_factory=Counter)
    notes: Counter = field(default_factory=Counter)
    handled_rows: set[int] = field(default_factory=set)
    budget_exhausted: bool = False


def expiry(cfg: Config, kind: str, res: Resolved | None, now: datetime) -> tuple[str, datetime]:
    """(status, expires_at): photos 30 d, logos 180 d, nothing found 7 d, failed or partial attempts retry after a day."""
    m = cfg.media
    if res is None:
        return "error", now + timedelta(hours=m.error_retry_hours)
    if res.partial:
        return "partial", now + timedelta(hours=m.error_retry_hours)
    if res.found:
        return "ok", now + timedelta(days=m.photo_ttl_days if kind == "author" else m.logo_ttl_days)
    return "negative", now + timedelta(days=m.negative_ttl_days)


def process_targets(cfg: Config, state: Store, media: MediaStore, cascade: Cascade, targets: Iterable[Target], now: datetime,
                    budget: int | None, max_seconds: float | None = None, clock: Callable[[], float] = time.monotonic) -> Processed:
    """Resolve targets in order until the entity budget or the time cap is spent (what is left stays untouched for the next run)."""
    out = Processed()
    t_end = clock() + max_seconds if max_seconds else None
    now_ts = ts(now)
    seen: set[tuple[str, str]] = set()
    spent = 0
    for t in targets:
        keys = t.keys
        handled = lambda: out.handled_rows.add(t.row_id) if t.row_id is not None else None  # noqa: E731
        if not keys:
            out.stats["invalid"] += 1
            handled()
            continue
        if any((t.kind, k) in seen for k in keys):
            out.stats["deduped"] += 1
            handled()
            continue
        checks = state.media_checks(t.kind, keys)
        if any(c["expires_at"] > now_ts for c in checks.values()):
            seen.update((t.kind, k) for k in keys)
            out.stats["fresh"] += 1
            handled()
            continue
        if (budget is not None and spent >= budget) or (t_end is not None and clock() > t_end):
            out.budget_exhausted = True
            break
        spent += 1
        res: Resolved | None = None
        try:
            res = cascade.resolve_author(t.ids, t.name, t.alt_names) if t.kind == "author" else cascade.resolve_institution(t.ids, t.website)
        except SourceError as exc:
            log.warning("%s %s: source failed: %s", t.kind, keys[0], exc)
            out.stats["errors"] += 1
        except Exception:  # noqa: BLE001 - one bad entity must not end the run
            log.exception("%s %s: unexpected failure", t.kind, keys[0])
            out.stats["errors"] += 1
        all_keys = list(dict.fromkeys([*keys, *(K.entity_keys(t.kind, res.ids) if res else [])]))
        if res is not None:
            for n in res.notes:
                out.notes[n] += 1
            if res.found:
                written = media.write_entry(t.kind, all_keys, res.ids, res.assets, now_ts)
                out.stats["files_written"] += sum(1 for v in written.values() if v != "unchanged")
            out.stats["found" if res.found else "negative"] += 1
        status, expires = expiry(cfg, t.kind, res, now)
        state.set_media_check(t.kind, all_keys, status, now_ts, ts(expires))
        seen.update((t.kind, k) for k in all_keys)
        out.stats[f"processed_{t.kind}"] += 1
        handled()
        if spent % 25 == 0:
            state.set_media_etags(cascade.etags, now_ts)   # checkpoint: a crashed run keeps what it fetched
    state.set_media_etags(cascade.etags, now_ts)
    return out


def _interleave(a: list[Target], b: list[Target]) -> list[Target]:
    return [x for pair in zip_longest(a, b) for x in pair if x is not None]


def run_media(cfg: Config, state: Store, media: MediaStore, http: MediaHttp, feed: list[Target], now: datetime) -> dict[str, Any]:
    """Retries due, then the queue and the feed alternately (neither starves the other), within `max_new_per_run`."""
    m = cfg.media
    now_ts = ts(now)
    cascade = Cascade(cfg, http, media, now_ts, state.media_etags())
    stats: dict[str, Any] = {}

    retries = [target_from_key(kind, key) for kind, key in state.media_retries(now_ts)]
    cursor = int(state.get_meta(CURSOR_META) or 0)
    rows: list[QueueRow] = []
    try:
        rows = fetch_rows(m.queue, cursor, m.requests_per_run, http.queue)
    except QueueUnavailable as exc:
        stats["queue"] = {"unavailable": str(exc)}
        log.warning("media queue unavailable: %s", exc)
    queue_targets = [target_from_key(r.kind, r.key, r.id) for r in rows if r.valid]
    invalid_rows = {r.id for r in rows if not r.valid}

    def rank(t: Target) -> tuple[int, str]:   # never-resolved entities first, then the longest-expired
        checks = state.media_checks(t.kind, t.keys)
        return (1, min(c["expires_at"] for c in checks.values())) if checks else (0, "")

    work = retries + _interleave(queue_targets, sorted(feed, key=rank))
    done = process_targets(cfg, state, media, cascade, work, now, m.max_new_per_run, m.max_seconds)

    if rows:
        new_cursor = advance_cursor(rows, done.handled_rows | invalid_rows, cursor)
        if new_cursor != cursor:
            state.set_meta(CURSOR_META, str(new_cursor))
        stats["queue"] = {"rows": len(rows), "invalid": len(invalid_rows), "cursor_before": cursor, "cursor_after": new_cursor,
                          "pending": sum(1 for r in rows if r.id > new_cursor)}
    stats.update(targets=len(work), retries=len(retries), budget=m.max_new_per_run, budget_exhausted=done.budget_exhausted,
                 **dict(done.stats), notes=dict(done.notes), cascade=dict(cascade.stats), requests=http.requests_made)
    return stats
