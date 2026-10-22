"""One ingestion run: harvest -> HF signals -> OpenAlex -> briefs -> thumbnails -> build -> validate.
Every stage is wrapped so a failure in one source degrades the output instead of killing the run."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import arxiv, hf as hf_mod
from .briefs import BriefResult, BriefStats, generate_briefs, resolve_provider
from .build import SiteBuilder
from .config import Config
from .http import PoliteClient
from .licenses import allows_redistribution
from .models import ts
from .openalex import OpenAlexClient
from .scoring import score_paper
from .store import Store, StoredPaper
from .thumbs import ThumbsUnavailable, make_thumbnail, thumb_path
from .validate import validate_site

log = logging.getLogger("atlas.pipeline")


@dataclass
class RunOptions:
    site_dir: Path = Path("site")
    state_dir: Path = Path("state")
    limit: int | None = None
    backfill_days: int | None = None
    skip_harvest: bool = False
    skip_hf: bool = False
    skip_openalex: bool = False
    skip_thumbs: bool = False
    skip_briefs: bool = False
    now: datetime | None = None


@dataclass
class RunResult:
    stats: dict[str, Any] = field(default_factory=dict)
    ok: bool = True


def git_sha7() -> str:
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env[:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# ------------------------------------------------------------------ ranking helper (pre-build)


def rank_day(cfg: Config, papers: list[StoredPaper], hf: dict[str, tuple[hf_mod.HfSignal, str]], day: str) -> list[StoredPaper]:
    """Papers of one feed day, best first (same formula the shards use)."""
    def key(sp: StoredPaper) -> tuple[float, float, str]:
        sig = hf.get(sp.record.arxiv_id, (None, ""))[0]
        pub = datetime.fromisoformat(sp.api.published.replace("Z", "+00:00"))
        age = max(0.0, (date.fromisoformat(day) - pub.date()).days)
        s = score_paper(cfg.scoring, hf_upvotes=sig.upvotes if sig else None, github_stars=sig.github_stars if sig else None,
                        n_categories=len(set(sp.record.categories)), age_days=age, has_code=bool(sig and sig.github_repo))
        return (-s, -pub.timestamp(), sp.record.arxiv_id)
    return sorted(papers, key=key)


# ------------------------------------------------------------------ stages


def stage_harvest(store: Store, cfg: Config, client: PoliteClient, today: date, opts: RunOptions, now: datetime) -> dict[str, Any]:
    a = cfg.arxiv
    cursor = store.get_meta("harvest_cursor")
    default_from = today - timedelta(days=1)
    if cursor:
        start = max(date.fromisoformat(cursor), today - timedelta(days=a.max_catchup_days))
    else:
        start = default_from - timedelta(days=a.cold_start_backfill_days)
    if opts.backfill_days is not None:
        start = min(start, default_from - timedelta(days=opts.backfill_days))
    stats: dict[str, Any] = {"from": start.isoformat(), "cursor_before": cursor, "sets": {}, "records_seen": 0,
                             "candidates": 0, "verified_new": 0, "added": 0, "rejected_revisions": 0, "complete": True,
                             "errors": []}
    candidates: dict[str, arxiv.OaiRecord] = {}
    limit_candidates = opts.limit * 4 if opts.limit else None

    def stop() -> bool:
        return limit_candidates is not None and len(candidates) >= limit_candidates

    for set_spec in a.sets:
        if stop():
            stats["complete"] = False
            break
        seen = 0
        try:
            for rec in arxiv.harvest(client, a.oai_base, set_spec, start.isoformat(), stop=stop):
                seen += 1
                if rec.arxiv_id in candidates or store.has_paper(rec.arxiv_id) or store.is_rejected(rec.arxiv_id, rec.datestamp):
                    continue
                if arxiv.is_recent_id(rec.arxiv_id, rec.datestamp) and arxiv.matches_categories(rec.categories, a.categories):
                    candidates[rec.arxiv_id] = rec
        except Exception as exc:
            stats["complete"] = False
            stats["errors"].append(f"{set_spec}: {type(exc).__name__}: {exc}")
            log.exception("Harvest of set %s failed; continuing with the other sets", set_spec)
        stats["sets"][set_spec] = seen
        stats["records_seen"] += seen
    if stop():
        stats["complete"] = False
    if limit_candidates and len(candidates) > limit_candidates:  # smoke-test runs: do not verify a whole OAI page
        candidates = dict(list(candidates.items())[:limit_candidates])
    stats["candidates"] = len(candidates)

    if candidates:
        try:
            entries = arxiv.verify_entries(client, a.api_base, list(candidates), a.verify_batch)
        except Exception as exc:
            stats["complete"] = False
            stats["errors"].append(f"verify: {type(exc).__name__}: {exc}")
            log.exception("API verification failed; adding nothing this run")
            entries = {}
        accepted: list[tuple[arxiv.OaiRecord, arxiv.ApiEntry]] = []
        rejected_rows: list[tuple[str, str]] = []
        for aid, rec in candidates.items():
            entry = entries.get(aid)
            if entry is None:
                continue  # withdrawn or API hiccup: try again next run
            if arxiv.is_new_paper(entry, rec.datestamp, a.new_paper_max_age_days):
                accepted.append((rec, entry))
            else:
                rejected_rows.append((aid, rec.datestamp))
        store.add_rejected(rejected_rows)
        stats["rejected_revisions"] = len(rejected_rows)
        stats["verified_new"] = len(accepted)
        accepted.sort(key=lambda t: (t[0].datestamp, t[0].arxiv_id), reverse=True)
        if opts.limit and len(accepted) > opts.limit:
            accepted = accepted[: opts.limit]
            stats["complete"] = False
        for rec, entry in accepted:
            if store.add_paper(rec, entry, rec.datestamp, ts(now)):
                stats["added"] += 1
    if stats["complete"]:
        store.set_meta("harvest_cursor", today.isoformat())
    stats["arxiv_requests"] = client.requests_made
    return stats


def stage_hf(store: Store, cfg: Config, client: PoliteClient, today: date, now: datetime) -> dict[str, Any]:
    signals = hf_mod.fetch_recent(client, cfg.hf.api_base, today, cfg.hf.days, cfg.hf.limit, cfg.hf.max_pages)
    known = store.known_ids(s.arxiv_id for s in signals)
    mine = [s for s in signals if s.arxiv_id in known]
    store.upsert_hf(mine, ts(now))
    return {"fetched": len(signals), "matched_in_state": len(mine), "requests": client.requests_made}


def stage_openalex(store: Store, cfg: Config, oa: OpenAlexClient, today: date, now: datetime) -> dict[str, Any]:
    c = cfg.openalex
    stats: dict[str, Any] = {"lookups": 0, "found": 0, "missing": 0, "errors": 0, "authors_fetched": 0, "targets": 0}
    window_start = (today - timedelta(days=c.retry_days)).isoformat()
    hf = store.hf_signals()
    rows = store.oa_work_rows()
    targets: list[str] = []
    for day in [d for d in store.feed_dates() if d >= window_start]:
        ranked = rank_day(cfg, store.papers_between(day, day), hf, day)[: c.enrich_top_n]
        targets.extend(sp.record.arxiv_id for sp in ranked)
    stats["targets"] = len(targets)
    for aid in targets:
        row = rows.get(aid)
        if row and row["status"] == "found":
            continue
        if row and row["status"] == "missing":
            last = datetime.fromisoformat(row["last_attempt"].replace("Z", "+00:00"))
            if (now - last) < timedelta(hours=c.retry_after_hours):
                continue
        if not oa.active:
            break
        res = oa.get_work(aid)
        stats["lookups"] += 1
        if res.status == "found":
            store.set_oa_work(aid, "found", res.work, ts(now))
            stats["found"] += 1
        elif res.status == "missing":
            store.set_oa_work(aid, "missing", None, ts(now))
            stats["missing"] += 1
        else:
            stats["errors"] += 1
    # author cards for every found work in the window
    rows = store.oa_work_rows()
    cards = store.author_rows()
    ttl = timedelta(days=c.author_ttl_days)
    need: list[str] = []
    for aid, row in rows.items():
        if row["status"] != "found" or not row["work"]:
            continue
        for au in row["work"]["authorships"][: c.max_authors_per_paper]:
            oid = au.get("id")
            if not oid or oid in need:
                continue
            have = cards.get(oid)
            if have is None:
                need.append(oid)
                continue
            age = now - datetime.fromisoformat(have[1].replace("Z", "+00:00"))
            # ids OpenAlex could not resolve (new/merged authors) are retried daily, resolved cards weekly
            if age > (timedelta(days=1) if have[0].get("missing") else ttl):
                need.append(oid)
    if need and oa.active:
        fetched = oa.get_authors(need)
        store.upsert_authors(fetched, ts(now))
        unresolved = {oid: {"id": oid, "missing": True} for oid in oa.answered_ids if oid not in fetched}
        store.upsert_authors(unresolved, ts(now))
        stats["authors_fetched"] = len(fetched)
        stats["authors_unresolved"] = len(unresolved)
        stats["authors_requested"] = len(need)
    rows = store.oa_work_rows()
    uniq = set(targets)
    found_targets = sum(1 for aid in uniq if rows.get(aid, {}).get("status") == "found")
    stats["targets_unique"] = len(uniq)
    stats["targets_found"] = found_targets
    stats["targets_found_pct"] = round(100 * found_targets / len(uniq), 1) if uniq else 0.0
    stats.update(
        credits_used=oa.credits_used, remaining_start=oa.first_remaining, remaining_end=oa.remaining,
        limit=oa.limit, stopped_reason=oa.stopped_reason, requests=oa.requests,
    )
    return stats


def stage_briefs(store: Store, cfg: Config, today: date, now: datetime) -> dict[str, Any]:
    resolved = resolve_provider(cfg.briefs)
    if resolved is None:
        return {"enabled": False}
    provider, pcfg = resolved
    hf = store.hf_signals()
    cached = {k for k, v in store.brief_rows().items() if v["source"] == "llm"}
    queue: list[tuple[str, str, str]] = []
    for day in sorted([d for d in store.feed_dates() if d >= (today - timedelta(days=2)).isoformat()], reverse=True):
        for sp in rank_day(cfg, store.papers_between(day, day), hf, day)[: cfg.briefs.top_n]:
            queue.append((sp.record.arxiv_id, sp.record.title, sp.record.abstract))

    def save(aid: str, res: BriefResult, cost: float) -> None:
        store.set_brief(aid, res.text[:600], "llm", provider.model, ts(now), cost)

    st: BriefStats = generate_briefs(provider, pcfg, queue, cached, max_cost_usd=cfg.briefs.max_cost_usd_per_run,
                                     max_output_tokens=cfg.briefs.max_output_tokens, save=save)
    return {"enabled": True, "provider": provider.name, "model": provider.model, "generated": st.generated,
            "cached": st.skipped_cached, "failed": st.failed, "cost_usd": round(st.cost_usd, 5), "stopped_reason": st.stopped_reason}


def stage_thumbs(store: Store, cfg: Config, client: PoliteClient, thumbs_dir: Path, today: date, now: datetime) -> dict[str, Any]:
    t = cfg.thumbs
    stats: dict[str, Any] = {"eligible": 0, "made": 0, "failed": 0, "skipped_existing": 0, "bytes": 0}
    hf = store.hf_signals()
    rows = store.thumb_rows()
    todo: list[StoredPaper] = []
    for day in sorted([d for d in store.feed_dates() if d >= (today - timedelta(days=cfg.feeds.retention_days)).isoformat()], reverse=True):
        for sp in rank_day(cfg, store.papers_between(day, day), hf, day)[: t.top_n]:
            if not allows_redistribution(sp.record.license, t.allowed_license_prefixes):
                continue
            stats["eligible"] += 1
            row = rows.get(sp.record.arxiv_id)
            if row and row["status"] == "ok" and thumb_path(thumbs_dir, sp.record.arxiv_id).exists():
                stats["skipped_existing"] += 1
                continue
            if row and row["status"] == "failed":
                age = now - datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
                if age < timedelta(days=t.retry_failed_after_days):
                    continue
            todo.append(sp)
    for sp in todo[: t.max_per_run]:
        aid = sp.record.arxiv_id
        try:
            n = make_thumbnail(client, cfg.arxiv.pdf_base, aid, thumb_path(thumbs_dir, aid), width_px=t.width_px,
                               quality=t.jpeg_quality, max_pdf_bytes=t.max_pdf_mb * 1024 * 1024)
            store.set_thumb(aid, "ok", n, ts(now))
            stats["made"] += 1
            stats["bytes"] += n
        except ThumbsUnavailable as exc:
            stats["unavailable"] = str(exc)
            log.warning("Thumbnails disabled: %s", exc)
            break
        except Exception as exc:
            store.set_thumb(aid, "failed", None, ts(now))
            stats["failed"] += 1
            log.warning("Thumbnail for %s failed: %s", aid, exc)
    stats["deferred"] = max(0, len(todo) - t.max_per_run)
    stats["requests"] = client.requests_made
    return stats


# ------------------------------------------------------------------ run


def run(cfg: Config, opts: RunOptions, *, arxiv_client: PoliteClient | None = None, hf_client: PoliteClient | None = None,
        oa_client: OpenAlexClient | None = None) -> RunResult:
    t0 = time.time()
    now = opts.now or datetime.now(timezone.utc)
    today = now.date()
    state_dir, site_dir = Path(opts.state_dir), Path(opts.site_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = state_dir / "thumbs"
    store = Store(state_dir / "atlas.sqlite")
    cold_start = store.is_empty()
    result = RunResult(stats={"started_at": ts(now), "git_sha": git_sha7(), "cold_start": cold_start,
                              "limit": opts.limit, "stages": {}})
    st = result.stats["stages"]

    own_arxiv = arxiv_client is None
    arxiv_client = arxiv_client or PoliteClient(cfg.user_agent, cfg.arxiv.min_interval_s, max_retries=cfg.arxiv.max_retries,
                                                max_retry_after_s=cfg.arxiv.max_retry_after_s)
    own_hf = hf_client is None
    hf_client = hf_client or PoliteClient(cfg.user_agent, cfg.hf.min_interval_s, max_retries=2, max_retry_after_s=30)
    own_oa = oa_client is None
    oa = oa_client or OpenAlexClient(cfg.openalex, cfg.user_agent)

    def stage(name: str, fn, *args) -> None:
        t = time.time()
        try:
            st[name] = fn(*args)
        except Exception as exc:
            log.exception("Stage %s failed; continuing", name)
            st[name] = {"error": f"{type(exc).__name__}: {exc}"}
        st[name]["seconds"] = round(time.time() - t, 1)

    try:
        if not opts.skip_harvest:
            stage("harvest", stage_harvest, store, cfg, arxiv_client, today, opts, now)
        if cfg.hf.enabled and not opts.skip_hf:
            stage("hf", stage_hf, store, cfg, hf_client, today, now)
        if cfg.openalex.enabled and not opts.skip_openalex:
            stage("openalex", stage_openalex, store, cfg, oa, today, now)
        if not opts.skip_briefs:
            stage("briefs", stage_briefs, store, cfg, today, now)
        if cfg.thumbs.enabled and not opts.skip_thumbs:
            stage("thumbs", stage_thumbs, store, cfg, arxiv_client, thumbs_dir, today, now)

        # retention: keep `retention_days` of feed days
        latest = (store.feed_dates() or [today.isoformat()])[-1]
        keep_from = (date.fromisoformat(latest) - timedelta(days=cfg.feeds.retention_days - 1)).isoformat()
        for aid in store.prune(keep_from):
            thumb_path(thumbs_dir, aid).unlink(missing_ok=True)

        t_build = time.time()
        builder = SiteBuilder(store, cfg, site_dir, thumbs_dir, now)
        bstats = builder.build()
        result.stats["stages"]["build"] = bstats.__dict__ | {"seconds": round(time.time() - t_build, 1)}
        report = validate_site(site_dir, cfg.output.max_site_mb)
        result.stats["validation"] = {"ok": report.ok, "checked": report.checked, "errors": report.errors[:20],
                                      "site_mb": round(report.site_bytes / 1e6, 2)}
        result.stats["totals"] = {
            "papers_in_state": store.count_papers(),
            "papers_in_window": bstats.papers_in_window,
            "with_openalex_work": bstats.papers_with_openalex,
            "enriched_pct": round(100 * bstats.papers_with_openalex / bstats.papers_in_window, 1) if bstats.papers_in_window else 0.0,
            "with_hf_signal": bstats.papers_with_hf,
            "with_brief": bstats.papers_with_brief,
            "with_author_ids": bstats.papers_with_author_ids,
        }
        result.ok = report.ok
        result.stats["duration_s"] = round(time.time() - t0, 1)
        builder.write_status(result.stats)
        # status.json embeds run stats; re-validate that one file against its schema
        final = validate_site(site_dir, cfg.output.max_site_mb)
        if not final.ok:
            result.ok = False
            result.stats["validation"]["errors"] = final.errors[:20]
            result.stats["validation"]["ok"] = False
    finally:
        store.close()
        if own_arxiv:
            arxiv_client.close()
        if own_hf:
            hf_client.close()
        if own_oa:
            oa.close()
    result.stats["duration_s"] = round(time.time() - t0, 1)
    return result
