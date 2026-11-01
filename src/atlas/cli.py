"""Command line: `atlas run`, `atlas validate`, `atlas media ...`, `atlas audio ...`, `atlas state ...`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .audio import publish as audio_publish
from .audio import stage as audio_stage
from .audio.plan import DayPlan
from .audio.render import render as audio_render
from .config import load_config
from .media import publish as media_publish
from .media.cascade import Cascade
from .media.papers import ingest_thumbnails
from .media.sources import MediaHttp
from .media.stage import process_targets, state_targets
from .media.store import MediaStore
from .models import ts
from .pipeline import RunOptions, git_sha7, rank_day, run
from .snapshot import restore as state_restore, snapshot as state_snapshot
from .store import Store
from .validate import validate_site


def _int_or_none(v: str | None) -> int | None:
    return int(v) if v not in (None, "") else None


def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # request URLs may carry query params; keep them out of logs


def _summary_markdown(stats: dict) -> str:
    st = stats.get("stages", {})
    h, o, t, b, md = st.get("harvest", {}), st.get("openalex", {}), st.get("thumbs", {}), st.get("briefs", {}), st.get("media", {})
    au, aus = st.get("audio", {}), st.get("audio_scripts", {})
    tot = stats.get("totals", {})
    rows = [
        ("Papers added this run", h.get("added")), ("Papers in window", tot.get("papers_in_window")),
        ("With OpenAlex work (window)", f"{tot.get('with_openalex_work')} ({tot.get('enriched_pct')}%)"),
        ("Top-N targets enriched", f"{o.get('targets_found')}/{o.get('targets_unique')} ({o.get('targets_found_pct')}%)"),
        ("Authors fetched", o.get("authors_fetched")), ("OpenAlex credits used", o.get("credits_used")),
        ("OpenAlex remaining", o.get("remaining_end")), ("OpenAlex stopped", o.get("stopped_reason")),
        ("Thumbnails made", t.get("made")), ("Briefs", b.get("generated", "off")),
        ("Media entities resolved", f"{md.get('processed_author', 0)} authors, {md.get('processed_institution', 0)} institutions"
                                    if md and "error" not in md else md.get("error", "off")),
        ("Media found / nothing", f"{md.get('found', 0)} / {md.get('negative', 0)}" if md and "error" not in md else None),
        ("Media queue", md.get("queue") if md else None),
        ("Audio clips (built / reused / failed)", f"{au.get('built', 0)} / {au.get('reused', 0)} / {au.get('failed', 0)}" if au and "built" in au else
                                                  (au.get("skipped") or au.get("error") or aus.get("error") or "off") if (au or aus) else "off"),
        ("Audio timing (aligned / estimated), RTF", f"{au.get('aligned', 0)} / {au.get('estimated', 0)}, {au.get('rtf')}" if au and "built" in au else None),
        ("Audio published", f"{tot.get('audio_clips_published')} clips, {tot.get('audio_mb_published')} MB, {tot.get('audio_days_published')} days" if tot.get("audio_clips_published") else None),
        ("Site size (MB)", stats.get("validation", {}).get("site_mb")), ("Validation ok", stats.get("validation", {}).get("ok")),
        ("Duration (s)", stats.get("duration_s")),
    ]
    return "## Atlas run\n\n| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n"


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    now = datetime.now(timezone.utc)
    state_dir = Path(args.state)
    limit = args.limit if args.limit is not None else _int_or_none(os.environ.get("ATLAS_LIMIT"))
    backfill = args.backfill_days if args.backfill_days is not None else _int_or_none(os.environ.get("ATLAS_BACKFILL_DAYS"))
    run_key = f"limit{limit}" if limit else "daily"
    stem = f"atlas_ingest_{run_key}_{git_sha7()}_{now:%Y%m%d_%H%M%S}"
    runs_dir = state_dir / "runs"
    if runs_dir.exists():  # keep two weeks of run logs/sidecars
        for old in runs_dir.iterdir():
            if old.is_file() and now.timestamp() - old.stat().st_mtime > 14 * 86400:
                old.unlink(missing_ok=True)
    _setup_logging(runs_dir / f"{stem}.log")
    log = logging.getLogger("atlas")
    meta_path = runs_dir / f"{stem}.meta.json"
    meta = {
        "run_tag": f"{now:%Y%m%d_%H%M%S}", "git_sha": git_sha7(), "hostname": socket.gethostname(), "script": "atlas.cli run",
        "params": {"limit": limit, "backfill_days": backfill, "config": str(args.config or "config.yaml"),
                   "offline": args.offline},
        "experiment_mode": "ci" if os.environ.get("GITHUB_ACTIONS") else "local",
        "completed": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    opts = RunOptions(
        site_dir=Path(args.site), state_dir=state_dir, limit=limit, backfill_days=backfill, now=now,
        skip_harvest=args.offline or args.skip_harvest, skip_hf=args.offline or args.skip_hf,
        skip_openalex=args.offline or args.skip_openalex, skip_thumbs=args.offline or args.skip_thumbs,
        skip_briefs=args.offline or args.skip_briefs, skip_media=args.offline or args.skip_media,
        skip_audio=args.offline or args.skip_audio, media_dir=Path(args.media_dir), audio_dir=Path(args.audio_dir),
    )
    log.info("Atlas run %s (limit=%s backfill=%s)", stem, limit, backfill)
    result = run(cfg, opts)
    meta.update(stats=result.stats, completed=result.ok)
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Done in %ss; validation ok=%s", result.stats.get("duration_s"), result.ok)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(_summary_markdown(result.stats))
    print(json.dumps({"ok": result.ok, "totals": result.stats.get("totals"), "validation": result.stats.get("validation")}, indent=2))
    return 0 if result.ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    _setup_logging(None)
    cfg = load_config(args.config)
    report = validate_site(Path(args.site), cfg.output.max_site_mb, media_max_mb=cfg.media.max_site_mb, audio_max_mb=cfg.audio.max_site_mb)
    print(json.dumps({"ok": report.ok, "checked": report.checked, "site_mb": round(report.site_bytes / 1e6, 2), "errors": report.errors[:30]}, indent=2))
    return 0 if report.ok else 1


def cmd_media_backfill(args: argparse.Namespace) -> int:
    """Resolve every author and institution already in a state DB. Resumable: finished entities are skipped by their TTL."""
    _setup_logging(None)
    cfg = load_config(args.config)
    now = datetime.now(timezone.utc)
    db = Path(args.from_db)
    if not db.exists():
        print(f"no such database: {db}", file=sys.stderr)
        return 2
    state, media, http = Store(db), MediaStore(args.media_dir, cfg.site_base_url), MediaHttp.create(cfg)
    try:
        cascade = Cascade(cfg, http, media, ts(now), state.media_etags())
        targets = state_targets(state, cfg.openalex.max_authors_per_paper, include_orphans=True)
        done = process_targets(cfg, state, media, cascade, targets, now, args.limit)
        thumbs = ingest_thumbnails(state, media, db.parent / "thumbs", cfg, ts(now))
        manifest = media.write_manifest(ts(now))
    finally:
        http.close()
        state.close()
    print(json.dumps({"targets": len(targets), **dict(done.stats), "budget_exhausted": done.budget_exhausted, "notes": dict(done.notes),
                      "cascade": dict(cascade.stats), "requests": http.requests_made, "thumbnails": thumbs, "manifest": manifest}, indent=2))
    return 0


def cmd_media_checkout(args: argparse.Namespace) -> int:
    _setup_logging(None)
    print(json.dumps({"checkout": media_publish.checkout_branch(Path(args.repo), Path(args.media_dir), args.branch, args.remote)}))
    return 0


def cmd_media_publish(args: argparse.Namespace) -> int:
    """Commit the new files of the media checkout as `chore(media): <counts>` and push HEAD to the branch."""
    _setup_logging(None)
    cfg = load_config(args.config)
    if not (Path(args.media_dir) / ".git").exists():
        print(json.dumps({"committed": False, "reason": f"{args.media_dir} is not a checkout of the media branch (run `atlas media checkout`)"}))
        return 1
    MediaStore(args.media_dir, cfg.site_base_url).write_manifest(ts(datetime.now(timezone.utc)))
    try:
        out = media_publish.commit_and_push(Path(args.media_dir), args.branch, args.remote, push=not args.no_push)
    except subprocess.CalledProcessError as exc:   # most likely a rejected push: say so briefly instead of a traceback
        lines = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
        key = [ln.strip() for ln in lines if any(w in ln for w in ("rejected", "error", "fatal"))]
        print(json.dumps({"committed": True, "pushed": False, "error": "; ".join(key or lines[-2:])[:300]}))
        return 1
    print(json.dumps(out, indent=2))
    return 0


# ------------------------------------------------------------------ atlas audio ...


def _audio_cfg(args: argparse.Namespace):
    """The config with the command-line overrides of the audio commands applied."""
    cfg = load_config(args.config)
    update = {k: v for k, v in (("align", getattr(args, "align", None)), ("max_papers", getattr(args, "limit", None))) if v}
    return cfg.model_copy(update={"audio": cfg.audio.model_copy(update=update)}) if update else cfg


def cmd_audio_run(args: argparse.Namespace) -> int:
    """Audio scripts + synthesis for one day of an existing state DB (what `atlas run` does after the briefs), in this process. Never fails the shell."""
    _setup_logging(None)
    cfg, now = _audio_cfg(args), datetime.now(timezone.utc)
    db = Path(args.state) / "atlas.sqlite"
    if not db.exists():
        print(json.dumps({"ok": False, "error": f"no state database at {db}"}))
        return 2
    store = Store(db)
    try:
        day = args.date or (store.feed_dates() or [None])[-1]
        if day is None:
            print(json.dumps({"ok": False, "error": "the state has no feed days"}))
            return 2
        ranked = rank_day(cfg, store.papers_between(day, day), store.hf_signals(), day)
        scripts = audio_stage.stage_audio_scripts(store, cfg, ranked, day, now, Path(args.state), resolved=None if args.no_llm else True)
    finally:
        store.close()
    synth = audio_stage.stage_audio(cfg, day, Path(args.state), Path(args.audio_dir), now, isolate=args.isolate)
    print(json.dumps({"scripts": scripts, "synth": synth}, indent=2))
    return 0


def cmd_audio_synth(args: argparse.Namespace) -> int:
    """Speak a plan file into the audio store. This is the child process of the pipeline's `audio` stage; stats go to --stats-out as JSON."""
    _setup_logging(None)
    cfg = load_config(args.config)
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
    plan = DayPlan.load(Path(args.plan))
    try:
        stats = audio_stage.run_synth(cfg, plan, Path(args.audio_dir), now).as_dict()
        code = 0
    except Exception as exc:
        logging.getLogger("atlas.audio").exception("audio synthesis failed")
        stats, code = {"date": plan.date, "error": f"{type(exc).__name__}: {exc}"}, 1
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return code


def cmd_audio_checkout(args: argparse.Namespace) -> int:
    _setup_logging(None)
    print(json.dumps({"checkout": audio_publish.checkout(Path(args.repo), Path(args.audio_dir), args.branch, args.remote)}))
    return 0


def cmd_audio_publish(args: argparse.Namespace) -> int:
    """Commit the audio checkout as ONE parentless commit and force-push it to the audio branch."""
    _setup_logging(None)
    if not (Path(args.audio_dir) / ".git").exists():
        print(json.dumps({"committed": False, "reason": f"{args.audio_dir} is not a checkout of the audio branch (run `atlas audio checkout`)"}))
        return 1
    try:
        out = audio_publish.commit_and_push(Path(args.audio_dir), args.branch, args.remote, push=not args.no_push)
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
        key = [ln.strip() for ln in lines if any(w in ln for w in ("rejected", "error", "fatal"))]
        print(json.dumps({"committed": True, "pushed": False, "error": "; ".join(key or lines[-2:])[:300]}))
        return 1
    print(json.dumps(out, indent=2))
    return 0


def cmd_audio_render(args: argparse.Namespace) -> int:
    """Manifest + clips for any list of Paper JSON objects, offline."""
    _setup_logging(None)
    cfg = _audio_cfg(args)
    out = audio_render(cfg, Path(args.papers), args.date, Path(args.out), limit=args.limit, apply_selection=args.select, llm=args.llm,
                       authors_from=args.authors_from, state_db=Path(args.state_db) if args.state_db else None)
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


def cmd_state_snapshot(args: argparse.Namespace) -> int:
    _setup_logging(None)
    print(json.dumps(state_snapshot(Path(args.state) / "atlas.sqlite", Path(args.repo), branch=args.branch, remote=args.remote, push=not args.no_push)))
    return 0   # durability must never fail a run


def cmd_state_restore(args: argparse.Namespace) -> int:
    _setup_logging(None)
    print(json.dumps(state_restore(Path(args.state) / "atlas.sqlite", Path(args.repo), branch=args.branch, remote=args.remote, force=args.force)))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="atlas", description="ScienceOS Atlas daily feed")
    p.add_argument("--config", default=None, help="config.yaml path (default: repo config.yaml or $ATLAS_CONFIG)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest, build the site, validate it")
    r.add_argument("--limit", type=int, default=None, help="keep at most N new papers this run (smoke test)")
    r.add_argument("--backfill-days", type=int, default=None, help="also harvest this many extra days back")
    r.add_argument("--site", default="site")
    r.add_argument("--state", default="state")
    r.add_argument("--media-dir", default="media-store", help="the media store; the `media` branch checkout in CI (default: %(default)s)")
    r.add_argument("--audio-dir", default="audio-store", help="the audio store; the `audio` branch checkout in CI (default: %(default)s)")
    r.add_argument("--offline", action="store_true", help="no network: rebuild site/ from state/ only")
    for s in ("harvest", "hf", "openalex", "thumbs", "briefs", "media", "audio"):
        r.add_argument(f"--skip-{s}", action="store_true")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("validate", help="validate an existing site/ directory")
    v.add_argument("--site", default="site")
    v.set_defaults(fn=cmd_validate)
    m = sub.add_parser("media", help="the central media store").add_subparsers(dest="media_cmd", required=True)
    mb = m.add_parser("backfill", help="resolve every author and institution already in a state DB")
    mb.add_argument("--from-db", required=True, help="state/atlas.sqlite")
    mb.add_argument("--media-dir", default="media-store", help="default: %(default)s")
    mb.add_argument("--limit", type=int, default=None, help="resolve at most N entities (default: all)")
    mb.set_defaults(fn=cmd_media_backfill)
    mc = m.add_parser("checkout", help="check the media branch out into --media-dir (orphan branch on the first run)")
    mp = m.add_parser("publish", help="commit new media files as chore(media): <counts> and push HEAD to the branch")
    for q in (mc, mp):
        q.add_argument("--media-dir", default="media-store", help="default: %(default)s")
        q.add_argument("--branch", default="media")
        q.add_argument("--remote", default="origin")
    mc.add_argument("--repo", default=".", help="the repository the worktree is added to")
    mc.set_defaults(fn=cmd_media_checkout)
    mp.add_argument("--no-push", action="store_true", help="commit only")
    mp.set_defaults(fn=cmd_media_publish)
    a = sub.add_parser("audio", help="the daily audio brief (Pocket TTS narration)").add_subparsers(dest="audio_cmd", required=True)
    ar = a.add_parser("run", help="audio scripts + synthesis for one day of a state DB, in this process")
    ar.add_argument("--state", default="state")
    ar.add_argument("--audio-dir", default="audio-store", help="default: %(default)s")
    ar.add_argument("--date", default=None, help="feed date (default: the newest day in the state)")
    ar.add_argument("--limit", type=int, default=None, help="at most N papers (overrides audio.max_papers)")
    ar.add_argument("--align", choices=["whisper", "heuristic"], default=None)
    ar.add_argument("--no-llm", action="store_true", help="fallback summaries only (brief + first abstract sentence), no API call")
    ar.add_argument("--isolate", action="store_true", help="synthesise in a child process, as the pipeline does")
    ar.set_defaults(fn=cmd_audio_run)
    asy = a.add_parser("synth", help="speak a plan file into the audio store (the child process of the pipeline's audio stage)")
    asy.add_argument("--plan", required=True)
    asy.add_argument("--audio-dir", default="audio-store")
    asy.add_argument("--stats-out", default=None)
    asy.add_argument("--now", default=None)
    asy.set_defaults(fn=cmd_audio_synth)
    ac = a.add_parser("checkout", help="check the audio branch out into --audio-dir (orphan branch on the first run)")
    ap = a.add_parser("publish", help="commit the audio store as ONE parentless commit and force-push it to the branch")
    for q in (ac, ap):
        q.add_argument("--audio-dir", default="audio-store", help="default: %(default)s")
        q.add_argument("--branch", default="audio")
        q.add_argument("--remote", default="origin")
    ac.add_argument("--repo", default=".", help="the repository the worktree is added to")
    ac.set_defaults(fn=cmd_audio_checkout)
    ap.add_argument("--no-push", action="store_true", help="make the commit object only")
    ap.set_defaults(fn=cmd_audio_publish)
    arn = a.add_parser("render", help="manifest + clips for a list of Paper JSON objects, offline (iPhone fixtures)")
    arn.add_argument("--papers", required=True, help="JSON: a list of Paper objects, {\"papers\": [...]}, or a feed shard")
    arn.add_argument("--date", required=True, help="feed date of the manifest, YYYY-MM-DD")
    arn.add_argument("--out", required=True, help="output directory: <date>.json + clips/")
    arn.add_argument("--limit", type=int, default=None, help="only the first N papers")
    arn.add_argument("--select", action="store_true", help="apply the daily selection (top N + top K per category) to the papers first")
    arn.add_argument("--llm", action="store_true", help="LLM spoken summaries (needs the provider's key in the environment); default: fallback text")
    arn.add_argument("--authors-from", default=None, help="site URL or directory with authors/<id>.json, to look up first-author institutions")
    arn.add_argument("--state-db", default=None, help="SQLite file that caches LLM summaries (default: in memory)")
    arn.add_argument("--align", choices=["whisper", "heuristic"], default=None)
    arn.set_defaults(fn=cmd_audio_render)
    sp = sub.add_parser("state", help="encrypted snapshot of state/atlas.sqlite on the `state` branch").add_subparsers(dest="state_cmd", required=True)
    for name, fn, text in (("snapshot", cmd_state_snapshot, "encrypt the DB and force-push it as the single commit of branch `state`"),
                           ("restore", cmd_state_restore, "restore the DB from branch `state` when it is missing")):
        q = sp.add_parser(name, help=text)
        q.add_argument("--state", default="state")
        q.add_argument("--repo", default=".")
        q.add_argument("--branch", default="state")
        q.add_argument("--remote", default="origin")
        q.set_defaults(fn=fn)
        if name == "snapshot":
            q.add_argument("--no-push", action="store_true")
        else:
            q.add_argument("--force", action="store_true", help="overwrite an existing database")
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
