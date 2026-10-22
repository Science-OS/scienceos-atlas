"""Command line: `atlas run`, `atlas validate`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .pipeline import RunOptions, git_sha7, run
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
    h, o, t, b = st.get("harvest", {}), st.get("openalex", {}), st.get("thumbs", {}), st.get("briefs", {})
    tot = stats.get("totals", {})
    rows = [
        ("Papers added this run", h.get("added")), ("Papers in window", tot.get("papers_in_window")),
        ("With OpenAlex work (window)", f"{tot.get('with_openalex_work')} ({tot.get('enriched_pct')}%)"),
        ("Top-N targets enriched", f"{o.get('targets_found')}/{o.get('targets_unique')} ({o.get('targets_found_pct')}%)"),
        ("Authors fetched", o.get("authors_fetched")), ("OpenAlex credits used", o.get("credits_used")),
        ("OpenAlex remaining", o.get("remaining_end")), ("OpenAlex stopped", o.get("stopped_reason")),
        ("Thumbnails made", t.get("made")), ("Briefs", b.get("generated", "off")),
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
        skip_briefs=args.offline or args.skip_briefs,
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
    report = validate_site(Path(args.site), cfg.output.max_site_mb)
    print(json.dumps({"ok": report.ok, "checked": report.checked, "site_mb": round(report.site_bytes / 1e6, 2), "errors": report.errors[:30]}, indent=2))
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="atlas", description="ScienceOS Atlas daily feed")
    p.add_argument("--config", default=None, help="config.yaml path (default: repo config.yaml or $ATLAS_CONFIG)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest, build the site, validate it")
    r.add_argument("--limit", type=int, default=None, help="keep at most N new papers this run (smoke test)")
    r.add_argument("--backfill-days", type=int, default=None, help="also harvest this many extra days back")
    r.add_argument("--site", default="site")
    r.add_argument("--state", default="state")
    r.add_argument("--offline", action="store_true", help="no network: rebuild site/ from state/ only")
    for s in ("harvest", "hf", "openalex", "thumbs", "briefs"):
        r.add_argument(f"--skip-{s}", action="store_true")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("validate", help="validate an existing site/ directory")
    v.add_argument("--site", default="site")
    v.set_defaults(fn=cmd_validate)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
