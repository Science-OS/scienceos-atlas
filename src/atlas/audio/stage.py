"""The two pipeline stages of the audio brief and the glue the CLI shares:

* `audio_scripts` - pick the day's papers from the ranked feed, get their spoken summaries (cached LLM text, or the fallback), write the plan.
* `audio`         - speak the plan into the audio store. Runs in a child process in real runs (a native crash, a hang or an out-of-memory in torch
                    or ctranslate2 must never take the feed down with it); in-process when an engine is injected (tests).

Neither stage ever raises into the pipeline: the caller wraps them, and each returns a stats dict that ends up in status.json."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import AudioCfg, Config
from ..matching import match_authors
from ..models import ts
from ..store import Store, StoredPaper
from .align import Aligner, WhisperAligner
from .engine import TtsEngine, load_engine, pocket_available, whisper_available
from .plan import DayPlan, PaperInput, PlanPaper, day_texts, script_for
from .select import Candidate, select
from .storage import AudioStore
from .summaries import SummaryNeed, SummaryProvider, ensure_summaries, resolve_summary_provider
from .synth import SynthStats, synthesize_plan

log = logging.getLogger("atlas.audio.stage")


def engine_installed() -> bool:
    return pocket_available()


def is_enabled(cfg: AudioCfg, installed: bool) -> tuple[bool, str | None]:
    """audio.enabled: false = off; true = on (a missing extra is logged by the caller, never fatal); auto = on when pocket-tts is installed."""
    mode = str(cfg.enabled).lower()
    if mode == "false":
        return False, "audio.enabled is false"
    if not installed:
        return False, "the audio extra (pocket-tts) is not installed" if mode in ("auto", "true") else None
    return True, None


# ------------------------------------------------------------------ scripts


def primary_category(sp: StoredPaper) -> str | None:
    return sp.api.primary_category or (sp.record.categories[0] if sp.record.categories else None)


def first_affiliation(sp: StoredPaper, oa_row: dict[str, Any] | None) -> str | None:
    """The first author's institution: OpenAlex's canonical name when the work is enriched, else arXiv's free-text affiliation."""
    if oa_row and oa_row.get("status") == "found" and oa_row.get("work"):
        match = match_authors([a["name"] for a in sp.record.authors], oa_row["work"]["authorships"]).get(0)
        if match and match.get("institutions"):
            return match["institutions"][0]["name"]
    return sp.record.authors[0].get("affiliation") if sp.record.authors else None


def paper_input(sp: StoredPaper, rank: int, brief: str | None, oa_row: dict[str, Any] | None) -> PaperInput:
    rec = sp.record
    return PaperInput(id=f"arxiv:{rec.arxiv_id}", arxiv_id=rec.arxiv_id, title=rec.title, authors=[a["name"] for a in rec.authors],
                      affiliation=first_affiliation(sp, oa_row), brief=brief, abstract=rec.abstract, category=primary_category(sp), rank=rank)


def build_plan(cfg: Config, day: str, inputs: list[PaperInput], store: Store, now: datetime,
               resolved: tuple[SummaryProvider, Any] | None) -> tuple[DayPlan, dict[str, Any]]:
    """Plan for `inputs` (already selected, in priority order): spoken summaries, then the script of each clip."""
    needs = [SummaryNeed(i.arxiv_id, i.title, i.abstract or "") for i in inputs if i.abstract]
    have, sstats = ensure_summaries(store, cfg, needs, now, resolved)
    papers, stats = [], {"llm": 0, "fallback": 0, "skipped_no_text": 0}
    for i in inputs:
        made = script_for(i, have.get(i.arxiv_id), max_words=cfg.audio.summary.words_max)
        if made is None:
            stats["skipped_no_text"] += 1
            continue
        text, source = made
        stats[source] += 1
        papers.append(PlanPaper(i.id, i.arxiv_id, i.title, text, source, i.why, i.rank))
    intros, outro = day_texts(date.fromisoformat(day))
    info = {"summaries": stats, "llm_enabled": resolved is not None}
    if sstats is not None:
        info["summaries"].update(generated=sstats.generated, failed=sstats.failed, cost_usd=round(sstats.cost_usd, 5), stopped_reason=sstats.stopped_reason)
    return DayPlan(day, papers, intros, outro), info


def plan_path(state_dir: Path, day: str) -> Path:
    return Path(state_dir) / "audio" / f"plan-{day}.json"


def stage_audio_scripts(store: Store, cfg: Config, ranked: list[StoredPaper], day: str, now: datetime, state_dir: Path, *,
                        resolved: tuple[SummaryProvider, Any] | None | bool = True) -> dict[str, Any]:
    """`ranked`: the day's papers best first, exactly the list the `today` shard is cut from. `resolved=True` = look the LLM up in the environment."""
    ac = cfg.audio
    if resolved is True:
        resolved = resolve_summary_provider(cfg)
    ranked = ranked[: cfg.feeds.today_max_items]
    cands = [Candidate(f"arxiv:{sp.record.arxiv_id}", primary_category(sp), r) for r, sp in enumerate(ranked, start=1)]
    chosen = select(cands, top_overall=ac.top_overall, per_category=ac.per_category, categories=ac.categories, max_papers=ac.max_papers)
    by_id = {f"arxiv:{sp.record.arxiv_id}": sp for sp in ranked}
    briefs = store.brief_rows()
    oa = store.oa_work_rows()
    inputs = []
    for c, why in chosen:
        sp = by_id[c.key]
        inp = paper_input(sp, c.rank, (briefs.get(sp.record.arxiv_id) or {}).get("text"), oa.get(sp.record.arxiv_id))
        inp.why = why
        inputs.append(inp)
    plan, info = build_plan(cfg, day, inputs, store, now, resolved)
    plan.save(plan_path(state_dir, day))
    return {"date": day, "ranked": len(ranked), "selected": len(chosen), "scripts": len(plan.papers),
            "top": sum(1 for _, w in chosen if w == "top"), "category_picks": sum(1 for _, w in chosen if w != "top"), **info}


# ------------------------------------------------------------------ synthesis


def open_aligner(cfg: AudioCfg) -> Aligner | None:
    if cfg.align != "whisper":
        return None
    if not whisper_available():
        log.warning("audio.align is whisper but faster-whisper is not installed: word timings will be estimated")
        return None
    try:
        return WhisperAligner(cfg.whisper_model)
    except Exception as exc:   # no model download possible, bad cache ...
        log.warning("whisper aligner unavailable (%s: %s): word timings will be estimated", type(exc).__name__, exc)
        return None


def run_synth(cfg: Config, plan: DayPlan, audio_dir: Path, now: datetime, *, engine: TtsEngine | None = None, aligner: Aligner | None | bool = True,
              deadline: float | None = None) -> SynthStats:
    """Speak a plan into `audio_dir`. `engine`/`aligner` are injected by tests; None/True = the real ones."""
    ac = cfg.audio
    own_engine = engine is None
    t0 = time.monotonic()
    engine = engine or load_engine(ac.voice, ac.language, ac.model)
    try:
        al = open_aligner(ac) if aligner is True else (aligner or None)
        loaded = time.monotonic() - t0
        # the time budget counts from here, not from the model load
        stats = synthesize_plan(plan, ac, AudioStore(audio_dir), engine, al, now, deadline=deadline)
        stats.load_seconds = loaded
        return stats
    finally:
        if own_engine:
            engine.close()


def stage_audio(cfg: Config, day: str, state_dir: Path, audio_dir: Path, now: datetime, *, engine: TtsEngine | None = None,
                aligner: Aligner | None | bool = True, isolate: bool = True) -> dict[str, Any]:
    path = plan_path(state_dir, day)
    if not path.exists():
        return {"date": day, "skipped": "no plan (audio_scripts did not run)"}
    plan = DayPlan.load(path)
    if engine is not None or not isolate:
        return run_synth(cfg, plan, Path(audio_dir), now, engine=engine, aligner=aligner).as_dict()
    return run_synth_child(cfg, path, day, state_dir, Path(audio_dir), now)


def run_synth_child(cfg: Config, plan_file: Path, day: str, state_dir: Path, audio_dir: Path, now: datetime) -> dict[str, Any]:
    """`atlas audio synth` in a child process with the exact config of this run (JSON is YAML) and a hard timeout."""
    work = Path(state_dir) / "audio"
    work.mkdir(parents=True, exist_ok=True)
    cfg_file, stats_file = work / "config.json", work / f"stats-{day}.json"
    cfg_file.write_text(json.dumps(cfg.model_dump(mode="json", by_alias=True)), encoding="utf-8")
    stats_file.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "atlas", "--config", str(cfg_file), "audio", "synth", "--plan", str(plan_file), "--audio-dir", str(audio_dir),
           "--stats-out", str(stats_file), "--now", ts(now)]
    hard_limit = cfg.audio.max_seconds + 600   # the child stops starting clips at max_seconds; this only catches a hang
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, timeout=hard_limit, check=False)
        code: Any = proc.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    try:
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stats = {"date": day, "error": f"audio synthesis ended without stats (exit {code}); clips written so far are kept"}
    if code != 0:
        stats["exit"] = code
    stats["child_seconds"] = round(time.time() - t0, 1)
    return stats
