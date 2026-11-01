"""The `audio` stage: speak a day's plan, store the clips, write the day manifest, prune old days.

Order of work: outro, intros (a few seconds of audio each), then the papers in priority order (top of the feed first). The stage stops starting new
clips at `audio.max_seconds`; whatever is missing simply has no entry in the manifest and the phone reads that paper with its own voice."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import AudioCfg
from ..models import ts
from . import SCHEMA_ID
from .align import Aligner
from .clips import BuiltClip, ClipBuilder, ClipError
from .engine import TtsEngine, credit_for
from .plan import DayPlan
from .storage import AudioStore, clip_id

log = logging.getLogger("atlas.audio.synth")


@dataclass
class SynthStats:
    date: str
    wanted: int = 0                    # papers in the plan
    papers: int = 0                    # paper clips in the manifest
    intros: int = 0
    outro: bool = False
    built: int = 0
    reused: int = 0
    failed: int = 0
    skipped_budget: int = 0
    aligned: int = 0                   # clips with whisper timings (all sentences)
    estimated: int = 0
    audio_seconds: float = 0.0         # of clips synthesised in this run
    synth_seconds: float = 0.0
    align_seconds: float = 0.0
    encode_seconds: float = 0.0
    load_seconds: float = 0.0          # loading the engine and the aligner (set by the caller)
    new_bytes: int = 0
    stopped_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    pruned: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["rtf"] = round(self.audio_seconds / (self.synth_seconds + self.align_seconds + self.encode_seconds), 2) if self.synth_seconds else None
        for k in ("audio_seconds", "synth_seconds", "align_seconds", "encode_seconds", "load_seconds"):
            d[k] = round(d[k], 2)
        return d


def engine_block(engine: TtsEngine) -> dict[str, str]:
    return {"name": engine.name, "version": engine.version, "model": engine.model, "voice": engine.voice, "credit": credit_for(engine.voice)}


def synthesize_plan(plan: DayPlan, cfg: AudioCfg, store: AudioStore, engine: TtsEngine, aligner: Aligner | None, now: datetime, *,
                    deadline: float | None = None) -> SynthStats:
    """Speak `plan`; clips and the day manifest end up in `store`. `deadline` is a time.monotonic() value (default: now + audio.max_seconds)."""
    t_start = time.monotonic()
    deadline = deadline if deadline is not None else t_start + cfg.max_seconds
    stats = SynthStats(plan.date, wanted=len(plan.papers))
    builder = ClipBuilder(cfg, engine, aligner, store)

    def make(label: str, text: str) -> dict[str, Any] | None:
        try:
            b: BuiltClip = builder.build(text)
        except ClipError as exc:
            stats.failed += 1
            stats.errors.append(f"{label}: {exc}")
            log.warning("clip %s failed: %s", label, exc)
            return None
        except Exception as exc:   # a bug or a native hiccup in one clip must not cost the others
            stats.failed += 1
            stats.errors.append(f"{label}: {type(exc).__name__}: {exc}")
            log.exception("clip %s failed", label)
            return None
        if b.reused:
            stats.reused += 1
        else:
            assert b.data is not None
            store.write_clip(b.clip["id"], b.data)
            builder.remember(b.clip)
            stats.built += 1
            stats.new_bytes += len(b.data)
            stats.audio_seconds += b.audio_s
            stats.synth_seconds += b.timings["synth_s"]
            stats.align_seconds += b.timings["align_s"]
            stats.encode_seconds += b.timings["encode_s"]
        if b.clip["timing"] == "aligned":
            stats.aligned += 1
        else:
            stats.estimated += 1
        log.info("clip %-14s %5.1fs %s%s", label, b.clip["durationMs"] / 1000.0, "reused" if b.reused else f"built in {sum(b.timings.values()):.1f}s",
                 "" if b.clip["timing"] == "aligned" else " (estimated timing)")
        return b.clip

    prev = store.load_day(plan.date) or {}   # an earlier run of the same day: whatever it finished and this run could not stays in the manifest

    def carried(old: dict[str, Any] | None) -> dict[str, Any] | None:
        """A clip of the earlier manifest, if its file is intact and it was made by this engine and voice (the manifest names one engine)."""
        return old if old and old["id"] == clip_id(engine.version, engine.voice, old["text"]) and store.intact(old) else None

    outro: dict[str, Any] | None = None
    intros: dict[str, dict[str, Any]] = {}
    papers: dict[str, dict[str, Any]] = {p.id: old for p in plan.papers if (old := carried((prev.get("papers") or {}).get(p.id)))}

    def flush() -> None:
        """Write the manifest of what exists so far: a run that is killed half-way still leaves a consistent, playable day."""
        if papers and outro is not None and intros:
            store.write_day(plan.date, {
                "schema": SCHEMA_ID, "date": plan.date, "generatedAt": ts(now), "engine": engine_block(engine), "gapMs": cfg.gap_ms,
                "intros": dict(sorted(intros.items(), key=lambda kv: int(kv[0]))), "outro": outro,
                "papers": {p.id: papers[p.id] for p in plan.papers if p.id in papers},       # plan (priority) order
            })

    outro = make("outro", plan.outro) or carried(prev.get("outro"))
    for n, text in sorted(plan.intros.items(), key=lambda kv: int(kv[0])):
        c = make(f"intro {n}", text) or carried((prev.get("intros") or {}).get(n))
        if c is not None:
            intros[n] = c
    for i, p in enumerate(plan.papers):
        if time.monotonic() >= deadline:
            stats.skipped_budget = len(plan.papers) - i
            stats.stopped_reason = f"time budget ({cfg.max_seconds:.0f}s) reached after {i} of {len(plan.papers)} papers"
            log.warning("audio: %s", stats.stopped_reason)
            break
        c = make(p.arxiv_id, p.text)
        if c is not None:
            papers[p.id] = c
            flush()
    stats.papers, stats.intros, stats.outro = len(papers), len(intros), outro is not None
    flush()
    if not (papers and outro is not None and intros):
        stats.stopped_reason = stats.stopped_reason or "no usable clips; manifest not written"
        log.warning("audio: manifest for %s not written (papers=%d intros=%d outro=%s)", plan.date, len(papers), len(intros), outro is not None)
    removed = store.prune(plan.date, cfg.retention_days)
    stats.pruned = {k: len(v) for k, v in removed.items()}
    return stats
