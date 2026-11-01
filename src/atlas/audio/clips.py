"""Text -> one clip: synthesise sentence by sentence, trim and re-pad each sentence's silence, join them, time every word, encode.

Layout of a clip (all silences are digital zeros, so the timeline is exact):

    [edge | sentence 1 speech | edge]  [sentence gap]  [edge | sentence 2 speech | edge]  ...

A sentence's span in the manifest is its `[edge | speech | edge]` block; its words lie inside the speech part. The clip id is a content key:
sha256(engine version | voice | text)[:32], so a paper that stays in the feed is never synthesised twice."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import AudioCfg
from .align import Aligner, HeuristicParams, SentenceTiming
from .dsp import SilentAudio, peak_limit, resample_mono, silence, speech_bounds, trim_and_pad
from .encode import encode_m4a
from .engine import TtsEngine
from .refine import anchored_estimate, find_pauses, snap_to_pauses
from .storage import AudioStore, clip_id, sha256_hex
from .text import Word, split_sentences, split_words

log = logging.getLogger("atlas.audio.clips")


class ClipError(RuntimeError):
    """A clip could not be made (engine failure, silence). The stage logs it and goes on with the next clip."""


@dataclass
class BuiltClip:
    clip: dict[str, Any]
    data: bytes | None                  # the m4a when it was just synthesised; None when an intact copy already existed
    reused: bool = False
    timings: dict[str, float] = field(default_factory=dict)   # synth_s, align_s, encode_s
    audio_s: float = 0.0
    sentences_whisper: int = 0
    sentences_estimated: int = 0
    matched: list[float] = field(default_factory=list)


def _ms(samples: float, sr: int) -> int:
    return int(round(samples * 1000.0 / sr))


class ClipBuilder:
    def __init__(self, cfg: AudioCfg, engine: TtsEngine, aligner: Aligner | None, store: AudioStore,
                 heuristic: HeuristicParams = HeuristicParams()):
        self.cfg, self.engine, self.aligner, self.store, self.heuristic = cfg, engine, aligner, store, heuristic
        self.sr = cfg.sample_rate
        self._index = store.clip_index()

    def id_for(self, text: str) -> str:
        return clip_id(self.engine.version, self.engine.voice, text)

    def reusable(self, cid: str) -> dict[str, Any] | None:
        old = self._index.get(cid)
        if old is None or not self.store.intact(old):
            return None
        if old["timing"] != "aligned" and self.aligner is not None:   # whisper is available now: a second pass can still give real timings
            return None
        return old

    # -- one sentence --------------------------------------------------------------------------------------------------
    def _speak(self, sentence: str, n_words: int) -> np.ndarray:
        """Synthesise a sentence. Sampling makes the model non-deterministic: an engine error, silence, or an output far too long or too short for its
        words (the model ran on, or stopped early) is retried once; the more plausible of the two is kept."""
        best: np.ndarray | None = None
        best_err = float("inf")
        last: Exception | None = None
        for _ in range(2):
            try:
                raw = np.asarray(self.engine.synthesize(sentence), dtype=np.float32).reshape(-1)
                if self.engine.sample_rate != self.sr:
                    raw = resample_mono(raw, self.engine.sample_rate, self.sr)
                a, b = speech_bounds(raw, self.sr)
            except SilentAudio as exc:
                last = exc
                continue
            except Exception as exc:
                last = exc
                continue
            dur = (b - a) / self.sr
            err = abs(dur - 0.38 * n_words)
            if best is None or err < best_err:
                best, best_err = raw, err
            if 0.12 * n_words <= dur <= 1.3 * n_words + 4.0:
                break
            log.warning("implausible length %.1fs for %d words; retrying", dur, n_words)
        if best is None:
            raise ClipError(f"no audio for {sentence[:40]!r}: {type(last).__name__}: {last}")
        return best

    # -- one clip ------------------------------------------------------------------------------------------------------
    def build(self, text: str) -> BuiltClip:
        cid = self.id_for(text)
        old = self.reusable(cid)
        if old is not None:
            return BuiltClip(old, None, reused=True, audio_s=old["durationMs"] / 1000.0)
        sr, cfg = self.sr, self.cfg
        words = split_words(text)
        spans = split_sentences(text)
        if not words or not spans:
            raise ClipError("empty text")
        by_sentence: list[list[Word]] = [[w for w in words if a <= w.char_start < b] for a, b in spans]
        if sum(len(g) for g in by_sentence) != len(words):
            raise ClipError("sentence split lost words")

        t_synth = t_align = 0.0
        pieces: list[np.ndarray] = []
        words_out: list[list[int]] = []
        sentences_out: list[list[int]] = []
        cursor = 0
        gap = silence(cfg.sentence_gap_ms, sr)
        built = BuiltClip({}, None)
        for si, ((a, b), group) in enumerate(zip(spans, by_sentence)):
            if si > 0:
                pieces.append(gap)
                cursor += len(gap)
            t0 = time.perf_counter()
            raw = self._speak(text[a:b], len(group))
            t_synth += time.perf_counter() - t0
            padded, s0, s1 = trim_and_pad(raw, sr, cfg.edge_ms)
            lo, hi = s0 / sr, s1 / sr
            t0 = time.perf_counter()
            timing: SentenceTiming | None = None
            group_words = [w.text for w in group]
            pauses = find_pauses(padded, sr, lo, hi)
            if self.aligner is not None:
                timing = self.aligner.align(padded, sr, group_words, lo, hi)
                if timing is not None and timing.source == "whisper":
                    timing = SentenceTiming(snap_to_pauses(timing.words, group_words, pauses, lo, hi), timing.source, timing.matched)
            if timing is None:
                timing = SentenceTiming(anchored_estimate(group_words, lo, hi, pauses, self.heuristic), "heuristic")
                built.sentences_estimated += 1
            else:
                built.sentences_whisper += 1
                built.matched.append(timing.matched)
            t_align += time.perf_counter() - t0
            for w, (ws, we) in zip(group, timing.words):
                words_out.append([_ms(cursor + ws * sr, sr), _ms(cursor + we * sr, sr), w.u16_start, w.u16_end])
            sentences_out.append([_ms(cursor, sr), _ms(cursor + len(padded), sr), group[0].u16_start, group[-1].u16_end])
            pieces.append(padded)
            cursor += len(padded)
        pcm = peak_limit(np.concatenate(pieces))
        duration_ms = _ms(len(pcm), sr)
        _finish_monotonic(words_out, duration_ms)
        _finish_monotonic(sentences_out, duration_ms)
        t0 = time.perf_counter()
        data = encode_m4a(pcm, sr, cfg.bitrate)
        t_enc = time.perf_counter() - t0
        timing_kind = "aligned" if self.aligner is not None and built.sentences_estimated == 0 else "estimated"
        built.clip = {
            "id": cid, "url": f"audio/clips/{cid}.m4a", "sha256": sha256_hex(data), "bytes": len(data), "durationMs": duration_ms,
            "text": text, "words": words_out, "sentences": sentences_out, "timing": timing_kind,
        }
        built.data = data
        built.audio_s = len(pcm) / sr
        built.timings = {"synth_s": t_synth, "align_s": t_align, "encode_s": t_enc}
        return built

    def remember(self, clip: dict[str, Any]) -> None:
        self._index[clip["id"]] = clip


def _finish_monotonic(rows: list[list[int]], duration_ms: int) -> None:
    """Integer milliseconds after rounding: starts and ends non-decreasing, end >= start, everything inside [0, duration]."""
    prev_start = prev_end = 0
    for r in rows:
        r[0] = min(max(r[0], prev_start), duration_ms)
        r[1] = min(max(r[1], r[0], prev_end), duration_ms)
        prev_start, prev_end = r[0], r[1]
