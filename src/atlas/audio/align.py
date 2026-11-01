"""Word timings for a synthesised sentence.

Two sources, both producing one (start, end) per word of the script, in seconds from the start of the sentence audio:

* `WhisperAligner` - faster-whisper (tiny.en, int8, CPU) re-listens to the sentence with `word_timestamps=True`; the recognised words are mapped onto the
  words of the script by `map_words` (difflib on lower-case alphanumeric tokens, so "state-of-the-art" and "state of the art" agree, numbers and
  names that were recognised differently are spread inside their gap, words nobody recognised are interpolated).
* `heuristic_times` - no model at all: the speech span of the sentence (silence already trimmed) is shared out by character weights with a pause
  after commas, colons and the like. Used when whisper is unavailable or recognised too little of the sentence.

`map_words` and `heuristic_times` are pure and unit-tested; the whisper model is only touched by `WhisperAligner`."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

import numpy as np

log = logging.getLogger("atlas.audio.align")

Times = list[tuple[float, float]]


def word_tokens(word: str) -> list[str]:
    """'State-of-the-art,' -> ['state', 'of', 'the', 'art']; "Don't" -> ['dont']; 'Café' -> ['cafe']. Lower case, accents folded, apostrophes dropped."""
    s = unicodedata.normalize("NFKD", word.lower())
    s = "".join(c for c in s if not unicodedata.combining(c)).replace("'", "").replace("’", "")
    return [t for t in re.split(r"[\W_]+", s) if t]


def _weight(word: str) -> float:
    return float(max(1, sum(c.isalnum() for c in word))) + 0.5


def _spread(words: list[str], lo: float, hi: float) -> Times:
    """Share [lo, hi] out over consecutive words by character weight (no gaps between them)."""
    if not words:
        return []
    w = [_weight(x) for x in words]
    unit = max(hi - lo, 0.0) / sum(w)
    out, t = [], lo
    for x in w:
        out.append((t, t + x * unit))
        t += x * unit
    return out


def map_words(script_words: list[str], recognised: list[tuple[str, float, float]], lo: float, hi: float) -> tuple[Times, float]:
    """Map recognised words (text, start, end) onto the script words. Returns (one (start, end) per script word, fraction of the script's
    tokens that were recognised exactly). All times are clamped to [lo, hi] (the speech span of the sentence) and are monotonic."""
    n = len(script_words)
    if n == 0:
        return [], 1.0
    s_tok: list[tuple[str, int]] = [(t, i) for i, w in enumerate(script_words) for t in word_tokens(w)]
    r_tok: list[tuple[str, float, float]] = []
    for text, start, end in recognised:
        toks = word_tokens(text)
        total = sum(len(t) for t in toks)
        t0 = start
        for t in toks:   # a recognised word that holds several tokens ("state-of-the-art") shares its span by length
            share = (end - start) * len(t) / total if total else 0.0
            r_tok.append((t, t0, t0 + share))
            t0 += share
    got: list[tuple[float, float] | None] = [None] * len(s_tok)
    exact = 0
    sm = SequenceMatcher(None, [t for t, _ in s_tok], [t for t, _, _ in r_tok], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                got[i1 + k] = (r_tok[j1 + k][1], r_tok[j1 + k][2])
            exact += i2 - i1
        elif tag == "replace":
            if i2 - i1 == j2 - j1:   # same number of tokens, different spelling (a name, a number written out): trust the position
                for k in range(i2 - i1):
                    got[i1 + k] = (r_tok[j1 + k][1], r_tok[j1 + k][2])
            else:                     # "26" heard as "twenty six", a name split in two: spread the heard span over the script tokens
                for k, tt in enumerate(_spread([t for t, _ in s_tok[i1:i2]], r_tok[j1][1], r_tok[j2 - 1][2])):
                    got[i1 + k] = tt
    spans: list[tuple[float, float] | None] = [None] * n
    for (_, wi), tt in zip(s_tok, got):
        if tt is not None:
            cur = spans[wi]
            spans[wi] = tt if cur is None else (min(cur[0], tt[0]), max(cur[1], tt[1]))
    # words nobody heard: share the gap between the neighbouring heard words
    i = 0
    while i < n:
        if spans[i] is not None:
            i += 1
            continue
        j = i
        while j < n and spans[j] is None:
            j += 1
        left = spans[i - 1][1] if i > 0 and spans[i - 1] is not None else lo
        right = spans[j][0] if j < n and spans[j] is not None else hi
        filled = _spread(script_words[i:j], left, max(left, right))
        for k, tt in enumerate(filled):
            spans[i + k] = tt
        i = j
    return _tidy([s for s in spans if s is not None], lo, hi), (exact / len(s_tok) if s_tok else 1.0)


def _tidy(times: Times, lo: float, hi: float) -> Times:
    """Inside [lo, hi], starts and ends non-decreasing, end >= start, and a word ends no later than the next one starts."""
    out: Times = []
    prev_start = lo
    for s, e in times:
        s = min(max(s, prev_start), hi)
        e = min(max(e, s), hi)
        out.append((s, e))
        prev_start = s
    for i in range(len(out) - 1):
        s, e = out[i]
        out[i] = (s, min(e, out[i + 1][0]))
    return out


@dataclass(frozen=True)
class HeuristicParams:
    comma_pause_s: float = 0.37      # after "," - measured on Pocket TTS / alba output: median gap 400 ms, mean 373 ms (README "Audio brief")
    colon_pause_s: float = 0.34      # after ":" ";" and dashes (median 340 ms)
    stop_pause_s: float = 0.20       # a full stop / ? / ! inside a sentence (an abbreviation, a quoted question)
    max_pause_share: float = 0.35    # never let pauses take more than this share of the speech span


def heuristic_times(words: list[str], lo: float, hi: float, params: HeuristicParams = HeuristicParams()) -> Times:
    """Pause-aware, character-weighted estimate inside the speech span [lo, hi] of one sentence (silence already trimmed)."""
    if not words:
        return []
    span = max(hi - lo, 0.0)
    pauses = []
    for w in words[:-1]:   # no pause after the last word: the sentence ends there
        tail = w.rstrip("\"')]”’")[-1:]
        pauses.append(params.comma_pause_s if tail == "," else params.colon_pause_s if tail in ";:–—" else params.stop_pause_s if tail in ".!?…" else 0.0)
    pauses.append(0.0)
    total_pause = sum(pauses)
    if total_pause > params.max_pause_share * span:
        scale = params.max_pause_share * span / total_pause
        pauses = [p * scale for p in pauses]
        total_pause = sum(pauses)
    weights = [_weight(w) for w in words]
    unit = (span - total_pause) / sum(weights)
    out: Times = []
    t = lo
    for w, p in zip(weights, pauses):
        out.append((t, t + w * unit))
        t += w * unit + p
    return _tidy(out, lo, hi)


@dataclass
class SentenceTiming:
    words: Times             # seconds from the start of the sentence audio
    source: str              # "whisper" | "heuristic"
    matched: float = 1.0     # fraction of the script's tokens that were recognised exactly (whisper only)


class Aligner(Protocol):
    name: str

    def align(self, pcm: np.ndarray, sample_rate: int, words: list[str], lo: float, hi: float) -> SentenceTiming | None:
        """Timings for the script words of one sentence, or None when this aligner cannot do it (the caller falls back)."""
        ...


# Whisper's word timestamps come from cross-attention and are biased by model size. Measured on 6 papers (36 sentences) of Pocket TTS / alba audio against
# the voice's own pauses (silence >= 120 ms marks where the word before ends and the word after starts): tiny.en places word starts a median 130 ms late
# and ends 75 ms late, small.en is within 15 ms at starts. Seconds added to every tiny.en word time; models not listed here are used as they are.
WHISPER_BIAS_S = {"tiny.en": -0.10}


class WhisperAligner:
    """faster-whisper on CPU. The model is loaded once; a sentence is a few seconds of audio, so this costs well under a second each."""
    name = "whisper"

    def __init__(self, model_name: str = "tiny.en", *, min_match: float = 0.5, compute_type: str = "int8", cpu_threads: int = 0,
                 initial_prompt: bool = False):
        from faster_whisper import WhisperModel
        self.model_name, self.min_match, self.use_prompt = model_name, min_match, initial_prompt
        self.bias = WHISPER_BIAS_S.get(model_name, 0.0)
        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)

    def transcribe(self, pcm: np.ndarray, sample_rate: int, prompt: str | None = None) -> list[tuple[str, float, float]]:
        from .dsp import resample_mono
        audio = resample_mono(pcm, sample_rate, 16000)
        segments, _ = self.model.transcribe(
            audio, language="en" if self.model_name.endswith(".en") else None, word_timestamps=True, temperature=0.0,
            condition_on_previous_text=False, vad_filter=False, initial_prompt=prompt if self.use_prompt else None)
        out = []
        for seg in segments:
            for w in seg.words or []:
                start = max(0.0, float(w.start) + self.bias)
                out.append((w.word, start, max(start, float(w.end) + self.bias)))
        return out

    def align(self, pcm: np.ndarray, sample_rate: int, words: list[str], lo: float, hi: float) -> SentenceTiming | None:
        try:
            heard = self.transcribe(pcm, sample_rate, " ".join(words))
        except Exception as exc:   # a decoder hiccup must not lose the clip: the caller estimates this sentence
            log.warning("whisper failed on a sentence (%s); estimating it", type(exc).__name__)
            return None
        if not heard:
            return None
        times, matched = map_words(words, heard, lo, hi)
        return SentenceTiming(times, "whisper", matched) if matched >= self.min_match else None


# ------------------------------------------------------------------ word error rate (re-transcription check, used by tests and the real-run report)


def wer(reference: list[str], hypothesis: list[str]) -> float:
    """Word error rate on token lists (Levenshtein over words), 0.0 = identical."""
    r, h = reference, hypothesis
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[len(h)] / len(r)
