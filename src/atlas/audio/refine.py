"""Use the audio itself to tighten word timings. Synthesised speech is clean, so a stretch of near-silence inside a sentence is a real pause of the
voice: the word before it has ended and the word after it has not started. Whisper's word timestamps are good to about 100 ms but blur exactly at
those pauses (the word after a comma is often placed inside the pause), so:

* `find_pauses`     - the silences (>= 120 ms) inside a sentence's speech;
* `snap_to_pauses`  - move the boundary between two words onto the pause it belongs to (whisper timings);
* `anchored_estimate` - the character-weighted estimate, but split at the real pauses when the commas of the text and the pauses of the audio
                        agree in number (the fallback when whisper is not available).

The first word of a sentence starts where the speech starts and the last one ends where it ends: that is exact by construction (silence is trimmed)."""

from __future__ import annotations

import numpy as np

from .align import HeuristicParams, Times, _spread, _tidy, heuristic_times
from .dsp import frame_rms

Pause = tuple[float, float]


def find_pauses(pcm: np.ndarray, sr: int, lo: float, hi: float, *, min_ms: float = 120.0, rel: float = 0.04, margin_ms: float = 30.0) -> list[Pause]:
    """Silences of at least `min_ms` strictly inside the speech span [lo, hi] (seconds): 10 ms frames whose RMS is below `rel` of the sentence's
    95th-percentile RMS. A silence touching the edges of the span (within `margin_ms`) is not a pause between words."""
    hop = max(1, int(sr * 0.01))
    e = frame_rms(pcm, hop)
    i, j = int(lo / 0.01), min(len(e), int(hi / 0.01))
    if j - i < 5:
        return []
    thr = rel * float(np.percentile(e[i:j], 95))
    quiet = e < thr
    m = int(margin_ms / 10)
    out: list[Pause] = []
    k = i
    while k < j:
        if quiet[k]:
            n = k
            while n < j and quiet[n]:
                n += 1
            if (n - k) * 10 >= min_ms and k > i + m and n < j - m:
                out.append((k * 0.01, n * 0.01))
            k = n
        else:
            k += 1
    return out


def _ends_with_pause_mark(word: str) -> bool:
    return word.rstrip("\"')]”’")[-1:] in (",", ";", ":", "–", "—")


def snap_to_pauses(times: Times, words: list[str], pauses: list[Pause], lo: float, hi: float, *, tol_s: float = 0.6, tol_plain_s: float = 0.2) -> Times:
    """For every real pause pick the word boundary closest to it - boundaries after a comma/colon within `tol_s`, any other within `tol_plain_s` -
    and set the earlier word's end to the pause start and the later word's start to the pause end. Anchors the sentence's first start and last end."""
    t = [list(x) for x in times]
    n = len(t)
    if n == 0:
        return []
    taken = -1
    for p0, p1 in pauses:
        mid = (p0 + p1) / 2
        best: tuple[float, int] | None = None
        for i in range(max(0, taken + 1), n - 1):
            b = (t[i][1] + t[i + 1][0]) / 2
            lim = tol_s if _ends_with_pause_mark(words[i]) else tol_plain_s
            d = abs(b - mid)
            if d <= lim and p0 > t[i][0] and p1 < t[i + 1][1] and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            i = best[1]
            t[i][1], t[i + 1][0] = p0, p1
            taken = i
    t[0][0], t[-1][1] = lo, hi
    return _tidy([(a, b) for a, b in t], lo, hi)


def anchored_estimate(words: list[str], lo: float, hi: float, pauses: list[Pause], params: HeuristicParams) -> Times:
    """Character-weighted estimate between the real pauses. Every comma/colon of the sentence must find a distinct pause near where the plain estimate
    expects it (within a quarter of the sentence, at least 0.5 s) - pauses elsewhere are dips inside words and are ignored; if any comma has no pause
    (the voice sometimes runs through a comma) the plain `heuristic_times` is used for the whole sentence."""
    plain = heuristic_times(words, lo, hi, params)
    cuts = [k for k, w in enumerate(words[:-1]) if _ends_with_pause_mark(w)]
    if not cuts or not pauses:
        return plain
    tol = max(0.5, 0.25 * (hi - lo))
    chosen: list[Pause] = []
    last = -1
    for k in cuts:
        expect = (plain[k][1] + plain[k + 1][0]) / 2
        cands = [(abs((p0 + p1) / 2 - expect), j) for j, (p0, p1) in enumerate(pauses) if j > last and abs((p0 + p1) / 2 - expect) <= tol]
        if not cands:
            return plain
        _, last = min(cands)
        chosen.append(pauses[last])
    out: Times = []
    start_word, seg_lo = 0, lo
    for k, (p0, p1) in zip(cuts, chosen):
        out += _spread(words[start_word:k + 1], seg_lo, max(seg_lo, p0))
        start_word, seg_lo = k + 1, p1
    out += _spread(words[start_word:], seg_lo, max(seg_lo, hi))
    return _tidy(out, lo, hi)
