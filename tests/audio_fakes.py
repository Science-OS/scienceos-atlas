"""A fake speech engine and a fake aligner with known timings, so the audio tests never load torch or a model."""

from __future__ import annotations

import numpy as np

from atlas.audio.align import SentenceTiming

SR = 24000
WORD_S, GAP_S, LEAD_S, TRAIL_S = 0.30, 0.06, 0.35, 0.50
COMMA_GAP_S = 0.40            # after a word that ends in a comma the voice pauses (a real, measurable silence)


def gap_after(word: str) -> float:
    return COMMA_GAP_S if word.endswith(",") else GAP_S


class FakeEngine:
    """One 0.30 s tone per word, 60 ms between words, 350 ms of silence before and 500 ms after (so the trimmer has something to cut)."""
    name, version, model, voice, sample_rate = "fake-tts", "1.0.0", "fake/model", "alba", SR

    def __init__(self, fail_on: tuple[str, ...] = (), silent_on: tuple[str, ...] = (), stretch: dict[str, float] | None = None):
        self.calls: list[str] = []
        self.fail_on, self.silent_on, self.stretch = fail_on, silent_on, stretch or {}
        self.closed = False

    def synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if any(f in text for f in self.fail_on):
            raise RuntimeError("engine exploded")
        if any(f in text for f in self.silent_on):
            return np.zeros(SR, dtype=np.float32)
        k = next((v for key, v in self.stretch.items() if key in text), 1.0)
        parts = [np.zeros(int(LEAD_S * SR), dtype=np.float32)]
        words = text.split()
        for i, _ in enumerate(words):
            t = np.arange(int(WORD_S * k * SR)) / SR
            tone = 0.3 * np.sin(2 * np.pi * (200 + 40 * (i % 5)) * t)
            parts.append(tone.astype(np.float32))
            if i < len(words) - 1:
                parts.append(np.zeros(int(gap_after(words[i]) * SR), dtype=np.float32))
        parts.append(np.zeros(int(TRAIL_S * SR), dtype=np.float32))
        return np.concatenate(parts)

    def close(self) -> None:
        self.closed = True


class FakeAligner:
    """Exact timings for FakeEngine output (a word takes 0.30 s, then 0.06 s of gap or 0.40 s after a comma). `blur` makes it behave like whisper
    around pauses: the word after a comma is reported `blur` seconds early (inside the pause), the word before it that much late."""
    name = "fake-whisper"

    def __init__(self, fail: bool = False, blur: float = 0.0):
        self.fail, self.blur, self.calls = fail, blur, 0

    def truth(self, words, lo):
        t, out = lo, []
        for w in words:
            out.append((t, t + WORD_S))
            t += WORD_S + gap_after(w)
        return out

    def align(self, pcm, sample_rate, words, lo, hi):
        self.calls += 1
        if self.fail:
            return None
        times = self.truth(words, lo)
        if self.blur:
            for i, w in enumerate(words[:-1]):
                if w.endswith(","):
                    times[i] = (times[i][0], times[i][1] + self.blur)
                    times[i + 1] = (times[i + 1][0] - self.blur, times[i + 1][1])
        return SentenceTiming(times, "whisper", 1.0)


def paper_json(n: int, *, title: str | None = None, authors: list[tuple[str, str | None]] | None = None, brief: str | None = "A short brief about the work.",
               abstract: str | None = "This work studies a thing. It does so carefully. The result is good.", cat: str = "cs.LG") -> dict:
    authors = authors if authors is not None else [("Ada Lovelace", None), ("Alan Turing", None)]
    return {
        "id": f"arxiv:2610.{n:05d}", "arxivId": f"2610.{n:05d}", "title": title or f"Paper number {n}",
        "authors": [{"name": a, "position": i, **({"affiliation": af} if af else {})} for i, (a, af) in enumerate(authors)],
        "categories": [cat], "primaryCategory": cat, "publishedAt": "2026-10-20T00:00:00Z", "links": [],
        **({"brief": {"text": brief, "source": "llm", "generatedAt": "2026-10-21T00:00:00Z"}} if brief else {}),
        **({"abstract": abstract} if abstract else {}),
    }
