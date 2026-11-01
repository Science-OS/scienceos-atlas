"""Sample-level helpers: find the speech inside a synthesised sentence, trim and re-pad its silence, resample for the aligner.
Float32 mono arrays in [-1, 1]; numpy only (PyAV does the resampling for the aligner)."""

from __future__ import annotations

import numpy as np


class SilentAudio(ValueError):
    """The engine returned nothing but silence (or nothing at all) for a sentence."""


def ms_to_samples(ms: float, sr: int) -> int:
    return int(round(ms * sr / 1000.0))


def silence(ms: float, sr: int) -> np.ndarray:
    return np.zeros(ms_to_samples(ms, sr), dtype=np.float32)


def frame_rms(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((x[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(axis=1)).astype(np.float32)


def speech_bounds(x: np.ndarray, sr: int, *, frame_ms: float = 10.0, floor_db: float = -48.0, rel_db: float = -40.0) -> tuple[int, int]:
    """(first, last+1) sample of the speech: the first and last 10 ms frame whose RMS is above max(-48 dBFS, 40 dB below the loudest frame).
    Pocket TTS output has no background noise, so a simple energy gate is enough. Raises SilentAudio when no frame passes."""
    hop = max(1, ms_to_samples(frame_ms, sr))
    rms = frame_rms(x, hop)
    if rms.size == 0 or not np.isfinite(rms).all():
        raise SilentAudio("empty or non-finite audio")
    thr = max(10 ** (floor_db / 20), float(rms.max()) * 10 ** (rel_db / 20))
    voiced = np.flatnonzero(rms > thr)
    if voiced.size == 0:
        raise SilentAudio("no speech above the noise floor")
    return int(voiced[0]) * hop, min(len(x), (int(voiced[-1]) + 1) * hop)


def trim_and_pad(x: np.ndarray, sr: int, edge_ms: float, *, fade_ms: float = 4.0) -> tuple[np.ndarray, int, int]:
    """Cut the silence off both ends of a sentence and put back exactly `edge_ms` of digital silence on each side, so every sentence
    starts and ends alike. Returns (audio, speech_start, speech_end) with the speech span in samples of the returned audio."""
    a, b = speech_bounds(x, sr)
    body = x[a:b].astype(np.float32).copy()
    f = min(ms_to_samples(fade_ms, sr), len(body) // 2)
    if f > 0:   # the gate may cut inside a quiet tail: a 4 ms ramp keeps the cut from clicking
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        body[:f] *= ramp
        body[-f:] *= ramp[::-1]
    pad = silence(edge_ms, sr)
    return np.concatenate([pad, body, pad]), len(pad), len(pad) + len(body)


def peak_limit(x: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """Scale down only when the signal would clip; never up (levels stay what the voice model produced)."""
    peak = float(np.abs(x).max()) if x.size else 0.0
    return (x * (ceiling / peak)).astype(np.float32) if peak > ceiling else x.astype(np.float32)


def resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """float32 mono -> float32 mono at another rate, with PyAV's (libswresample) filter."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    import av
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sr_out)
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(x, dtype=np.float32).reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = sr_in
    frames = list(resampler.resample(frame)) + list(resampler.resample(None))
    return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]).astype(np.float32)
