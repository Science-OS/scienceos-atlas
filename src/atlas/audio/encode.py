"""AAC-LC in an m4a (MPEG-4) container, mono, at the engine's sample rate. PyAV bundles ffmpeg (a wheel, nothing to install);
the `ffmpeg` binary is the fallback when PyAV is missing. Output is `faststart` so a client can play it while it downloads."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _bitrate_bps(bitrate: str) -> int:
    b = bitrate.strip().lower()
    return int(float(b[:-1]) * 1000) if b.endswith("k") else int(b)


def _to_int16(pcm: np.ndarray) -> np.ndarray:
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)


def encode_with_av(pcm: np.ndarray, sample_rate: int, bitrate: str) -> bytes:
    import av
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "clip.m4a"
        with av.open(str(path), mode="w", format="ipod", options={"movflags": "faststart"}) as out:
            stream = out.add_stream("aac", rate=sample_rate, layout="mono")
            stream.bit_rate = _bitrate_bps(bitrate)
            samples = _to_int16(pcm)
            step = 1024 * 16
            pts = 0
            for i in range(0, len(samples), step):
                chunk = np.ascontiguousarray(samples[i : i + step].reshape(1, -1))
                frame = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
                frame.sample_rate = sample_rate
                frame.pts = pts
                pts += chunk.shape[1]
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)
        return path.read_bytes()


def encode_with_ffmpeg(pcm: np.ndarray, sample_rate: int, bitrate: str, binary: str | None = None) -> bytes:
    binary = binary or shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError("neither PyAV nor an ffmpeg binary is available to encode AAC")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "clip.m4a"
        subprocess.run([binary, "-v", "error", "-y", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-", "-c:a", "aac", "-b:a", bitrate,
                        "-movflags", "+faststart", str(path)], input=_to_int16(pcm).tobytes(), check=True, capture_output=True)
        return path.read_bytes()


def encode_m4a(pcm: np.ndarray, sample_rate: int, bitrate: str = "48k") -> bytes:
    try:
        import av  # noqa: F401
    except ImportError:
        return encode_with_ffmpeg(pcm, sample_rate, bitrate)
    return encode_with_av(pcm, sample_rate, bitrate)


def probe_duration_ms(data: bytes) -> int:
    """Duration of an encoded clip as a player sees it (container duration, edit list applied) - used by tests and the real-run report."""
    import av
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "clip.m4a"
        path.write_bytes(data)
        with av.open(str(path)) as c:
            stream = c.streams.audio[0]
            return int(round(float(c.duration) / 1000.0)) if c.duration else int(round(float(stream.duration * stream.time_base) * 1000))
