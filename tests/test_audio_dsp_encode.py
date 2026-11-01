from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from atlas.audio.dsp import SilentAudio, peak_limit, resample_mono, silence, speech_bounds, trim_and_pad
from atlas.audio.encode import encode_m4a, encode_with_ffmpeg, probe_duration_ms

SR = 24000


def tone(seconds: float, amp: float = 0.3, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_speech_bounds_finds_the_tone_between_silences():
    x = np.concatenate([silence(400, SR), tone(1.0), silence(700, SR)])
    a, b = speech_bounds(x, SR)
    assert abs(a / SR - 0.4) <= 0.011 and abs(b / SR - 1.4) <= 0.011


def test_silence_or_nan_raises():
    with pytest.raises(SilentAudio):
        speech_bounds(np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(SilentAudio):
        speech_bounds(np.full(SR, np.nan, dtype=np.float32), SR)
    with pytest.raises(SilentAudio):
        speech_bounds(np.zeros(0, dtype=np.float32), SR)


def test_trim_and_pad_leaves_exactly_the_edge_silence_on_both_sides():
    for lead, trail in [(0.4, 0.9), (0.02, 0.0), (0.12, 0.12)]:
        x = np.concatenate([silence(lead * 1000, SR), tone(0.8), silence(trail * 1000, SR)])
        y, s0, s1 = trim_and_pad(x, SR, 120)
        assert s0 == int(0.12 * SR) and len(y) - s1 == int(0.12 * SR)
        assert abs((s1 - s0) / SR - 0.8) <= 0.021
        assert np.abs(y[:s0]).max() == 0 and np.abs(y[s1:]).max() == 0
        assert np.abs(y[s0 + 200: s1 - 200]).max() > 0.2


def test_trim_fades_the_cut_so_it_cannot_click():
    y, s0, s1 = trim_and_pad(np.concatenate([silence(100, SR), tone(0.5, amp=0.5), silence(100, SR)]), SR, 120)
    assert abs(y[s0]) < 0.05 and abs(y[s1 - 1]) < 0.05


def test_peak_limit_only_scales_down():
    assert np.abs(peak_limit(tone(0.1, amp=1.6))).max() <= 0.97 + 1e-6
    quiet = tone(0.1, amp=0.2)
    assert np.allclose(peak_limit(quiet), quiet)


def test_resample_24k_to_16k_keeps_the_duration():
    y = resample_mono(tone(1.0), SR, 16000)
    assert abs(len(y) - 16000) <= 64
    assert np.abs(y).max() > 0.2
    assert resample_mono(tone(0.1), SR, SR).shape[0] == int(0.1 * SR)


def test_encoded_clip_is_aac_lc_mono_24k_near_48_kbps():
    av = pytest.importorskip("av")
    pcm = np.concatenate([tone(2.0), silence(500, SR), tone(2.0, hz=330)])
    data = encode_m4a(pcm, SR, "48k")
    assert data[4:8] == b"ftyp"
    assert abs(probe_duration_ms(data) - 4500) <= 30
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.m4a"
        p.write_bytes(data)
        with av.open(str(p)) as c:
            s = c.streams.audio[0]
            assert s.codec_context.name == "aac" and s.codec_context.sample_rate == SR and s.codec_context.channels == 1
            assert (c.format.name.split(",")[0]) in ("mov", "mp4", "m4a")
    kbps = len(data) * 8 / 4.5 / 1000
    assert 30 <= kbps <= 60, kbps       # 48 kbps target; a tone and silence compress below it


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg binary")
def test_ffmpeg_fallback_produces_an_equivalent_clip():
    pytest.importorskip("av")
    pcm = tone(2.0)
    data = encode_with_ffmpeg(pcm, SR, "48k")
    assert data[4:8] == b"ftyp" and abs(probe_duration_ms(data) - 2000) <= 40
