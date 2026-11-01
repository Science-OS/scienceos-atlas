"""Smoke test with the real models. Skipped unless ATLAS_AUDIO_SMOKE=1 (it loads Pocket TTS and faster-whisper; needs `uv sync --extra audio` and the
weights in the Hugging Face cache or network access): `ATLAS_AUDIO_SMOKE=1 HF_HOME=... uv run pytest tests/test_audio_real_model.py`."""

from __future__ import annotations

import json
import os

import pytest

from atlas.audio.align import wer, word_tokens

pytestmark = pytest.mark.skipif(os.environ.get("ATLAS_AUDIO_SMOKE") != "1", reason="set ATLAS_AUDIO_SMOKE=1 to run the real-model smoke test")


def test_real_pocket_tts_and_whisper_produce_a_valid_aligned_clip(tmp_path):
    pytest.importorskip("pocket_tts")
    pytest.importorskip("faster_whisper")
    from audio_fakes import paper_json
    from atlas.audio.render import render
    from atlas.config import load_config
    from atlas.validate import audio_day_errors, make_validators

    papers = tmp_path / "p.json"
    papers.write_text(json.dumps([paper_json(1, title="Schrödinger Bridges for Image Generation", brief="A new bridge model makes images faster.",
                                             abstract="We study diffusion. It works.")]))
    out = render(load_config(), papers, "2026-10-21", tmp_path / "o")
    assert out["ok"] and out["synth"]["aligned"] == out["synth"]["built"] and out["synth"]["estimated"] == 0
    doc = json.loads((tmp_path / "o" / "2026-10-21.json").read_text(encoding="utf-8"))
    assert audio_day_errors(doc, make_validators()["audio_day"], tmp_path / "o" / "clips", label="smoke", expect_date="2026-10-21") == []
    clip = doc["papers"]["arxiv:2610.00001"]
    assert clip["timing"] == "aligned" and 5_000 < clip["durationMs"] < 40_000
    # round trip: whisper hears the script back almost word for word
    from atlas.audio.align import WhisperAligner
    import av
    import numpy as np

    c = av.open(str(tmp_path / "o" / "clips" / f"{clip['id']}.m4a"))
    pcm = np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(audio=0)]).astype(np.float32)
    if pcm.dtype != np.float32 or np.abs(pcm).max() > 1.5:
        pcm = pcm / 32768.0
    heard = WhisperAligner("tiny.en").transcribe(pcm, c.streams.audio[0].codec_context.sample_rate)
    ref = [t for w in clip["text"].split() for t in word_tokens(w)]
    hyp = [t for w, _, _ in heard for t in word_tokens(w)]
    assert wer(ref, hyp) < 0.25
