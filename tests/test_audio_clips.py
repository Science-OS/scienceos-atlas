from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from atlas.audio.clips import ClipBuilder, ClipError
from atlas.audio.dsp import resample_mono
from atlas.audio.encode import probe_duration_ms
from atlas.audio.storage import AudioStore, clip_id, sha256_hex
from atlas.audio.text import split_words, utf16_len
from atlas.config import AudioCfg
from audio_fakes import SR, FakeAligner, FakeEngine

TEXT = "Gödel – Schrödinger équations. From Zoë Müller and colleagues at ETH. It works well, mostly! Really?"


def builder(tmp_path: Path, *, engine=None, aligner=True, cfg: AudioCfg | None = None) -> ClipBuilder:
    return ClipBuilder(cfg or AudioCfg(), engine or FakeEngine(), FakeAligner() if aligner is True else aligner, AudioStore(tmp_path))


def test_clip_id_is_the_documented_content_key():
    assert clip_id("4.0.1", "alba", "Hello.") == hashlib.sha256(b"4.0.1|alba|Hello.").hexdigest()[:32]
    assert len(clip_id("4.0.1", "alba", "x")) == 32
    assert len({clip_id("4.0.1", "alba", "x"), clip_id("4.0.2", "alba", "x"), clip_id("4.0.1", "other", "x")}) == 3


def test_a_clip_has_one_timing_per_word_with_exact_utf16_offsets(tmp_path):
    built = builder(tmp_path).build(TEXT)
    clip = built.clip
    words = split_words(TEXT)
    assert len(clip["words"]) == len(words) == len(TEXT.split())
    raw = TEXT.encode("utf-16-le")
    for row, w in zip(clip["words"], words):
        assert row[2:] == [w.u16_start, w.u16_end]
        assert raw[row[2] * 2: row[3] * 2].decode("utf-16-le") == w.text
    assert clip["timing"] == "aligned" and built.sentences_estimated == 0
    assert clip["id"] == clip_id("1.0.0", "alba", TEXT) and clip["url"] == f"audio/clips/{clip['id']}.m4a"
    assert clip["text"] == TEXT


def test_word_and_sentence_timings_are_monotonic_and_inside_the_clip(tmp_path):
    clip = builder(tmp_path).build(TEXT).clip
    d = clip["durationMs"]
    ps = pe = 0
    for s, e, *_ in clip["words"]:
        assert 0 <= s <= e <= d and s >= ps and e >= pe
        ps, pe = s, e
    ps = pe = 0
    for s, e, u0, u1 in clip["sentences"]:
        assert 0 <= s <= e <= d and s >= ps and e >= pe and 0 <= u0 < u1 <= utf16_len(TEXT)
        ps, pe = s, e
    assert len(clip["sentences"]) == 4
    for (s, e, u0, u1) in clip["sentences"]:       # each sentence span encloses the words of that sentence
        inner = [w for w in clip["words"] if u0 <= w[2] and w[3] <= u1]
        assert inner and inner[0][0] >= s and inner[-1][1] <= e


def test_layout_geometry_edges_and_gaps(tmp_path):
    clip = builder(tmp_path, cfg=AudioCfg(edge_ms=120, sentence_gap_ms=250)).build("One two. Three four five.").clip
    (s1, e1, *_), (s2, e2, *_) = clip["sentences"]
    assert s1 == 0 and s2 - e1 == 250                                       # the gap between sentences
    w = clip["words"]
    assert w[0][0] == 120                                                    # first word right after the 120 ms edge
    assert abs((w[1][0] - w[0][0]) - 360) <= 2                               # FakeEngine: 300 ms tone + 60 ms gap
    assert e2 == clip["durationMs"]
    assert abs((e1 - s1) - (120 + 660 + 120)) <= 3                           # edge + two words with a gap + edge


def test_encoded_file_matches_the_manifest(tmp_path):
    pytest.importorskip("av")
    built = builder(tmp_path).build(TEXT)
    clip = built.clip
    assert built.data is not None and clip["bytes"] == len(built.data) and clip["sha256"] == sha256_hex(built.data)
    assert abs(probe_duration_ms(built.data) - clip["durationMs"]) <= 60        # AAC frame granularity


def test_a_clip_that_is_already_stored_is_reused_without_synthesis(tmp_path):
    engine = FakeEngine()
    store = AudioStore(tmp_path)
    first = ClipBuilder(AudioCfg(), engine, FakeAligner(), store).build(TEXT)
    store.write_clip(first.clip["id"], first.data)
    store.write_day("2026-10-30", {"intros": {}, "outro": None, "papers": {"arxiv:1": first.clip}})
    n_calls = len(engine.calls)
    again = ClipBuilder(AudioCfg(), engine, FakeAligner(), store).build(TEXT)
    assert again.reused and again.data is None and again.clip == first.clip
    assert len(engine.calls) == n_calls


def test_a_stored_clip_with_a_damaged_file_is_not_reused(tmp_path):
    engine = FakeEngine()
    store = AudioStore(tmp_path)
    first = ClipBuilder(AudioCfg(), engine, FakeAligner(), store).build("Hello there.")
    store.write_clip(first.clip["id"], first.data[:-10])                  # truncated
    store.write_day("2026-10-30", {"papers": {"arxiv:1": first.clip}})
    again = ClipBuilder(AudioCfg(), engine, FakeAligner(), store).build("Hello there.")
    assert not again.reused and again.data is not None


def test_an_estimated_clip_is_upgraded_once_an_aligner_is_available(tmp_path):
    store = AudioStore(tmp_path)
    est = ClipBuilder(AudioCfg(), FakeEngine(), None, store).build("Hello there.")
    assert est.clip["timing"] == "estimated"
    store.write_clip(est.clip["id"], est.data)
    store.write_day("2026-10-30", {"papers": {"arxiv:1": est.clip}})
    assert ClipBuilder(AudioCfg(), FakeEngine(), None, store).build("Hello there.").reused                 # still no aligner: nothing to gain
    up = ClipBuilder(AudioCfg(), FakeEngine(), FakeAligner(), store).build("Hello there.")
    assert not up.reused and up.clip["timing"] == "aligned" and up.clip["id"] == est.clip["id"]


def test_heuristic_fallback_when_the_aligner_gives_up_marks_the_clip_estimated(tmp_path):
    built = builder(tmp_path, aligner=FakeAligner(fail=True)).build(TEXT)
    assert built.clip["timing"] == "estimated" and built.sentences_estimated == 4
    words = built.clip["words"]
    assert len(words) == len(TEXT.split())
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(words, words[1:]))
    assert words[0][0] >= 120 and words[-1][1] <= built.clip["durationMs"] - 120 + 3      # inside the speech of the sentences


def test_no_aligner_means_estimated(tmp_path):
    assert builder(tmp_path, aligner=None).build("Hello there. Bye now.").clip["timing"] == "estimated"


def test_estimate_error_against_known_truth_is_small(tmp_path):
    """FakeEngine speaks every word for exactly 300 ms with 60 ms gaps - the rhythm the character-weighted estimate assumes - so the estimate must
    land within a fraction of a word of the truth."""
    text = "Alpha bravo charlie delta echo foxtrot golf hotel."
    truth = builder(tmp_path / "a").build(text).clip["words"]
    est = builder(tmp_path / "b", aligner=FakeAligner(fail=True)).build(text).clip["words"]
    errs = [abs(t[0] - e[0]) for t, e in zip(truth, est)]
    assert max(errs) < 250 and sum(errs) / len(errs) < 120


def test_an_engine_failure_is_a_clip_error_not_a_crash(tmp_path):
    with pytest.raises(ClipError):
        builder(tmp_path, engine=FakeEngine(fail_on=("boom",))).build("This will boom now.")
    with pytest.raises(ClipError):
        builder(tmp_path, engine=FakeEngine(silent_on=("quiet",))).build("This is quiet.")


def test_an_implausibly_long_sentence_is_retried_once_and_the_better_try_is_kept(tmp_path):
    class Flaky(FakeEngine):
        def synthesize(self, text):
            self.stretch = {"": 40.0} if not self.calls else {}      # first try: 12 s per word
            return super().synthesize(text)

    eng = Flaky()
    clip = ClipBuilder(AudioCfg(), eng, FakeAligner(), AudioStore(tmp_path)).build("Short one here.").clip
    assert len(eng.calls) == 2 and clip["durationMs"] < 3000


def test_empty_text_is_a_clip_error(tmp_path):
    with pytest.raises(ClipError):
        builder(tmp_path).build("   ")


def test_engine_with_another_sample_rate_is_resampled_to_the_configured_rate(tmp_path):
    pytest.importorskip("av")

    class Slow(FakeEngine):
        sample_rate = 16000

        def synthesize(self, text):
            return resample_mono(super().synthesize(text), SR, 16000)

    clip = ClipBuilder(AudioCfg(sample_rate=24000), Slow(), FakeAligner(), AudioStore(tmp_path)).build("Hello there.").clip
    assert abs(clip["durationMs"] - 900) <= 40            # 120 + 300 + 60 + 300 + 120 ms, the same as at 24 kHz
