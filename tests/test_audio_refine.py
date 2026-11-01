from __future__ import annotations

import numpy as np

from atlas.audio.align import HeuristicParams, heuristic_times
from atlas.audio.clips import ClipBuilder
from atlas.audio.dsp import silence, trim_and_pad
from atlas.audio.refine import anchored_estimate, find_pauses, snap_to_pauses
from atlas.audio.storage import AudioStore
from atlas.config import AudioCfg
from audio_fakes import SR, FakeAligner, FakeEngine
from test_audio_align import monotonic


def tone(seconds: float, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def sentence_audio(*parts) -> tuple[np.ndarray, float, float]:
    """parts: tone seconds (float) or silence seconds (negative float) -> (padded audio, lo, hi)."""
    x = np.concatenate([tone(p) if p > 0 else silence(-p * 1000, SR) for p in parts])
    y, s0, s1 = trim_and_pad(x, SR, 120)
    return y, s0 / SR, s1 / SR


def test_find_pauses_finds_real_silences_only():
    y, lo, hi = sentence_audio(0.6, -0.06, 0.6, -0.35, 0.6, -0.12, 0.4)
    ps = find_pauses(y, SR, lo, hi)
    assert len(ps) == 2                                            # the 60 ms gap is below the 100 ms minimum
    (a0, a1), (b0, b1) = ps
    assert abs(a0 - (lo + 1.26)) < 0.03 and abs((a1 - a0) - 0.35) < 0.03
    assert abs(b0 - (lo + 2.21)) < 0.03 and abs((b1 - b0) - 0.12) < 0.04


def test_edges_and_tone_only_audio_have_no_pauses():
    y, lo, hi = sentence_audio(1.5)
    assert find_pauses(y, SR, lo, hi) == []
    assert find_pauses(y, SR, lo, lo + 0.01) == []


def test_snap_moves_the_word_after_a_comma_out_of_the_pause():
    words = ["Fast,", "slow", "and", "steady"]
    #                  whisper put 'slow' 150 ms into the pause and let 'Fast,' run on into it
    whisper = [(0.12, 0.62), (0.80, 1.10), (1.55, 1.75), (1.85, 2.30)]
    pause = (0.52, 1.02)                                            # the voice is silent from 0.52 s to 1.02 s
    out = snap_to_pauses(whisper, words, [pause], 0.12, 2.30)
    assert out[0] == (0.12, 0.52) and out[1][0] == 1.02
    assert out[2:] == whisper[2:] or (out[2][0] == 1.55 and out[3][1] == 2.30)
    monotonic(out, 0.12, 2.30)


def test_snap_ignores_a_pause_that_is_far_from_every_boundary_and_keeps_first_and_last_anchored():
    words = ["a", "b", "c"]
    whisper = [(0.30, 0.50), (0.60, 0.80), (0.90, 1.20)]
    out = snap_to_pauses(whisper, words, [(1.9, 2.1)], 0.12, 1.30)
    assert out[0][0] == 0.12 and out[-1][1] == 1.30 and out[1] == (0.60, 0.80)


def test_snap_never_inverts_a_word_and_uses_each_boundary_once():
    words = ["x,", "y,", "z"]
    whisper = [(0.1, 0.4), (0.9, 1.2), (1.7, 2.0)]
    out = snap_to_pauses(whisper, words, [(0.45, 0.85), (1.25, 1.65)], 0.1, 2.0)
    assert out[0][1] == 0.45 and out[1][0] == 0.85 and out[1][1] == 1.25 and out[2][0] == 1.65
    monotonic(out, 0.1, 2.0)
    # a pause that starts before the word it should end (whisper far off): left alone
    bad = snap_to_pauses([(0.5, 0.8), (1.0, 1.4)], ["p,", "q"], [(0.2, 0.9)], 0.1, 1.4)
    monotonic(bad, 0.1, 1.4)


def test_anchored_estimate_splits_at_the_real_pauses_when_counts_agree():
    words = "First part, second part, third".split()
    params = HeuristicParams()
    pauses = [(1.0, 1.6), (2.4, 3.0)]
    t = anchored_estimate(words, 0.12, 3.8, pauses, params)
    monotonic(t, 0.12, 3.8)
    assert t[0][0] == 0.12 and t[1][1] == 1.0 and t[2][0] == 1.6 and t[3][1] == 2.4 and t[4][0] == 3.0 and abs(t[4][1] - 3.8) < 1e-9


def test_anchored_estimate_falls_back_to_the_plain_estimate_when_a_comma_finds_no_pause():
    words = "First part, second part".split()
    params = HeuristicParams()
    plain = heuristic_times(words, 0.0, 3.0, params)
    assert anchored_estimate(words, 0.0, 3.0, [], params) == plain
    assert anchored_estimate("no commas here at all".split(), 0.0, 3.0, [(1.0, 1.3)], params) == heuristic_times("no commas here at all".split(), 0.0, 3.0, params)


def test_a_dip_inside_a_word_is_not_mistaken_for_the_comma_pause():
    words = "This work proves that joint embedding architectures learn the same thing, up to a rotation".split()
    params = HeuristicParams()
    plain = heuristic_times(words, 0.12, 7.0, params)
    far = anchored_estimate(words, 0.12, 7.0, [(0.5, 0.6)], params)          # a 100 ms dip at the very start: nowhere near the comma
    assert far == plain
    near = anchored_estimate(words, 0.12, 7.0, [(0.5, 0.6), (5.3, 5.7)], params)
    cut = words.index("thing,")
    assert abs(near[cut][1] - 5.3) < 1e-9 and abs(near[cut + 1][0] - 5.7) < 1e-9                   # the pause near the comma is used, the stray dip is ignored
    monotonic(near, 0.12, 7.0)


# ------------------------------------------------------------------ in the clip builder, with a fake voice that really pauses at commas

TEXT = "Alpha beta, gamma delta epsilon, zeta eta."


def truth_words(clip):
    return [w[0] for w in clip["words"]]


def test_blurred_whisper_timings_are_snapped_onto_the_real_pauses(tmp_path):
    exact = ClipBuilder(AudioCfg(), FakeEngine(), FakeAligner(), AudioStore(tmp_path / "a")).build(TEXT).clip
    blurred = ClipBuilder(AudioCfg(), FakeEngine(), FakeAligner(blur=0.15), AudioStore(tmp_path / "b")).build(TEXT).clip
    after = [2, 5]                                                  # gamma, zeta follow the commas
    for i in after:
        assert abs(blurred["words"][i][0] - exact["words"][i][0]) <= 20            # snapped onto where the voice resumes
    for i in (1, 4):                                                # beta, epsilon: ended at the pause start
        assert abs(blurred["words"][i][1] - exact["words"][i][1]) <= 20
    assert blurred["timing"] == "aligned"


def test_the_estimate_uses_the_real_pauses_too(tmp_path):
    exact = ClipBuilder(AudioCfg(), FakeEngine(), FakeAligner(), AudioStore(tmp_path / "a")).build(TEXT).clip
    est = ClipBuilder(AudioCfg(), FakeEngine(), None, AudioStore(tmp_path / "b")).build(TEXT).clip
    assert est["timing"] == "estimated"
    for i in (2, 5):
        assert abs(est["words"][i][0] - exact["words"][i][0]) <= 20
    errs = [abs(a[0] - b[0]) for a, b in zip(exact["words"], est["words"])]
    assert max(errs) <= 130 and sum(errs) / len(errs) <= 60         # between exact anchors only the unequal word lengths (5 vs 7 letters) are guessed
