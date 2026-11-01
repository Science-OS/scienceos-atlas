from __future__ import annotations

import numpy as np

from atlas.audio.align import HeuristicParams, SentenceTiming, WhisperAligner, heuristic_times, map_words, wer, word_tokens


def monotonic(times, lo, hi):
    ps = pe = lo
    for s, e in times:
        assert lo - 1e-9 <= s <= e <= hi + 1e-9, (s, e)
        assert s >= ps - 1e-9 and e >= pe - 1e-9, "not monotonic"
        ps, pe = s, e
    for (_, e), (s2, _) in zip(times, times[1:]):
        assert e <= s2 + 1e-9, "words overlap"


def test_word_tokens_fold_case_accents_hyphens_and_apostrophes():
    assert word_tokens("State-of-the-art,") == ["state", "of", "the", "art"]
    assert word_tokens("Don't") == ["dont"] and word_tokens("don’t") == ["dont"]
    assert word_tokens("Café!") == ["cafe"] and word_tokens("Schrödinger–Newton") == ["schrodinger", "newton"]
    assert word_tokens("—") == [] and word_tokens("3.5") == ["3", "5"]


def test_map_words_exact_match():
    words = "Hello big world".split()
    rec = [("Hello", 0.1, 0.4), ("big", 0.5, 0.7), ("world.", 0.8, 1.2)]
    times, matched = map_words(words, rec, 0.0, 1.5)
    assert matched == 1.0
    assert times == [(0.1, 0.4), (0.5, 0.7), (0.8, 1.2)]


def test_map_words_hyphenated_script_word_spans_several_recognised_words():
    words = "A state-of-the-art model".split()
    rec = [("A", 0.0, 0.1), ("state", 0.2, 0.5), ("of", 0.5, 0.6), ("the", 0.6, 0.7), ("art", 0.7, 1.0), ("model", 1.1, 1.5)]
    times, matched = map_words(words, rec, 0.0, 2.0)
    assert matched == 1.0
    assert times[1] == (0.2, 1.0) and times[2] == (1.1, 1.5)


def test_map_words_and_the_other_way_round_whisper_merges_what_the_script_splits():
    words = "the art of well-being".split()
    rec = [("the", 0.0, 0.2), ("art", 0.2, 0.5), ("of", 0.5, 0.6), ("wellbeing", 0.7, 1.4)]
    times, matched = map_words(words, rec, 0.0, 2.0)
    monotonic(times, 0.0, 2.0)
    assert times[3][0] == 0.7 or times[3][0] < 0.9    # the hyphenated word starts where whisper's word starts


def test_map_words_number_heard_as_words_is_spread_inside_its_gap():
    words = "We tried 26 models today".split()
    rec = [("We", 0.0, 0.2), ("tried", 0.3, 0.6), ("twenty", 0.7, 1.0), ("six", 1.0, 1.3), ("models", 1.4, 1.8), ("today", 1.9, 2.3)]
    times, matched = map_words(words, rec, 0.0, 2.5)
    assert times[2] == (0.7, 1.3)
    assert times[3][0] == 1.4 and times[4][0] == 1.9
    assert matched == 0.8       # 4 of the 5 script tokens were heard as written
    monotonic(times, 0.0, 2.5)


def test_map_words_unheard_words_are_interpolated_between_their_neighbours():
    words = "alpha beta gamma delta epsilon".split()
    rec = [("alpha", 0.0, 0.4), ("epsilon", 2.0, 2.4)]
    times, matched = map_words(words, rec, 0.0, 2.5)
    assert matched == 0.4
    monotonic(times, 0.0, 2.5)
    assert times[0] == (0.0, 0.4) and times[4] == (2.0, 2.4)
    assert 0.4 <= times[1][0] and times[3][1] <= 2.0                    # strictly between the heard neighbours
    assert times[1][1] - times[1][0] < times[2][1] - times[2][0] < times[3][1] - times[3][0] + 1.0   # longer words get more time


def test_map_words_ignores_extra_recognised_words_and_clamps_to_the_speech_span():
    words = "one two".split()
    rec = [("uh", -0.5, 0.0), ("one", 0.05, 0.3), ("well", 0.4, 0.5), ("two", 0.6, 9.0)]
    times, _ = map_words(words, rec, 0.1, 1.0)
    monotonic(times, 0.1, 1.0)
    assert times[0][0] == 0.1 and times[1][1] == 1.0


def test_map_words_nothing_recognised_falls_back_to_a_spread_of_the_whole_span():
    times, matched = map_words("a b c".split(), [], 0.2, 1.1)
    assert matched == 0.0
    monotonic(times, 0.2, 1.1)
    assert times[0][0] == 0.2 and abs(times[-1][1] - 1.1) < 1e-9


def test_map_words_overlapping_whisper_words_are_made_monotonic():
    words = "x y z".split()
    rec = [("x", 0.0, 0.5), ("y", 0.3, 0.9), ("z", 0.2, 1.0)]     # garbage order
    times, _ = map_words(words, rec, 0.0, 1.0)
    monotonic(times, 0.0, 1.0)


def test_heuristic_times_cover_the_span_in_order_with_pauses_after_punctuation():
    words = "Fast, slow and steady wins: the race".split()
    times = heuristic_times(words, 0.12, 3.12)
    monotonic(times, 0.12, 3.12)
    assert times[0][0] == 0.12 and abs(times[-1][1] - 3.12) < 1e-9
    gap_after_comma = times[1][0] - times[0][1]
    gap_after_plain = times[3][0] - times[2][1]
    gap_after_colon = times[5][0] - times[4][1]
    assert gap_after_plain == 0 and abs(gap_after_comma - HeuristicParams().comma_pause_s) < 1e-9
    assert abs(gap_after_colon - HeuristicParams().colon_pause_s) < 1e-9
    # longer words take longer
    words2 = "a internationalization".split()
    t2 = heuristic_times(words2, 0.0, 2.0)
    assert (t2[1][1] - t2[1][0]) > 5 * (t2[0][1] - t2[0][0])


def test_heuristic_pauses_are_capped_to_a_share_of_the_span():
    words = "a, b, c, d, e, f, g".split()
    times = heuristic_times(words, 0.0, 1.0)
    speech = sum(e - s for s, e in times)
    assert speech >= 1.0 - HeuristicParams().max_pause_share - 1e-9
    monotonic(times, 0.0, 1.0)


def test_heuristic_single_word_and_empty():
    assert heuristic_times([], 0, 1) == []
    assert heuristic_times(["Hello."], 0.1, 0.9) == [(0.1, 0.9)]


def test_wer():
    assert wer("a b c".split(), "a b c".split()) == 0.0
    assert wer("a b c".split(), "a x c".split()) == 1 / 3
    assert wer("a b c".split(), "a c".split()) == 1 / 3
    assert wer([], []) == 0.0 and wer([], ["x"]) == 1.0


def _aligner_with(heard, min_match=0.5):
    al = object.__new__(WhisperAligner)          # no model: only the mapping and the fallback decision are under test
    al.min_match, al.model_name, al.use_prompt = min_match, "tiny.en", False
    al.transcribe = lambda pcm, sr, prompt=None: heard     # type: ignore[method-assign]
    return al


def test_whisper_aligner_returns_timings_when_enough_was_recognised():
    al = _aligner_with([("hello", 0.1, 0.4), ("there", 0.5, 0.9)])
    out = al.align(np.zeros(10, dtype=np.float32), 24000, ["Hello", "there."], 0.0, 1.0)
    assert isinstance(out, SentenceTiming) and out.source == "whisper" and out.matched == 1.0


def test_whisper_aligner_gives_up_when_it_recognised_too_little_or_failed():
    al = _aligner_with([("totally", 0.1, 0.4), ("different", 0.5, 0.9)])
    assert al.align(np.zeros(10, dtype=np.float32), 24000, ["Hello", "there."], 0.0, 1.0) is None
    assert _aligner_with([]).align(np.zeros(10, dtype=np.float32), 24000, ["Hello"], 0.0, 1.0) is None
    boom = _aligner_with([])
    boom.transcribe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decoder"))    # type: ignore[method-assign]
    assert boom.align(np.zeros(10, dtype=np.float32), 24000, ["Hello"], 0.0, 1.0) is None


def test_whisper_bias_table_and_application():
    from atlas.audio import align
    assert align.WHISPER_BIAS_S["tiny.en"] == -0.10

    class Seg:   # what faster-whisper returns
        def __init__(self, words):
            self.words = words

    class W:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    al = object.__new__(WhisperAligner)
    al.model_name, al.use_prompt, al.bias = "tiny.en", False, align.WHISPER_BIAS_S["tiny.en"]
    al.model = type("M", (), {"transcribe": lambda self, audio, **kw: ([Seg([W(" Hello", 0.05, 0.30), W(" there", 0.40, 0.80)])], None)})()
    heard = al.transcribe(np.zeros(2400, dtype=np.float32), 24000)
    assert heard[0] == (" Hello", 0.0, 0.2) or (heard[0][0] == " Hello" and heard[0][1] == 0.0 and abs(heard[0][2] - 0.2) < 1e-9)     # clamped at 0, never negative
    assert abs(heard[1][1] - 0.30) < 1e-9 and abs(heard[1][2] - 0.70) < 1e-9
