from __future__ import annotations

from datetime import date

import pytest

from atlas.audio.text import (INTRO_COUNTS, OUTRO_TEXT, byline, clean_affiliation, clean_latex, clean_title, intro_text, normalise_for_speech,
                              paper_clip_text, spoken_date, split_sentences, split_words, utf16_len, utf16_offsets)


# ------------------------------------------------------------------ UTF-16 offsets

def test_utf16_offsets_match_the_definition_for_non_ascii_text():
    text = "Gödel – Schrödinger: Café ü 😀 end"
    off = utf16_offsets(text)
    for i in range(len(text) + 1):
        assert off[i] == len(text[:i].encode("utf-16-le")) // 2
    assert utf16_len(text) == off[-1] == len(text) + 1    # exactly one astral character (2 code units)


def test_every_word_span_slices_the_exact_word_out_of_the_utf16_text():
    text = "Café – Schrödinger's équation: ü-test 😀 done."
    raw = text.encode("utf-16-le")
    words = split_words(text)
    assert [w.text for w in words] == text.split()
    for w in words:
        assert raw[w.u16_start * 2: w.u16_end * 2].decode("utf-16-le") == w.text
        assert text[w.char_start:w.char_end] == w.text
    assert words[-1].u16_end == utf16_len(text)


def test_accents_and_dashes_survive_the_normaliser_but_keep_a_tidy_word_list():
    t = normalise_for_speech("Gödel–Hilbert Équations for Müller")
    assert t == "Gödel–Hilbert Équations for Müller"
    assert [w.text for w in split_words(t)] == t.split()


# ------------------------------------------------------------------ normaliser

@pytest.mark.parametrize("raw,expected", [
    (r"$\alpha$-Divergence for $O(n^2)$ Attention", "alpha-Divergence for O(n squared) Attention"),
    (r"Learning \textbf{Fast} \& Slow", "Learning Fast and Slow"),
    (r"G\"odel and Schr\"odinger", "Gödel and Schrödinger"),
    (r"{LLM} Agents with \emph{Memory}", "LLM Agents with Memory"),
    ("50% faster, 3x cheaper & 2020–2024", "50 percent faster, 3 times cheaper and 2020 to 2024"),
    ("LLMs -- a survey", "LLMs, a survey"),
    ("Fast — and slow", "Fast, and slow"),
    ("Deep learning [12, 13] at https://example.com/x works", "Deep learning at works"),
    ("Deep learning (Smith et al., 2020) as in arXiv:2610.12345v2 works", "Deep learning as in works"),
    ("C++ and C# for e.g. systems vs. scripts", "C plus plus and C sharp for for example, systems versus scripts"),
    ("深度学习 for α-β search", "for alpha-beta search"),
    ("Smith et al. showed it", "Smith and colleagues showed it"),
    ("A    spaced\ttitle​", "A spaced title"),
])
def test_normaliser(raw, expected):
    assert normalise_for_speech(raw) == expected


def test_normaliser_leaves_no_soundless_tokens_or_dangling_marks():
    for raw in ["A - B", "A  –  B", "Bullets • here", "x ~ y", "Hello !", "Wait , what ?", "( )", "A / B"]:
        out = normalise_for_speech(raw)
        assert all(any(c.isalnum() for c in tok) for tok in out.split()), out
        assert "  " not in out and not out.startswith((",", " ")) and not out.endswith(" ")


def test_clean_latex_handles_math_wrappers_and_accents():
    assert clean_latex(r"$\mathcal{O}(n \log n)$") == "O(n log n)"
    assert clean_latex(r"$x_i^2$") == "x i squared"
    assert " ".join(clean_latex(r"na\"ive Bayes \& \textsc{Lasso}").split()) == "naïve Bayes and Lasso"
    assert "\\" not in clean_latex(r"\unknowncmd{abc} and \alpha")


def test_titles_lose_their_trailing_full_stop_but_keep_question_marks():
    assert clean_title("Attention Is All You Need.") == "Attention Is All You Need"
    assert clean_title("Is Attention All You Need?") == "Is Attention All You Need?"


# ------------------------------------------------------------------ clip text

def test_paper_clip_text_follows_the_contract():
    t = paper_clip_text("Intrinsic Distribution Matching", "Deyuan Liu", 5, "Inclusion AI", "It makes things better. It is fast.")
    assert t == "Intrinsic Distribution Matching. From Deyuan Liu and colleagues at Inclusion AI. It makes things better. It is fast."


def test_byline_variants():
    assert byline("Ada Lovelace", 1, None) == "From Ada Lovelace."
    assert byline("Ada Lovelace", 2, None) == "From Ada Lovelace and colleagues."
    assert byline("Ada Lovelace", 1, "ETH Zurich") == "From Ada Lovelace at ETH Zurich."
    assert byline("", 3, "ETH") == "From the authors."
    assert byline("李雷", 3, "ETH") == "From the authors."          # a name the voice cannot read is not spoken as noise
    assert byline("Zoë Müller", 2, "University of Zürich; Dept. of CS") == "From Zoë Müller and colleagues at University of Zürich."


def test_a_question_title_keeps_its_mark_in_the_clip_text():
    t = paper_clip_text("Is Attention All You Need?", "A B", 2, None, "It is. Yes.")
    assert t.startswith("Is Attention All You Need? From A B and colleagues. It is.")


def test_affiliations_that_are_not_sayable_are_dropped():
    assert clean_affiliation("Inclusion AI") == "Inclusion AI"
    assert clean_affiliation("MIT (CSAIL)") == "MIT"
    assert clean_affiliation("a@b.com") is None
    assert clean_affiliation("x" * 90) is None
    assert clean_affiliation("Dept of CS, 12345 Somewhere") is None
    assert clean_affiliation(None) is None and clean_affiliation("") is None


# ------------------------------------------------------------------ intros and outro

def test_intro_for_six_and_for_one():
    assert intro_text(6, date(2026, 10, 30)) == "Here are your six briefs for Friday, October 30."
    assert intro_text(1, date(2026, 10, 30)) == "Here is your brief for Friday, October 30."


def test_intros_exist_for_one_to_eight_and_use_the_feed_date_not_the_machine_date():
    assert INTRO_COUNTS == tuple(range(1, 9))
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    for n in INTRO_COUNTS[1:]:
        assert intro_text(n, date(2026, 1, 5)) == f"Here are your {words[n - 1]} briefs for Monday, January 5."
    assert spoken_date(date(2026, 11, 1)) == "Sunday, November 1"      # no leading zero, English names whatever the locale
    assert OUTRO_TEXT == "That's all for today. Tap a paper to read it."


# ------------------------------------------------------------------ sentences

def test_split_sentences_basic_and_covering():
    text = "Title here. From Ada Lovelace and colleagues. It works! Does it? Yes."
    spans = split_sentences(text)
    assert [text[a:b] for a, b in spans] == ["Title here.", "From Ada Lovelace and colleagues.", "It works!", "Does it?", "Yes."]
    assert sum(len(text[a:b].split()) for a, b in spans) == len(text.split())


def test_split_sentences_does_not_break_on_initials_abbreviations_or_numbers():
    text = "From J. R. Smith at U.S. Labs. Dr. Who agrees with version 2.5 of the model. Done."
    parts = [text[a:b] for a, b in split_sentences(text)]
    assert parts == ["From J. R. Smith at U.S. Labs.", "Dr. Who agrees with version 2.5 of the model.", "Done."]


def test_split_sentences_single_sentence_and_trailing_space():
    assert split_sentences("One sentence only") == [(0, 17)]
    assert split_sentences("One sentence only.  ") == [(0, 18)]
    assert split_sentences("") == []
