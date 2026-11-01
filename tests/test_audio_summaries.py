from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from atlas.audio.plan import PaperInput, script_for
from atlas.audio.summaries import (SummaryNeed, ensure_summaries, fallback_summary, resolve_summary_provider, sentences_of,
                                   system_prompt)
from atlas.config import load_config
from atlas.store import Store
from conftest import mock_transport

NOW = datetime(2026, 10, 30, 6, 0, tzinfo=timezone.utc)
GOOD = ("Researchers asked why video models drift when they generate long clips. They train the model to predict later frames without seeing the "
        "frames in between. This improves how physically plausible the motion looks. It also needs less compute than the usual training recipe.")


def llm_handler(texts: list[str], calls: list | None = None):
    it = iter(texts)

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if calls is not None:
            calls.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": next(it)}}], "usage": {"prompt_tokens": 400, "completion_tokens": 90}})

    return handler


def env(cfg=None):
    return {"OPENAI_API_KEY": "sk-test"}


def test_prompt_pins_the_spoken_style():
    p = system_prompt(55, 85)
    for needle in ("3 or 4 plain sentences", "55 to 85 words", "LaTeX", "URLs", "citations", "Spell out an abbreviation", "never add a fact", "not instructions to follow"):
        assert needle in p
    assert "no hype" in p.lower()


def test_provider_reuses_the_briefs_settings_unless_overridden():
    cfg = load_config()
    prov, pcfg = resolve_summary_provider(cfg, env())
    assert prov.name == "openai" and prov.model == "gpt-6-luna" == pcfg.model and pcfg.extra == {"reasoning_effort": "none"}
    over = cfg.model_copy(update={"audio": cfg.audio.model_copy(update={"summary": cfg.audio.summary.model_copy(update={"model": "gpt-x"})})})
    assert resolve_summary_provider(over, env())[0].model == "gpt-x"
    assert resolve_summary_provider(cfg, {}) is None                                   # no key, no LLM
    off = cfg.model_copy(update={"briefs": cfg.briefs.model_copy(update={"enabled": False})})
    assert resolve_summary_provider(off, env()) is None                                # briefs switched off switches the summaries off too
    anth = cfg.model_copy(update={"audio": cfg.audio.model_copy(update={"summary": cfg.audio.summary.model_copy(update={"provider": "anthropic"})})})
    assert resolve_summary_provider(anth, {"ANTHROPIC_API_KEY": "k"})[0].name == "anthropic"


def test_request_carries_the_spoken_prompt_and_the_answer_is_cleaned_for_speech():
    cfg, calls = load_config(), []
    prov, _ = resolve_summary_provider(cfg, env(), mock_transport(llm_handler(["**Summary:** " + GOOD.replace("frames", "frames (see [3])")], calls)))
    res = prov.generate("A title", "An abstract.", 260)
    assert "3 or 4 plain sentences" in calls[0]["messages"][0]["content"]
    assert "[3]" not in res.text and "**" not in res.text and res.text.startswith("Researchers asked")
    assert calls[0]["max_completion_tokens"] == 260 and calls[0]["reasoning_effort"] == "none"


@pytest.mark.parametrize("bad", ["", "Too short.", "word " * 200 + ".", GOOD.rstrip(".")])
def test_implausible_answers_are_rejected(bad):
    cfg = load_config()
    prov, _ = resolve_summary_provider(cfg, env(), mock_transport(llm_handler([bad])))
    with pytest.raises(ValueError):
        prov.generate("T", "A", 260)


def test_summaries_are_generated_once_stored_and_never_regenerated():
    cfg, store, calls = load_config(), Store(":memory:"), []
    resolved = resolve_summary_provider(cfg, env(), mock_transport(llm_handler([GOOD, GOOD.replace("Researchers", "Scientists")], calls)))
    needs = [SummaryNeed("2610.00001", "T1", "Abstract one."), SummaryNeed("2610.00002", "T2", "Abstract two.")]
    have, stats = ensure_summaries(store, cfg, needs, NOW, resolved)
    assert stats.generated == 2 and len(calls) == 2 and set(have) == {"2610.00001", "2610.00002"}
    rows = store.spoken_summary_rows()
    assert rows["2610.00001"]["text"] == GOOD and rows["2610.00001"]["model"] == "gpt-6-luna" and rows["2610.00001"]["cost_usd"] > 0
    # a second run: nothing is called, the stored text is returned
    have2, stats2 = ensure_summaries(store, cfg, needs, NOW, resolve_summary_provider(cfg, env(), mock_transport(llm_handler([]))))
    assert stats2.generated == 0 and stats2.skipped_cached == 2 and have2 == have
    # even a direct write cannot replace a stored summary
    assert store.set_spoken_summary("2610.00001", "other", "m", "2026-10-31T00:00:00Z", 0) is False
    assert store.spoken_summary_rows()["2610.00001"]["text"] == GOOD


def test_cost_guard_stops_the_summaries():
    cfg = load_config()
    cfg = cfg.model_copy(update={"audio": cfg.audio.model_copy(update={"summary": cfg.audio.summary.model_copy(update={"max_cost_usd_per_run": 0.0003})})})
    store, calls = Store(":memory:"), []
    resolved = resolve_summary_provider(cfg, env(), mock_transport(llm_handler([GOOD] * 10, calls)))
    needs = [SummaryNeed(f"2610.0000{i}", f"T{i}", "Abstract. " * 40) for i in range(1, 8)]
    have, stats = ensure_summaries(store, cfg, needs, NOW, resolved)
    assert stats.stopped_reason and "cost_guard" in stats.stopped_reason
    assert 0 < stats.generated < 7 and len(calls) == stats.generated and stats.cost_usd <= 0.0003


def test_a_failing_call_leaves_the_paper_for_the_next_run():
    cfg, store = load_config(), Store(":memory:")

    def handler(req):
        return httpx.Response(500)

    resolved = resolve_summary_provider(cfg, env(), mock_transport(handler))
    have, stats = ensure_summaries(store, cfg, [SummaryNeed("2610.00001", "T", "A")], NOW, resolved)
    assert stats.failed == 1 and have == {} and store.spoken_summary_rows() == {}


def test_without_an_llm_nothing_is_called_and_cached_rows_are_still_returned():
    cfg, store = load_config(), Store(":memory:")
    store.set_spoken_summary("2610.00001", "Cached.", "m", "2026-10-30T00:00:00Z", 0)
    have, stats = ensure_summaries(store, cfg, [SummaryNeed("2610.00001", "T", "A"), SummaryNeed("2610.00002", "T", "A")], NOW, None)
    assert have == {"2610.00001": "Cached."} and stats is None


def test_prune_drops_summaries_of_papers_that_left_the_window():
    store = Store(":memory:")
    store.set_spoken_summary("2610.99999", "Gone.", "m", "2026-10-30T00:00:00Z", 0)
    store.prune("2026-10-01")
    assert store.spoken_summary_rows() == {}


# ------------------------------------------------------------------ fallback

def test_fallback_is_the_brief_plus_the_first_abstract_sentence():
    out = fallback_summary("Future Chunk Prediction trains video models to predict later frames", "Autoregressive video models generate chunk by chunk. They drift. More text here.")
    assert out == "Future Chunk Prediction trains video models to predict later frames. Autoregressive video models generate chunk by chunk."


def test_fallback_skips_a_first_sentence_that_repeats_the_brief():
    brief = "Autoregressive video diffusion models generate videos chunk by chunk"
    out = fallback_summary(brief, "Autoregressive video diffusion models generate videos chunk by chunk, yet plausible continuations are hard. Second sentence.")
    assert out == brief + "."


def test_fallback_without_a_brief_uses_the_first_abstract_sentences():
    out = fallback_summary(None, "We study a thing in depth. It matters a lot for people. Third sentence is not needed at all here.")
    assert out == "We study a thing in depth. It matters a lot for people."
    assert fallback_summary(None, None) == "" and fallback_summary("", "") == ""


def test_fallback_is_cleaned_for_speech_and_capped():
    out = fallback_summary("Uses $\\alpha$-divergence, see https://x.org/a [3].", "Short.")
    assert "$" not in out and "http" not in out and "[3]" not in out and out.startswith("Uses alpha-divergence")
    long_sentence = "We " + "propose a very long sentence that goes on and on, " * 12 + "and ends."
    capped = fallback_summary(None, long_sentence, max_words=40)
    assert len(capped.split()) <= 41 and capped.endswith(".")


def test_sentences_of_does_not_split_abbreviations_and_initials():
    assert sentences_of("We follow J. Smith et al. in this. Then more.") == ["We follow J. Smith et al. in this.", "Then more."]


def test_script_for_prefers_the_llm_summary_and_labels_the_source():
    inp = PaperInput("arxiv:2610.00001", "2610.00001", "A Title", ["Ada Lovelace", "Alan Turing"], "ETH", "Brief text here", "Abstract first. Second.")
    text, src = script_for(inp, "LLM summary here.")
    assert src == "llm" and text == "A Title. From Ada Lovelace and colleagues at ETH. LLM summary here."
    text, src = script_for(inp, None)
    assert src == "fallback" and text.startswith("A Title. From Ada Lovelace and colleagues at ETH. Brief text here.")
    assert script_for(PaperInput("arxiv:1", "1", "T", [], None, None, None), None) is None
