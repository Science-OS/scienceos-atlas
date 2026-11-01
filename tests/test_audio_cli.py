"""`atlas audio render|run|synth` with the fake engine: offline, no models, no network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from atlas import cli
from atlas.audio import render as render_mod
from atlas.audio import stage as audio_stage
from atlas.audio.render import load_papers, render
from atlas.audio.storage import manifest_clips
from atlas.audio.text import utf16_len
from atlas.config import load_config
from atlas.validate import audio_day_errors, make_validators
from audio_fakes import FakeAligner, FakeEngine, paper_json
from conftest import mock_transport
from test_audio_pipeline import seed_audio
from test_audio_summaries import GOOD, llm_handler
from test_build_validate import world  # noqa: F401  (fixture)

VAL = make_validators()["audio_day"]
DAY = "2026-10-21"


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(audio_stage, "load_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(audio_stage, "open_aligner", lambda cfg: FakeAligner())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the offline render touched the network")

    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx, "get", boom)


def last_json(out: str):
    """The CLI prints log lines and then one pretty-printed JSON document that starts on a line of its own."""
    lines = out.splitlines()
    start = max(i for i, ln in enumerate(lines) if ln == "{")
    return json.loads("\n".join(lines[start:]))


def write(tmp_path: Path, obj, name="papers.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def check(out: Path, day: str = DAY):
    doc = json.loads((out / f"{day}.json").read_text(encoding="utf-8"))
    assert audio_day_errors(doc, VAL, out / "clips", label="render", expect_date=day) == []
    assert {p.name for p in (out / "clips").glob("*.m4a")} == {f"{c['id']}.m4a" for c in manifest_clips(doc)}
    return doc


def test_render_writes_a_valid_manifest_and_exactly_the_clips_it_references(tmp_path, no_network, capsys):
    papers = [paper_json(1, title="Schrödinger–Gödel: Café équations", authors=[("Zoë Müller", "ETH Zürich"), ("B", None)]), paper_json(2), paper_json(3)]
    code = cli.main(["audio", "render", "--papers", str(write(tmp_path, papers)), "--date", DAY, "--out", str(tmp_path / "out")])
    report = last_json(capsys.readouterr().out)
    assert code == 0 and report["ok"] and report["papers"] == 3 and report["manifest"].endswith(f"{DAY}.json")
    doc = check(tmp_path / "out")
    assert list(doc["papers"]) == ["arxiv:2610.00001", "arxiv:2610.00002", "arxiv:2610.00003"] and len(doc["intros"]) == 8
    first = doc["papers"]["arxiv:2610.00001"]
    assert first["text"].startswith("Schrödinger–Gödel: Café équations. From Zoë Müller and colleagues at ETH Zürich. A short brief about the work.")
    raw = first["text"].encode("utf-16-le")
    assert first["words"][0][3] - first["words"][0][2] == len("Schrödinger–Gödel:") and raw[first["words"][1][2] * 2:first["words"][1][3] * 2].decode("utf-16-le") == "Café"
    assert first["sentences"][-1][3] == utf16_len(first["text"])
    assert doc["papers"]["arxiv:2610.00002"]["text"].startswith("Paper number 2. From Ada Lovelace and colleagues. A short brief about the work. This work studies a thing.")


def test_render_report_is_json_with_sizes(tmp_path, no_network):
    out = render(load_config(), write(tmp_path, [paper_json(1), paper_json(2)]), DAY, tmp_path / "o", engine=FakeEngine(), aligner=FakeAligner())
    assert out["ok"] and out["papers"] == 2 and out["clips"] == 11 and out["bytes"] == sum(p.stat().st_size for p in (tmp_path / "o").rglob("*") if p.is_file())
    assert out["summaries"] == {"llm": 0, "fallback": 2, "skipped_no_text": 0} and out["llm_enabled"] is False


def test_render_accepts_a_feed_shard_a_papers_wrapper_and_a_limit(tmp_path, no_network):
    shard = {"schema": "scienceos.atlas.feed-shard/1", "items": [{"paper": paper_json(i), "rank": i} for i in range(1, 6)]}
    assert [p["id"] for p, r in load_papers(write(tmp_path, shard))] == [f"arxiv:2610.{i:05d}" for i in range(1, 6)]
    assert load_papers(write(tmp_path, {"papers": [paper_json(1)]}))[0][1] == 1
    out = render(load_config(), write(tmp_path, shard), DAY, tmp_path / "o", limit=2, engine=FakeEngine(), aligner=FakeAligner())
    assert out["papers"] == 2
    assert list(check(tmp_path / "o")["papers"]) == ["arxiv:2610.00001", "arxiv:2610.00002"]


def test_render_select_applies_the_daily_selection(tmp_path, no_network):
    papers = [paper_json(i, cat="cs.LG" if i % 2 else "cs.CV") for i in range(1, 31)]
    cfg = load_config()
    cfg = cfg.model_copy(update={"audio": cfg.audio.model_copy(update={"top_overall": 3, "per_category": 1, "categories": 2})})
    out = render(cfg, write(tmp_path, papers), DAY, tmp_path / "o", apply_selection=True, engine=FakeEngine(), aligner=FakeAligner())
    assert out["papers"] == 3                                   # the category picks (the best LG and CV paper) are inside the top 3 already
    assert list(check(tmp_path / "o")["papers"]) == ["arxiv:2610.00001", "arxiv:2610.00002", "arxiv:2610.00003"]


def test_render_looks_up_first_author_institutions_from_a_site_directory_only_when_asked(tmp_path, no_network):
    site = tmp_path / "site" / "authors"
    site.mkdir(parents=True)
    (site / "A1.json").write_text(json.dumps({"author": {"affiliations": [{"name": "Inclusion AI"}]}}))
    p = paper_json(1, authors=[("Deyuan Liu", None), ("Other", None)])
    p["authors"][0]["authorId"] = "oa:A1"
    render(load_config(), write(tmp_path, [p]), DAY, tmp_path / "o1", engine=FakeEngine(), aligner=FakeAligner())
    assert " at Inclusion AI" not in check(tmp_path / "o1")["papers"]["arxiv:2610.00001"]["text"]
    render(load_config(), write(tmp_path, [p]), DAY, tmp_path / "o2", authors_from=str(tmp_path / "site"), engine=FakeEngine(), aligner=FakeAligner())
    assert "From Deyuan Liu and colleagues at Inclusion AI." in check(tmp_path / "o2")["papers"]["arxiv:2610.00001"]["text"]


def test_render_with_llm_uses_the_provider_and_the_cost_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls: list = []
    out = render(load_config(), write(tmp_path, [paper_json(1), paper_json(2)]), DAY, tmp_path / "o", llm=True, engine=FakeEngine(), aligner=FakeAligner(),
                 transport=mock_transport(llm_handler([GOOD, GOOD], calls)))
    assert len(calls) == 2 and out["summaries"]["llm"] == 2 and out["summaries"]["generated"] == 2 and out["llm_enabled"]
    assert GOOD in check(tmp_path / "o")["papers"]["arxiv:2610.00001"]["text"]


def test_render_without_llm_never_resolves_a_provider(tmp_path, monkeypatch, no_network):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")                 # a key in the environment is not enough: the render is offline unless asked

    def fail(*a, **k):
        raise AssertionError("provider resolved")

    monkeypatch.setattr(render_mod, "resolve_summary_provider", fail)
    assert render(load_config(), write(tmp_path, [paper_json(1)]), DAY, tmp_path / "o", engine=FakeEngine(), aligner=FakeAligner())["ok"]


def test_render_reports_failure_when_nothing_could_be_spoken(tmp_path, capsys):
    p = write(tmp_path, [paper_json(1)])
    out = render(load_config(), p, DAY, tmp_path / "o", engine=FakeEngine(fail_on=("e", "a", "o")), aligner=FakeAligner())
    assert out["ok"] is False and not (tmp_path / "o" / f"{DAY}.json").exists()


def test_render_skips_papers_with_nothing_to_say(tmp_path, no_network):
    papers = [paper_json(1, brief=None, abstract=None), paper_json(2)]
    out = render(load_config(), write(tmp_path, papers), DAY, tmp_path / "o", engine=FakeEngine(), aligner=FakeAligner())
    assert out["papers"] == 1 and out["summaries"]["skipped_no_text"] == 1


# ------------------------------------------------------------------ run / synth from a state DB

def test_audio_run_works_from_a_state_database(tmp_path, capsys):
    seed_audio(tmp_path / "state")
    code = cli.main(["audio", "run", "--state", str(tmp_path / "state"), "--audio-dir", str(tmp_path / "as"), "--no-llm", "--limit", "2"])
    out = last_json(capsys.readouterr().out)
    assert code == 0 and out["scripts"]["selected"] == 2 and out["scripts"]["summaries"]["fallback"] == 2 and out["synth"]["papers"] == 2
    days = sorted((tmp_path / "as" / "days").glob("*.json"))
    assert [d.stem for d in days] == ["2026-10-22"]
    doc = json.loads(days[0].read_text())
    assert len(doc["papers"]) == 2
    assert audio_day_errors(doc, VAL, tmp_path / "as" / "clips", label="x") == []


def test_audio_run_refuses_a_missing_state(tmp_path, capsys):
    assert cli.main(["audio", "run", "--state", str(tmp_path / "nope"), "--audio-dir", str(tmp_path / "as")]) == 2


def test_audio_synth_speaks_a_plan_and_writes_the_stats_file(tmp_path, capsys):
    from atlas.audio.plan import DayPlan, PlanPaper, day_texts
    from datetime import date
    intros, outro = day_texts(date(2026, 10, 22))
    plan = DayPlan("2026-10-22", [PlanPaper("arxiv:2610.00001", "2610.00001", "T", "T. From A. Body here.", "fallback", "top", 1)], intros, outro)
    plan.save(tmp_path / "plan.json")
    code = cli.main(["audio", "synth", "--plan", str(tmp_path / "plan.json"), "--audio-dir", str(tmp_path / "as"), "--stats-out", str(tmp_path / "stats.json"),
                     "--now", "2026-10-22T05:30:00Z"])
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert code == 0 and stats["built"] == 10 and stats["papers"] == 1
    assert json.loads((tmp_path / "as" / "days" / "2026-10-22.json").read_text())["generatedAt"] == "2026-10-22T05:30:00Z"


def test_audio_synth_reports_a_crash_in_the_stats_file(tmp_path, monkeypatch):
    from atlas.audio.plan import DayPlan
    DayPlan("2026-10-22", [], {}, "x").save(tmp_path / "plan.json")

    def boom(*a, **k):
        raise RuntimeError("model missing")

    monkeypatch.setattr(audio_stage, "load_engine", boom)
    code = cli.main(["audio", "synth", "--plan", str(tmp_path / "plan.json"), "--audio-dir", str(tmp_path / "as"), "--stats-out", str(tmp_path / "stats.json")])
    assert code == 1 and "model missing" in json.loads((tmp_path / "stats.json").read_text())["error"]


def test_validate_command_applies_the_audio_budget_from_the_config(world, tmp_path, capsys):
    from test_audio_site import build_with_audio, make_day
    make_day(tmp_path / "a")
    _, _, site = build_with_audio(world, tmp_path / "a")
    good = tmp_path / "good.yaml"
    good.write_text("audio:\n  max_site_mb: 50\n")
    assert cli.main(["--config", str(good), "validate", "--site", str(site)]) == 0
    tight = tmp_path / "tight.yaml"
    tight.write_text("audio:\n  max_site_mb: 0\n")
    capsys.readouterr()
    assert cli.main(["--config", str(tight), "validate", "--site", str(site)]) == 1
    assert "audio budget" in capsys.readouterr().out
