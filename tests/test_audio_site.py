"""Day synthesis (with fake engine and aligner), the audio store, the copy into the site, and what `validate_site` says about it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas.audio import SCHEMA_ID
from atlas.audio.plan import DayPlan, PlanPaper, day_texts
from atlas.audio.site import publish_to_site
from atlas.audio.storage import AudioStore, dump_manifest, manifest_clips, sha256_hex
from atlas.audio.synth import synthesize_plan
from atlas.config import AudioCfg, load_config
from atlas.validate import audio_day_errors, make_validators, validate_site
from audio_fakes import FakeAligner, FakeEngine
from test_build_validate import NOW, load, world  # noqa: F401  (world is a fixture)

DAY = "2026-10-22"
NOWDT = datetime(2026, 10, 22, 5, 30, tzinfo=timezone.utc)
VAL = make_validators()["audio_day"]


def plan_for(day: str = DAY, n: int = 3, prefix: str = "P") -> DayPlan:
    from datetime import date
    intros, outro = day_texts(date.fromisoformat(day))
    papers = [PlanPaper(f"arxiv:2610.{i:05d}", f"2610.{i:05d}", f"{prefix} {i}", f"{prefix} {i}. From Ada and colleagues. Sentence number {i} is here.", "fallback", "top", i)
              for i in range(1, n + 1)]
    return DayPlan(day, papers, intros, outro)


def make_day(root: Path, day: str = DAY, n: int = 3, prefix: str = "P", cfg: AudioCfg | None = None, engine=None, aligner=True, now=NOWDT):
    store = AudioStore(root)
    stats = synthesize_plan(plan_for(day, n, prefix), cfg or AudioCfg(), store, engine or FakeEngine(), FakeAligner() if aligner is True else aligner, now)
    return store, stats


# ------------------------------------------------------------------ synthesis

def test_a_day_has_intros_for_one_to_eight_an_outro_and_a_clip_per_paper(tmp_path):
    store, stats = make_day(tmp_path)
    doc = store.load_day(DAY)
    assert doc["schema"] == SCHEMA_ID and doc["date"] == DAY and doc["gapMs"] == 600
    assert doc["engine"] == {"name": "fake-tts", "version": "1.0.0", "model": "fake/model", "voice": "alba", "credit": "Voice: Alba · Pocket TTS by Kyutai (CC BY 4.0)"}
    assert sorted(doc["intros"], key=int) == [str(i) for i in range(1, 9)]
    assert doc["intros"]["6"]["text"] == "Here are your six briefs for Thursday, October 22."
    assert doc["outro"]["text"] == "That's all for today. Tap a paper to read it."
    assert list(doc["papers"]) == ["arxiv:2610.00001", "arxiv:2610.00002", "arxiv:2610.00003"]
    assert doc["generatedAt"] == "2026-10-22T05:30:00Z"
    assert stats.built == 12 and stats.reused == 0 and stats.failed == 0 and stats.aligned == 12 and stats.papers == 3 and stats.intros == 8 and stats.outro
    assert audio_day_errors(doc, VAL, store.clips_dir, label="t", expect_date=DAY) == []
    assert len(list(store.clips_dir.glob("*.m4a"))) == 12


def test_a_second_run_of_the_same_day_synthesises_nothing(tmp_path):
    engine = FakeEngine()
    store, _ = make_day(tmp_path, engine=engine)
    calls = len(engine.calls)
    first = store.day_path(DAY).read_bytes()
    _, stats = make_day(tmp_path, engine=engine, now=datetime(2026, 10, 22, 9, 0, tzinfo=timezone.utc))
    assert len(engine.calls) == calls and stats.built == 0 and stats.reused == 12
    doc = json.loads(store.day_path(DAY).read_text())
    assert doc["generatedAt"] == "2026-10-22T09:00:00Z" and doc["papers"] == json.loads(first)["papers"]


def test_a_paper_that_stays_in_the_feed_is_reused_on_the_next_day(tmp_path):
    engine = FakeEngine()
    store, _ = make_day(tmp_path, "2026-10-21", engine=engine)
    calls = len(engine.calls)
    plan = plan_for("2026-10-22")
    store2 = AudioStore(tmp_path)
    stats = synthesize_plan(plan, AudioCfg(), store2, engine, FakeAligner(), NOWDT)
    # same paper texts and the outro are reused; the 8 intros carry the date and are new
    assert stats.reused == 4 and stats.built == 8 and len(engine.calls) > calls
    assert store2.days() == ["2026-10-21", "2026-10-22"]


def test_one_failing_clip_costs_only_that_clip(tmp_path):
    store, stats = make_day(tmp_path, engine=FakeEngine(fail_on=("number 2",)))
    doc = store.load_day(DAY)
    assert stats.failed == 1 and "arxiv:2610.00002" not in doc["papers"] and len(doc["papers"]) == 2
    assert audio_day_errors(doc, VAL, store.clips_dir, label="t") == []


def test_nothing_usable_writes_no_manifest(tmp_path):
    store, stats = make_day(tmp_path, engine=FakeEngine(fail_on=("e",)))        # every text contains an "e"
    assert store.days() == [] and stats.papers == 0 and "manifest not written" in stats.stopped_reason


def test_time_budget_stops_before_the_next_paper_but_keeps_what_exists(tmp_path, monkeypatch):
    ticks = iter(range(100))
    monkeypatch.setattr("atlas.audio.synth.time", SimpleNamespace(monotonic=lambda: next(ticks)))
    store = AudioStore(tmp_path)
    stats = synthesize_plan(plan_for(n=5), AudioCfg(), store, FakeEngine(), FakeAligner(), NOWDT, deadline=3.5)
    doc = store.load_day(DAY)
    assert stats.papers == 3 and stats.skipped_budget == 2 and "time budget" in stats.stopped_reason
    assert list(doc["papers"]) == ["arxiv:2610.00001", "arxiv:2610.00002", "arxiv:2610.00003"] and len(doc["intros"]) == 8
    # the next run (a fresh budget) finishes the day without redoing the first three papers
    stats2 = synthesize_plan(plan_for(n=5), AudioCfg(), store, FakeEngine(), FakeAligner(), NOWDT)
    assert stats2.reused == 12 and stats2.built == 2 and len(store.load_day(DAY)["papers"]) == 5


def test_papers_an_earlier_run_finished_survive_when_this_run_fails_them(tmp_path):
    store, _ = make_day(tmp_path)
    old = store.load_day(DAY)["papers"]["arxiv:2610.00002"]
    plan = plan_for()
    plan.papers[1].text = plan.papers[1].text + " Changed and boom."              # new text, new clip id, and the engine will fail on it
    synthesize_plan(plan, AudioCfg(), store, FakeEngine(fail_on=("boom",)), FakeAligner(), NOWDT)
    assert store.load_day(DAY)["papers"]["arxiv:2610.00002"] == old


def test_clips_of_another_engine_version_are_not_carried_into_the_new_manifest(tmp_path):
    store, _ = make_day(tmp_path)
    old_engine = FakeEngine()
    old_engine.version = "0.9.0"
    stats = synthesize_plan(plan_for(), AudioCfg(), store, old_engine, FakeAligner(), NOWDT)         # same texts, other version: new ids, all new
    doc = store.load_day(DAY)
    assert stats.built == 12 and doc["engine"]["version"] == "0.9.0"
    assert audio_day_errors(doc, VAL, store.clips_dir, label="t") == []
    newer = FakeEngine(fail_on=("number 2",))                                                        # back on 1.0.0, and paper 2 fails: the 0.9.0 clip must not be carried
    stats = synthesize_plan(plan_for(), AudioCfg(), store, newer, FakeAligner(), NOWDT)
    doc = store.load_day(DAY)
    assert doc["engine"]["version"] == "1.0.0" and list(doc["papers"]) == ["arxiv:2610.00001", "arxiv:2610.00003"] and stats.failed == 1
    assert audio_day_errors(doc, VAL, store.clips_dir, label="t") == []


def test_a_run_that_is_killed_halfway_leaves_a_consistent_playable_day(tmp_path):
    class Killed(FakeEngine):
        def synthesize(self, text):
            if "number 3" in text:
                raise KeyboardInterrupt
            return super().synthesize(text)

    store = AudioStore(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        synthesize_plan(plan_for(), AudioCfg(), store, Killed(), FakeAligner(), NOWDT)
    doc = store.load_day(DAY)
    assert list(doc["papers"]) == ["arxiv:2610.00001", "arxiv:2610.00002"]
    assert audio_day_errors(doc, VAL, store.clips_dir, label="t") == []


def test_retention_prunes_old_days_and_the_clips_only_they_used(tmp_path):
    cfg = AudioCfg(retention_days=3)
    engine = FakeEngine()
    for i, day in enumerate(["2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]):
        store, stats = make_day(tmp_path, day, n=2, prefix=f"D{i}", cfg=cfg, engine=engine)
    assert store.days() == ["2026-10-20", "2026-10-21", "2026-10-22"]
    ids = {c["id"] for d in store.days() for c in manifest_clips(store.load_day(d))}
    assert {p.stem for p in store.clips_dir.glob("*.m4a")} == ids                     # nothing orphaned, nothing referenced is gone
    assert stats.pruned["days"] == 1 and stats.pruned["clips"] >= 1


# ------------------------------------------------------------------ site copy and validation

def build_with_audio(world, audio_root: Path, cfg=None):  # noqa: F811
    from atlas.build import SiteBuilder
    store, _builder, site = world
    cfg = cfg or load_config()
    b = SiteBuilder(store, cfg, site, site.parent / "state" / "thumbs", NOWDT, None, AudioStore(audio_root))
    stats = b.build()
    b.write_status({"stages": {}})
    return b, stats, site


def test_site_gets_the_manifest_and_exactly_the_clips_it_references(world, tmp_path):
    make_day(tmp_path / "a")
    b, stats, site = build_with_audio(world, tmp_path / "a")
    doc = load(site, f"audio/{DAY}.json")
    assert doc["papers"]["arxiv:2610.00001"]["url"].startswith("audio/clips/")
    assert {p.name for p in (site / "audio/clips").glob("*")} == {f"{c['id']}.m4a" for c in manifest_clips(doc)}
    assert stats.audio["days"] == [DAY] and stats.audio["clips"] == 12 and stats.audio["skipped"] == {}
    report = validate_site(site)
    assert report.ok, report.errors
    assert report.checked["audio_days"] == 1 and report.checked["audio_clips"] == 12
    status = load(site, "status.json")
    assert status["audio"]["enabled"] is True and status["audio"]["published"]["days"] == [DAY] and status["audio"]["published"]["clips"] == 12


def test_the_published_manifest_is_byte_identical_to_the_branch_copy(world, tmp_path):
    store, _ = make_day(tmp_path / "a")
    _, _, site = build_with_audio(world, tmp_path / "a")
    assert (site / "audio" / f"{DAY}.json").read_bytes() == store.day_path(DAY).read_bytes()
    for c in manifest_clips(store.load_day(DAY)):
        assert sha256_hex((site / c["url"]).read_bytes()) == c["sha256"]


def test_without_an_audio_store_there_is_no_audio_section_in_the_status(world):
    store, builder, site = world
    builder.build()
    builder.write_status({})
    assert "audio" not in load(site, "status.json") and not (site / "audio").exists() and validate_site(site).ok


def test_a_day_with_a_missing_or_damaged_clip_is_left_out_not_fatal(world, tmp_path):
    store, _ = make_day(tmp_path / "a", DAY)
    make_day(tmp_path / "a", "2026-10-21", prefix="Q")
    victim = store.load_day(DAY)["papers"]["arxiv:2610.00002"]
    store.clip_path(victim["id"]).write_bytes(b"damaged")
    _, stats, site = build_with_audio(world, tmp_path / "a")
    assert stats.audio["days"] == ["2026-10-21"] and DAY in stats.audio["skipped"] and "bytes/sha256" in stats.audio["skipped"][DAY]
    assert validate_site(site).ok and not (site / "audio" / f"{DAY}.json").exists()
    assert load(site, "status.json")["audio"]["published"]["skipped"][DAY]


def test_budget_keeps_the_newest_days_first(tmp_path):
    for day, p in (("2026-10-20", "A"), ("2026-10-21", "B"), (DAY, "C")):
        make_day(tmp_path / "a", day, prefix=p)
    store = AudioStore(tmp_path / "a")
    sizes = {d: len(store.day_path(d).read_bytes()) + sum(c["bytes"] for c in manifest_clips(store.load_day(d))) for d in store.days()}
    sized = lambda nbytes: SimpleNamespace(retention_days=8, max_site_mb=nbytes / (1024 * 1024))     # a budget in bytes (the real one is whole MB)
    out = publish_to_site(store, tmp_path / "s1", DAY, sized(sizes[DAY] + sizes["2026-10-21"] + 10), VAL)
    assert out["days"] == ["2026-10-21", DAY] and "over the" in out["skipped"]["2026-10-20"]
    assert {p.stem for p in (tmp_path / "s1/audio/clips").glob("*")} == {c["id"] for d in out["days"] for c in manifest_clips(store.load_day(d))}
    none = publish_to_site(store, tmp_path / "s2", DAY, sized(10), VAL)
    assert none["days"] == [] and set(none["skipped"]) == set(store.days())
    full = publish_to_site(store, tmp_path / "s3", DAY, AudioCfg(), VAL)
    assert full["days"] == store.days() and full["bytes"] < sum(sizes.values())      # the outro is the same clip on every day: stored and counted once


def test_days_outside_the_retention_window_or_after_the_latest_feed_day_are_not_published(tmp_path):
    for day in ("2026-10-10", "2026-10-21", DAY, "2026-10-25"):
        make_day(tmp_path / "a", day, prefix=day[-2:], cfg=AudioCfg(retention_days=100))      # the store itself keeps them all
    cfg = AudioCfg(retention_days=3)
    out = publish_to_site(AudioStore(tmp_path / "a"), tmp_path / "s", DAY, cfg, VAL)
    assert out["days"] == ["2026-10-21", DAY] and set(out["skipped"]) == {"2026-10-10", "2026-10-25"}


# ------------------------------------------------------------------ what validate_site catches

@pytest.fixture
def published(world, tmp_path):  # noqa: F811
    make_day(tmp_path / "a")
    _, _, site = build_with_audio(world, tmp_path / "a")
    assert validate_site(site).ok
    return site


def edit_manifest(site: Path, fn):
    p = site / "audio" / f"{DAY}.json"
    doc = json.loads(p.read_text())
    fn(doc)
    p.write_bytes(dump_manifest(doc))


def errors(site: Path, **kw) -> list[str]:
    return validate_site(site, **kw).errors


def test_validate_catches_a_missing_clip(published):
    doc = json.loads((published / "audio" / f"{DAY}.json").read_text())
    (published / doc["outro"]["url"]).unlink()
    assert any("file is missing" in e for e in errors(published))


def test_validate_catches_changed_bytes(published):
    doc = json.loads((published / "audio" / f"{DAY}.json").read_text())
    f = published / doc["papers"]["arxiv:2610.00001"]["url"]
    f.write_bytes(f.read_bytes()[:-1] + b"\x00")
    assert any("bytes/sha256 differ" in e for e in errors(published))


def test_validate_catches_a_wrong_utf16_span(published):
    edit_manifest(published, lambda d: d["papers"]["arxiv:2610.00001"]["words"][1].__setitem__(2, d["papers"]["arxiv:2610.00001"]["words"][1][2] + 1))
    assert any("UTF-16 span" in e for e in errors(published))


def test_validate_catches_a_non_monotonic_word(published):
    def swap(doc):
        w = doc["papers"]["arxiv:2610.00001"]["words"]
        w[2][:2], w[3][:2] = w[3][:2], w[2][:2]

    edit_manifest(published, swap)
    assert any("not monotonic" in e for e in errors(published))


def test_validate_catches_a_word_past_the_end_of_the_clip(published):
    def too_long(doc):
        c = doc["papers"]["arxiv:2610.00001"]
        c["words"][-1][1] = c["durationMs"] + 5

    edit_manifest(published, too_long)
    assert any("outside 0-" in e for e in errors(published))


def test_validate_catches_missing_words_a_forged_id_and_a_wrong_date(published):
    edit_manifest(published, lambda d: d["papers"]["arxiv:2610.00001"]["words"].pop())
    assert any("timings for" in e for e in errors(published))

    def forge(doc):
        doc["papers"]["arxiv:2610.00001"]["text"] += " Extra."

    edit_manifest(published, forge)
    assert any("id is not sha256" in e for e in errors(published))
    edit_manifest(published, lambda d: d.__setitem__("date", "2026-10-21"))
    assert any("does not match the file name" in e for e in errors(published))


def test_validate_catches_schema_violations(published):
    edit_manifest(published, lambda d: d.__setitem__("gapMs", -5))
    assert any("gapMs" in e for e in errors(published))
    edit_manifest(published, lambda d: (d.__setitem__("gapMs", 600), d["intros"].__setitem__("9", d["outro"])))
    assert any("intros" in e for e in errors(published))


def test_validate_catches_orphan_clips_and_the_size_budget(published):
    (published / "audio" / "clips" / ("0" * 32 + ".m4a")).write_bytes(b"x")
    assert any("not referenced" in e for e in errors(published))
    (published / "audio" / "clips" / ("0" * 32 + ".m4a")).unlink()
    assert validate_site(published).ok
    assert any("audio budget" in e for e in errors(published, audio_max_mb=0))
    assert any("budget" in e for e in errors(published, max_site_mb=0))
