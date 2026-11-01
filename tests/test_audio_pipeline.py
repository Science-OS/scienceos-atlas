"""`atlas run` with an audio store: the stages after the briefs, the site copy, the status, and the promise that audio never fails the feed."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx

from atlas import cli
from atlas.audio import stage as audio_stage
from atlas.audio.plan import DayPlan
from atlas.audio.storage import AudioStore, manifest_clips
from atlas.config import load_config
from atlas.openalex import OpenAlexClient
from atlas.pipeline import RunOptions, run
from atlas.store import Store
from audio_fakes import FakeAligner, FakeEngine
from test_build_validate import NOW
from test_media_pipeline import Arxiv, seed_state


def seed_audio(state_dir):
    """The media-pipeline state (4 papers, Rod Page at ETH Zurich first author of one), all moved to the newest feed day so one day holds 4 papers."""
    seed_state(state_dir)
    store = Store(state_dir / "atlas.sqlite")
    store.conn.execute("UPDATE papers SET feed_date='2026-10-22'")
    store.conn.commit()
    store.close()


def cfg_with(**audio):
    cfg = load_config()
    return cfg.model_copy(update={"audio": cfg.audio.model_copy(update=audio)})


_DEFAULT = object()


def go(tmp_path, clock, *, cfg=None, engine=_DEFAULT, aligner=True, now=NOW, audio_dir=True, **opts):
    cfg = cfg or load_config()
    arxiv = Arxiv(clock)
    oa = OpenAlexClient(cfg.openalex, "ua", transport=httpx.MockTransport(lambda r: httpx.Response(500)), api_key="", sleep=lambda s: None)
    options = RunOptions(site_dir=tmp_path / "site", state_dir=tmp_path / "state", audio_dir=(tmp_path / "audio-store") if audio_dir else None, now=now,
                         skip_harvest=True, skip_hf=True, skip_openalex=True, skip_briefs=True, skip_thumbs=True, skip_media=True, **opts)
    res = run(cfg, options, arxiv_client=arxiv.client, hf_client=arxiv.client, oa_client=oa,
              audio_engine=FakeEngine() if engine is _DEFAULT else engine, audio_aligner=FakeAligner() if aligner is True else aligner)
    return res, arxiv


def latest_day(tmp_path) -> str:
    store = Store(tmp_path / "state" / "atlas.sqlite")
    try:
        return store.feed_dates()[-1]
    finally:
        store.close()


def test_audio_is_built_published_validated_and_reported(tmp_path, clock):
    seed_audio(tmp_path / "state")
    res, arxiv = go(tmp_path, clock)
    assert res.ok, res.stats["validation"]
    day = latest_day(tmp_path)
    site = tmp_path / "site"
    st = res.stats["stages"]
    assert st["audio_scripts"]["selected"] == st["audio_scripts"]["scripts"] == 4 and st["audio_scripts"]["summaries"]["fallback"] == 4
    assert st["audio"]["built"] == 13 and st["audio"]["failed"] == 0 and st["audio"]["papers"] == 4 and st["audio"]["aligned"] == 13
    doc = json.loads((site / "audio" / f"{day}.json").read_text())
    shard = json.loads((site / f"feeds/{day}.json").read_text())
    assert set(doc["papers"]) == {it["paper"]["id"] for it in shard["items"]}            # every paper of this small feed day got a clip
    assert {p.name for p in (site / "audio/clips").glob("*")} == {f"{c['id']}.m4a" for c in manifest_clips(doc)}
    status = json.loads((site / "status.json").read_text())["audio"]
    assert status["enabled"] and status["date"] == day and status["clipsBuilt"] == 13 and status["papersSelected"] == 4
    assert status["published"]["days"] == [day] and status["published"]["clips"] == 13 and status["summaries"] == {"llm": 0, "fallback": 4}
    assert res.stats["totals"]["audio_clips_published"] == 13 and not arxiv.urls


def test_the_script_names_the_first_author_with_the_institution_from_openalex(tmp_path, clock):
    from test_build_validate import NEW_ID
    seed_audio(tmp_path / "state")                                  # Rod Page at ETH Zurich is the first author of NEW_ID
    res, _ = go(tmp_path, clock)
    doc = json.loads((tmp_path / "site" / "audio" / f"{latest_day(tmp_path)}.json").read_text())
    text = doc["papers"][f"arxiv:{NEW_ID}"]["text"]
    assert "From Roderic D. M. Page and colleagues at ETH Zurich." in text
    assert text.index("From Roderic") > 5 and text.endswith(".")


def test_the_selected_papers_are_the_top_of_the_shard_the_phone_reads(tmp_path, clock):
    seed_audio(tmp_path / "state")
    res, _ = go(tmp_path, clock, cfg=cfg_with(top_overall=2, per_category=0, categories=0))
    day = latest_day(tmp_path)
    shard = json.loads((tmp_path / "site" / f"feeds/{day}.json").read_text())
    doc = json.loads((tmp_path / "site" / "audio" / f"{day}.json").read_text())
    assert list(doc["papers"]) == [it["paper"]["id"] for it in shard["items"][:2]]
    plan = DayPlan.load(tmp_path / "state" / "audio" / f"plan-{day}.json")
    assert [p.id for p in plan.papers] == list(doc["papers"]) and [p.rank for p in plan.papers] == [1, 2] and {p.why for p in plan.papers} == {"top"}


def test_a_later_run_reuses_every_clip(tmp_path, clock):
    seed_audio(tmp_path / "state")
    engine = FakeEngine()
    go(tmp_path, clock, engine=engine)
    calls = len(engine.calls)
    res, _ = go(tmp_path, clock, engine=engine, now=NOW + timedelta(hours=3))
    assert res.ok and len(engine.calls) == calls
    assert res.stats["stages"]["audio"]["built"] == 0 and res.stats["stages"]["audio"]["reused"] == 13


def test_a_failing_engine_never_fails_the_feed(tmp_path, clock):
    seed_audio(tmp_path / "state")
    res, _ = go(tmp_path, clock, engine=FakeEngine(fail_on=("e", "a", "o")))
    assert res.ok, res.stats["validation"]
    assert res.stats["stages"]["audio"]["failed"] > 0 and not (tmp_path / "site" / "audio").exists()
    status = json.loads((tmp_path / "site" / "status.json").read_text())["audio"]
    assert status["published"] == {"days": [], "clips": 0, "bytes": 0} and status["clipsFailed"] > 0


def test_a_crash_inside_the_stage_is_recorded_and_the_feed_still_ships(tmp_path, clock, monkeypatch):
    seed_audio(tmp_path / "state")

    def boom(*a, **k):
        raise RuntimeError("scripts exploded")

    monkeypatch.setattr(audio_stage, "stage_audio_scripts", boom)
    res, _ = go(tmp_path, clock)
    assert res.ok and "scripts exploded" in res.stats["stages"]["audio_scripts"]["error"] and "audio" not in res.stats["stages"]
    assert "scripts exploded" in json.loads((tmp_path / "site" / "status.json").read_text())["audio"]["note"]


def test_a_corrupt_store_is_left_out_of_the_site_not_the_feed(tmp_path, clock):
    seed_audio(tmp_path / "state")
    go(tmp_path, clock)
    store = AudioStore(tmp_path / "audio-store")
    victim = next(iter(manifest_clips(store.load_day(latest_day(tmp_path)))))
    store.clip_path(victim["id"]).write_bytes(b"rot")
    res, _ = go(tmp_path, clock, skip_audio=True)                                   # no stages: just the site copy
    assert res.ok and not (tmp_path / "site" / "audio").exists()
    assert "bytes/sha256" in json.loads((tmp_path / "site" / "status.json").read_text())["audio"]["published"]["skipped"][latest_day(tmp_path)]


def test_skip_audio_still_publishes_what_the_store_has(tmp_path, clock):
    seed_audio(tmp_path / "state")
    go(tmp_path, clock)
    (tmp_path / "state" / "audio").rename(tmp_path / "plans-gone")
    res, _ = go(tmp_path, clock, skip_audio=True, engine=FakeEngine(fail_on=("e",)))
    assert res.ok and "audio" not in res.stats["stages"] and (tmp_path / "site" / "audio" / f"{latest_day(tmp_path)}.json").is_file()


def test_audio_disabled_in_config_means_no_stages_and_nothing_published(tmp_path, clock):
    seed_audio(tmp_path / "state")
    go(tmp_path, clock)
    res, _ = go(tmp_path, clock, cfg=cfg_with(enabled="false"))
    assert res.ok and "audio" not in res.stats["stages"] and "audio_scripts" not in res.stats["stages"] and not (tmp_path / "site" / "audio").exists()
    assert "audio" not in json.loads((tmp_path / "site" / "status.json").read_text())


def test_no_audio_dir_means_no_audio_at_all(tmp_path, clock):
    seed_audio(tmp_path / "state")
    res, _ = go(tmp_path, clock, audio_dir=False)
    assert res.ok and "audio" not in res.stats["stages"] and not (tmp_path / "audio-store").exists() and not (tmp_path / "site" / "audio").exists()


def test_without_the_audio_extra_the_stage_says_so_and_the_run_is_unaffected(tmp_path, clock, monkeypatch):
    seed_audio(tmp_path / "state")
    monkeypatch.setattr(audio_stage, "engine_installed", lambda: False)
    arxiv = Arxiv(clock)
    cfg = load_config()
    oa = OpenAlexClient(cfg.openalex, "ua", transport=httpx.MockTransport(lambda r: httpx.Response(500)), api_key="", sleep=lambda s: None)
    res = run(cfg, RunOptions(site_dir=tmp_path / "site", state_dir=tmp_path / "state", audio_dir=tmp_path / "audio-store", now=NOW, skip_harvest=True, skip_hf=True,
                              skip_openalex=True, skip_briefs=True, skip_thumbs=True, skip_media=True),
              arxiv_client=arxiv.client, hf_client=arxiv.client, oa_client=oa)
    assert res.ok and res.stats["stages"]["audio"]["skipped"].startswith("the audio extra") and "audio_scripts" not in res.stats["stages"]
    assert "audio extra" in json.loads((tmp_path / "site" / "status.json").read_text())["audio"]["note"]


def test_is_enabled_matrix():
    cfg = load_config().audio
    assert audio_stage.is_enabled(cfg, True) == (True, None)
    assert audio_stage.is_enabled(cfg, False)[0] is False
    assert audio_stage.is_enabled(cfg.model_copy(update={"enabled": "false"}), True) == (False, "audio.enabled is false")
    assert audio_stage.is_enabled(cfg.model_copy(update={"enabled": True}), True) == (True, None)
    assert audio_stage.is_enabled(cfg.model_copy(update={"enabled": "true"}), False)[0] is False       # forced on but not installed: skipped, never fatal


def test_llm_summaries_are_used_when_a_key_exists_and_stored_once(tmp_path, clock, monkeypatch):
    from test_audio_summaries import GOOD, llm_handler
    from atlas.audio import summaries
    seed_audio(tmp_path / "state")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    real = summaries.resolve_provider
    monkeypatch.setattr(summaries, "resolve_provider", lambda cfg, env=None, transport=None: real(cfg, env, httpx.MockTransport(llm_handler([GOOD] * 10))))
    res, _ = go(tmp_path, clock)
    assert res.ok and res.stats["stages"]["audio_scripts"]["summaries"]["llm"] == 4 and res.stats["stages"]["audio_scripts"]["summaries"]["fallback"] == 0
    doc = json.loads((tmp_path / "site" / "audio" / f"{latest_day(tmp_path)}.json").read_text())
    assert all(GOOD in c["text"] for c in doc["papers"].values())
    store = Store(tmp_path / "state" / "atlas.sqlite")
    assert len(store.spoken_summary_rows()) == 4
    store.close()
    status = json.loads((tmp_path / "site" / "status.json").read_text())["audio"]
    assert status["summaries"]["llm"] == 4 and status["summaries"]["costUsd"] > 0


# ------------------------------------------------------------------ the isolated child process

def test_real_runs_synthesise_in_a_child_process_with_the_same_config(tmp_path, clock, monkeypatch):
    seed_audio(tmp_path / "state")
    seen = {}

    def fake_run(cmd, timeout=None, check=False, **kw):
        seen["cmd"], seen["timeout"] = cmd, timeout
        assert cmd[:3] == [sys.executable, "-m", "atlas"]
        monkeypatch.setattr(audio_stage, "load_engine", lambda *a, **k: FakeEngine())
        monkeypatch.setattr(audio_stage, "open_aligner", lambda cfg: FakeAligner())
        return SimpleNamespace(returncode=cli.main(cmd[3:]))            # run the child's command line in this process

    monkeypatch.setattr(audio_stage.subprocess, "run", fake_run)
    monkeypatch.setattr(audio_stage, "engine_installed", lambda: True)
    res, _ = go(tmp_path, clock, cfg=cfg_with(max_seconds=100), engine=None, audio_isolate=True)
    assert res.ok, res.stats["validation"]
    audio = res.stats["stages"]["audio"]
    assert audio["built"] == 13 and audio["papers"] == 4 and "child_seconds" in audio and "exit" not in audio
    assert "audio" in seen["cmd"] and "synth" in seen["cmd"] and seen["timeout"] == 700
    day = latest_day(tmp_path)
    assert len(json.loads((tmp_path / "site" / "audio" / f"{day}.json").read_text())["papers"]) == 4


def test_child_process_helper_round_trips_config_plan_and_stats(tmp_path, monkeypatch):
    cfg = cfg_with(max_seconds=100)
    plan = DayPlan("2026-10-22", [], {}, "x")
    plan_file = tmp_path / "state" / "audio" / "plan-2026-10-22.json"
    plan.save(plan_file)
    captured = {}

    def fake_run(cmd, timeout=None, check=False, **kw):
        captured["cmd"], captured["timeout"] = cmd, timeout
        cfg_file = cmd[cmd.index("--config") + 1]
        captured["cfg"] = load_config(cfg_file)                                     # JSON is YAML: the child loads the parent's exact config
        out = cmd[cmd.index("--stats-out") + 1]
        assert out == str(tmp_path / "state" / "audio" / "stats-2026-10-22.json")
        Path(out).write_text(json.dumps({"date": "2026-10-22", "built": 3}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(audio_stage.subprocess, "run", fake_run)
    stats = audio_stage.run_synth_child(cfg, plan_file, "2026-10-22", tmp_path / "state", tmp_path / "audio-store", NOW)
    assert stats["built"] == 3 and "exit" not in stats
    assert captured["cfg"] == cfg and captured["timeout"] == 100 + 600
    assert captured["cmd"][3:7] == ["--config", captured["cmd"][4], "audio", "synth"] and "--plan" in captured["cmd"]


def test_a_child_that_crashes_or_hangs_costs_only_the_audio(tmp_path, monkeypatch):
    cfg = cfg_with(max_seconds=1)
    plan_file = tmp_path / "state" / "audio" / "plan-2026-10-22.json"
    DayPlan("2026-10-22", [], {}, "x").save(plan_file)
    monkeypatch.setattr(audio_stage.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=-11))        # SIGSEGV in a native library
    stats = audio_stage.run_synth_child(cfg, plan_file, "2026-10-22", tmp_path / "state", tmp_path / "a", NOW)
    assert "without stats" in stats["error"] and stats["exit"] == -11

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("atlas", 601)

    monkeypatch.setattr(audio_stage.subprocess, "run", hang)
    stats = audio_stage.run_synth_child(cfg, plan_file, "2026-10-22", tmp_path / "state", tmp_path / "a", NOW)
    assert "without stats" in stats["error"] and stats["exit"] == "timeout"


def test_stage_audio_without_a_plan_says_so(tmp_path):
    assert "no plan" in audio_stage.stage_audio(load_config(), "2026-10-22", tmp_path / "state", tmp_path / "a", NOW)["skipped"]
