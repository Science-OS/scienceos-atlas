from __future__ import annotations

import json
import shutil
import sqlite3
from types import SimpleNamespace

import pytest

from atlas import cli
from atlas.media.store import MediaStore
from media_fakes import BASE, PAGE, FakeNet, make_http
from test_media_pipeline import SECOND_CC_BY, seed_state
from test_build_validate import NEW_ID


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("media:\n  min_interval_s: 0\n  queue:\n    url: ''\n")
    return str(p)


@pytest.fixture
def net(monkeypatch):
    net = FakeNet()
    monkeypatch.setattr(cli, "MediaHttp", SimpleNamespace(create=lambda cfg: make_http(cfg, net)))
    return net


def test_backfill_resolves_every_author_and_institution_in_the_state_and_moves_the_thumbnails(tmp_path, config, net, capsys):
    seed_state(tmp_path / "state")
    code = cli.main(["--config", config, "media", "backfill", "--from-db", str(tmp_path / "state" / "atlas.sqlite"), "--media-dir", str(tmp_path / "ms")])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["targets"] == 3 and out["processed_author"] == 2 and out["processed_institution"] == 1 and out["found"] == 2
    assert out["thumbnails"]["ingested"] == 2 and out["manifest"]["counts"]["paper"] == 2 and out["manifest"]["counts"]["author"] == 1
    store = MediaStore(tmp_path / "ms", BASE)
    assert store.read_entry("author", f"wikidata-{PAGE['qid']}")["assets"][0]["hosted"]
    assert store.read_entry("paper", f"arxiv-{NEW_ID}") and store.read_entry("paper", f"arxiv-{SECOND_CC_BY}")


def test_backfill_is_resumable_and_honours_a_limit(tmp_path, config, net, capsys):
    seed_state(tmp_path / "state")
    db = str(tmp_path / "state" / "atlas.sqlite")
    args = ["--config", config, "media", "backfill", "--from-db", db, "--media-dir", str(tmp_path / "ms")]
    cli.main([*args, "--limit", "1"])
    first = json.loads(capsys.readouterr().out)
    assert first["budget_exhausted"] and first["targets"] == 3
    cli.main(args)
    second = json.loads(capsys.readouterr().out)
    assert second["fresh"] == 1 and second["processed_author"] + second.get("processed_institution", 0) == 2     # finished entities are skipped
    cli.main(args)
    assert json.loads(capsys.readouterr().out)["fresh"] == 3


def test_backfill_needs_an_existing_database(tmp_path, config, net, capsys):
    assert cli.main(["--config", config, "media", "backfill", "--from-db", str(tmp_path / "nope.sqlite")]) == 2


def test_backfill_on_a_copy_leaves_the_original_state_alone(tmp_path, config, net, capsys):
    seed_state(tmp_path / "state")
    copy = tmp_path / "copy"
    shutil.copytree(tmp_path / "state", copy)
    cli.main(["--config", config, "media", "backfill", "--from-db", str(copy / "atlas.sqlite"), "--media-dir", str(tmp_path / "ms")])
    db = sqlite3.connect(tmp_path / "state" / "atlas.sqlite")
    assert db.execute("SELECT COUNT(*) FROM media_checks").fetchone()[0] == 0
    db.close()


def test_run_summary_shows_the_media_stage_and_survives_its_failure():
    base = {"stages": {"harvest": {}, "openalex": {}, "thumbs": {}, "briefs": {}}, "totals": {}, "validation": {}}
    ok = cli._summary_markdown({**base, "stages": {**base["stages"], "media": {"processed_author": 4, "processed_institution": 2, "found": 3, "negative": 3,
                                                                            "queue": {"rows": 10, "cursor_after": 99}}}})
    assert "4 authors, 2 institutions" in ok and "3 / 3" in ok and "cursor_after" in ok
    failed = cli._summary_markdown({**base, "stages": {**base["stages"], "media": {"error": "RuntimeError: boom"}}})
    assert "RuntimeError: boom" in failed
    assert "off" in cli._summary_markdown(base)


def test_offline_run_never_touches_the_media_sources(tmp_path, config, net, capsys):
    seed_state(tmp_path / "state")
    code = cli.main(["--config", config, "run", "--offline", "--state", str(tmp_path / "state"), "--site", str(tmp_path / "site"),
                     "--media-dir", str(tmp_path / "ms")])
    assert code == 0 and not net.requests and not (tmp_path / "ms" / "c").exists()


def test_run_with_a_media_dir_resolves_and_publishes(tmp_path, config, net, capsys):
    seed_state(tmp_path / "state")
    code = cli.main(["--config", config, "run", "--skip-harvest", "--skip-hf", "--skip-openalex", "--skip-thumbs", "--skip-briefs",
                     "--state", str(tmp_path / "state"), "--site", str(tmp_path / "site"), "--media-dir", str(tmp_path / "ms")])
    printed = capsys.readouterr().out                                    # log lines first, then the JSON result
    out = json.loads(printed[printed.index('{\n  "ok"'):])
    assert code == 0 and out["ok"] and out["totals"]["authors_with_photo"] == 1
    assert (tmp_path / "site" / "media" / "manifest.json").is_file() and (tmp_path / "ms" / "c" / "institution" / "ror-05a28rw58.json").is_file()
    meta = next((tmp_path / "state" / "runs").glob("*.meta.json"))
    assert json.loads(meta.read_text())["stats"]["stages"]["media"]["found"] == 2
