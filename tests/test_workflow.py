"""The daily workflow stays thin: every step with logic is a CLI subcommand, and the order of the steps is what the contract needs."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from atlas.cli import main

WF = yaml.safe_load((Path(__file__).parents[1] / ".github/workflows/daily.yml").read_text())
BUILD = WF["jobs"]["build"]
STEPS = {s.get("name") or s.get("uses"): s for s in BUILD["steps"]}
ORDER = [s.get("name") or s.get("uses") for s in BUILD["steps"]]


def test_build_job_can_push_the_media_and_state_branches():
    assert BUILD["permissions"]["contents"] == "write" and BUILD["permissions"]["actions"] == "write"
    assert WF["permissions"]["contents"] == "read"                      # everything else stays read-only
    assert WF["jobs"]["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}


def test_state_is_restored_before_the_run_and_snapshotted_after_it():
    assert ORDER.index("Restore state") < ORDER.index("Restore state snapshot") < ORDER.index("Check out the media branch") < ORDER.index("Ingest, build, validate")
    assert ORDER.index("Ingest, build, validate") < ORDER.index("Publish media branch") < ORDER.index("Save state") < ORDER.index("Snapshot state")
    assert STEPS["Restore state snapshot"]["run"] == "uv run atlas state restore"
    assert STEPS["Snapshot state"]["run"] == "uv run atlas state snapshot"
    assert STEPS["Check out the media branch"]["run"] == "uv run atlas media checkout --media-dir media-store"
    assert STEPS["Publish media branch"]["run"] == "uv run atlas media publish --media-dir media-store"


def test_the_secret_is_only_exposed_to_the_two_state_steps_and_they_cannot_fail_the_run():
    for name in ("Restore state snapshot", "Snapshot state"):
        assert STEPS[name]["env"] == {"ATLAS_STATE_KEY": "${{ secrets.ATLAS_STATE_KEY }}"} and STEPS[name]["continue-on-error"] is True
    assert sum("ATLAS_STATE_KEY" in str(s) for s in BUILD["steps"]) == 2
    assert STEPS["Publish media branch"]["if"] == "always()" and STEPS["Snapshot state"]["if"].startswith("always()")


def test_the_run_step_writes_into_the_media_checkout_by_default(capsys):
    assert STEPS["Ingest, build, validate"]["run"] == "uv run atlas run"
    for args in (["run", "--help"], ["media", "backfill", "--help"], ["media", "publish", "--help"]):
        try:
            main(args)
        except SystemExit:
            pass
        assert "media-store" in capsys.readouterr().out.replace("\n", "")


# ------------------------------------------------------------------ the audio brief

CI = yaml.safe_load((Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text())
PYPROJECT = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
LOCK = tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text())


def test_the_audio_branch_is_checked_out_before_the_run_and_published_after_it():
    assert ORDER.index("Check out the audio branch") < ORDER.index("Ingest, build, validate") < ORDER.index("Publish audio branch")
    assert ORDER.index("Check out the media branch") < ORDER.index("Check out the audio branch")
    assert STEPS["Check out the audio branch"]["run"] == "uv run atlas audio checkout --audio-dir audio-store"
    assert STEPS["Publish audio branch"]["run"] == "uv run atlas audio publish --audio-dir audio-store"
    assert ORDER.index("Publish media branch") < ORDER.index("Publish audio branch") < ORDER.index("Save state")


def test_audio_steps_can_never_fail_the_run():
    assert STEPS["Publish audio branch"]["if"] == "always()" and STEPS["Publish audio branch"]["continue-on-error"] is True
    assert STEPS["Check out the audio branch"]["continue-on-error"] is True
    assert STEPS["Save speech models"]["continue-on-error"] is True
    for name in ("Check out the audio branch", "Publish audio branch", "Ingest, build, validate"):
        assert isinstance(STEPS[name]["timeout-minutes"], int)
    assert STEPS["Ingest, build, validate"]["timeout-minutes"] < BUILD["timeout-minutes"]            # the job timeout is the backstop, not the plan


def test_the_audio_extra_is_installed_from_the_lock_and_the_models_are_cached():
    core = next(s["run"] for s in BUILD["steps"] if str(s.get("run", "")).startswith("uv sync"))
    assert "--extra audio" not in core and "--frozen" in core and "--extra thumbs" in core       # the feed never depends on the speech stack
    audio = STEPS["Install the audio extra"]
    assert audio["continue-on-error"] is True and isinstance(audio["timeout-minutes"], int)
    assert "--extra audio" in audio["run"] and "--frozen" in audio["run"] and "--extra thumbs" in audio["run"]
    assert BUILD["env"]["HF_HOME"].endswith("/.hf-cache")
    restore, save = STEPS["Restore speech models"], STEPS["Save speech models"]
    assert restore["uses"].startswith("actions/cache/restore@") and save["uses"].startswith("actions/cache/save@")
    assert restore["with"]["path"] == save["with"]["path"] == ".hf-cache" and restore["with"]["key"] == save["with"]["key"]
    assert "pocket-tts" in restore["with"]["key"] and "whisper" in restore["with"]["key"]
    assert ORDER.index("Restore speech models") < ORDER.index("Ingest, build, validate")
    assert "cache-hit" in save["if"]                                                                  # a static key can only be saved once


def test_the_audio_branch_needs_no_new_permissions_or_secrets():
    assert BUILD["permissions"] == {"contents": "write", "actions": "write"} and WF["permissions"] == {"contents": "read"}
    assert "secrets." not in str(STEPS["Publish audio branch"]) and "secrets." not in str(STEPS["Check out the audio branch"])
    assert WF["jobs"]["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}


def test_ci_tests_do_not_install_torch():
    sync = next(s["run"] for s in CI["jobs"]["test"]["steps"] if str(s.get("run", "")).startswith("uv sync"))
    assert "--extra audio" not in sync and "--frozen" in sync


def test_pyproject_routes_torch_to_the_cpu_index_on_linux_and_declares_the_extra():
    extra = " ".join(PYPROJECT["project"]["optional-dependencies"]["audio"])
    for dep in ("pocket-tts", "torch", "faster-whisper", "av"):
        assert dep in extra
    assert any(d.startswith("numpy") for d in PYPROJECT["project"]["dependencies"])                  # imported at start-up: never optional
    uv = PYPROJECT["tool"]["uv"]
    assert uv["sources"]["torch"] == [{"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}]
    assert uv["index"] == [{"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "explicit": True}]
    dev = " ".join(PYPROJECT["dependency-groups"]["dev"])
    assert "av" in dev                                                                                # the audio unit tests run in CI without torch


def test_the_lock_has_cpu_torch_for_linux_and_no_cuda_libraries():
    names = [p["name"] for p in LOCK["package"]]
    assert not [n for n in names if n.startswith("nvidia-") or n in ("triton", "cuda-bindings", "cuda-toolkit")]
    torch = [p for p in LOCK["package"] if p["name"] == "torch"]
    assert any(p["version"].endswith("+cpu") and "download.pytorch.org/whl/cpu" in p["source"]["registry"] for p in torch)
    assert {"pocket-tts", "faster-whisper", "av", "ctranslate2"} <= set(names)
    root = next(p for p in LOCK["package"] if p["name"] == "scienceos-atlas")
    assert "audio" in root["metadata"]["provides-extras"] and "audio" in root["optional-dependencies"]
