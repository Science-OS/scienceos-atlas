"""The daily workflow stays thin: every step with logic is a CLI subcommand, and the order of the steps is what the contract needs."""

from __future__ import annotations

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
