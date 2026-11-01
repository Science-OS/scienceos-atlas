"""The `audio` branch: one parentless commit, force-pushed; unchanged trees are not pushed; retention shows up as deletions on the branch."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from atlas.audio import publish as apub
from atlas.audio.storage import AudioStore
from atlas.cli import main
from test_audio_site import make_day
from test_media_publish import rgit, sandbox  # noqa: F401  (sandbox is a fixture)


def branch_files(origin: Path, branch: str = "audio") -> list[str]:
    return rgit(origin, "ls-tree", "-r", "--name-only", branch).splitlines()


def test_first_run_creates_an_orphan_branch_and_pushes_one_parentless_commit(sandbox):  # noqa: F811
    origin, work = sandbox
    dest = work / "audio-store"
    assert apub.checkout(work, dest, "audio") == "orphan"
    assert not (dest / "README.md").exists()
    make_day(dest)
    out = apub.commit_and_push(dest, "audio")
    assert out["committed"] and out["pushed"]
    assert out["message"].startswith("chore(audio): 2026-10-22, 1 day, 12 clips,")
    assert rgit(origin, "rev-list", "--count", "audio") == "1"
    assert rgit(origin, "log", "-1", "--format=%P", "audio") == ""                  # no parent
    files = branch_files(origin)
    assert "days/2026-10-22.json" in files and sum(f.startswith("clips/") and f.endswith(".m4a") for f in files) == 12
    assert rgit(origin, "log", "-1", "--format=%an", "audio") == apub.BOT_NAME
    assert rgit(origin, "show", "audio:days/2026-10-22.json") == (dest / "days/2026-10-22.json").read_text().strip()


def test_a_tree_the_remote_already_has_is_not_pushed_again(sandbox):  # noqa: F811
    origin, work = sandbox
    first = work / "s1"
    apub.checkout(work, first, "audio")
    make_day(first)
    c1 = apub.commit_and_push(first, "audio")["commit"]
    second = work / "s2"
    assert apub.checkout(work, second, "audio") == "existing"                      # what the next CI run sees
    assert (second / "days/2026-10-22.json").is_file()
    out = apub.commit_and_push(second, "audio")
    assert out == {"committed": False, "pushed": False, "reason": "unchanged since the last publish"}
    assert rgit(origin, "rev-parse", "audio") == c1


def test_later_runs_replace_the_branch_history_stays_one_commit_and_old_days_disappear(sandbox):  # noqa: F811
    origin, work = sandbox
    cfg_days = 2
    from atlas.config import AudioCfg
    first = work / "s1"
    apub.checkout(work, first, "audio")
    make_day(first, "2026-10-20", prefix="A", cfg=AudioCfg(retention_days=cfg_days))
    make_day(first, "2026-10-21", prefix="B", cfg=AudioCfg(retention_days=cfg_days))
    c1 = apub.commit_and_push(first, "audio")["commit"]
    second = work / "s2"
    apub.checkout(work, second, "audio")
    make_day(second, "2026-10-22", prefix="C", cfg=AudioCfg(retention_days=cfg_days))      # prunes 2026-10-20 and the clips only it used
    c2 = apub.commit_and_push(second, "audio")["commit"]
    assert c1 != c2 and rgit(origin, "rev-parse", "audio") == c2
    assert rgit(origin, "rev-list", "--count", "audio") == "1" and rgit(origin, "log", "-1", "--format=%P", "audio") == ""
    files = branch_files(origin)
    assert "days/2026-10-20.json" not in files and "days/2026-10-21.json" in files and "days/2026-10-22.json" in files
    ids = {c["id"] for d in AudioStore(second).days() for c in [*json.loads((second / "days" / f"{d}.json").read_text())["papers"].values()]}
    assert all(f"clips/{i}.m4a" in files for i in ids)
    assert sorted(f[6:-4] for f in files if f.startswith("clips/")) == sorted(p.stem for p in (second / "clips").glob("*.m4a"))      # branch == store


def test_no_push_makes_the_commit_object_but_leaves_the_remote_alone(sandbox):  # noqa: F811
    origin, work = sandbox
    dest = work / "s"
    apub.checkout(work, dest, "audio")
    make_day(dest)
    out = apub.commit_and_push(dest, "audio", push=False)
    assert out["committed"] and not out["pushed"]
    assert subprocess.run(["git", "--git-dir", str(origin), "rev-parse", "--verify", "-q", "audio"], capture_output=True).returncode != 0
    assert rgit(work / ".git", "cat-file", "-t", out["commit"]) == "commit"


def test_an_empty_store_publishes_nothing(sandbox):  # noqa: F811
    origin, work = sandbox
    dest = work / "s"
    apub.checkout(work, dest, "audio")
    assert apub.commit_and_push(dest, "audio") == {"committed": False, "pushed": False, "reason": "the audio store is empty"}


def test_cli_checkout_and_publish_round_trip(sandbox, capsys):  # noqa: F811
    origin, work = sandbox
    dest = work / "audio-store"
    assert main(["audio", "checkout", "--repo", str(work), "--audio-dir", str(dest)]) == 0
    assert json.loads(capsys.readouterr().out) == {"checkout": "orphan"}
    make_day(dest)
    capsys.readouterr()                                                              # synthesis logs went to the captured stdout
    assert main(["audio", "publish", "--audio-dir", str(dest), "--no-push"]) == 0
    assert json.loads(capsys.readouterr().out)["pushed"] is False
    assert main(["audio", "publish", "--audio-dir", str(dest)]) == 0
    assert json.loads(capsys.readouterr().out)["pushed"] is True and "days/2026-10-22.json" in branch_files(origin)


def test_cli_publish_refuses_a_plain_directory_and_reports_a_rejected_push_briefly(sandbox, tmp_path, capsys):  # noqa: F811
    origin, work = sandbox
    plain = tmp_path / "plain"
    make_day(plain)
    assert main(["audio", "publish", "--audio-dir", str(plain)]) == 1
    assert "not a checkout of the audio branch" in capsys.readouterr().out
    dest = work / "s"
    apub.checkout(work, dest, "audio")
    make_day(dest)
    capsys.readouterr()
    (origin / "hooks").mkdir(exist_ok=True)
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'rejected: no pushes today' >&2\nexit 1\n")
    hook.chmod(0o755)
    assert main(["audio", "publish", "--audio-dir", str(dest)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["pushed"] is False and "rejected" in out["error"]
