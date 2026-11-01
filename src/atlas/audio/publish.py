"""The `audio` branch: ONE commit with no parent, force-pushed on every run (like the `state` branch; unlike the append-only `media` branch).
The branch holds exactly what the site serves - clips/<id>.m4a and days/<date>.json for the last `retention_days` - so its size is bounded and
old audio disappears from the repository's history instead of piling up. Clips that are unchanged since the last run are the same git objects
on the remote already, so a push only uploads what is new."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..media.publish import BOT_EMAIL, BOT_NAME, checkout_branch, git
from .storage import AudioStore

log = logging.getLogger("atlas.audio.publish")

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def checkout(repo: Path, dest: Path, branch: str = "audio", remote: str = "origin") -> str:
    """The branch as a git worktree at `dest` ('present', 'existing' or 'orphan' on the first run)."""
    return checkout_branch(Path(repo), Path(dest), branch, remote)


def summarize(dest: Path) -> str:
    store = AudioStore(dest)
    days = store.days()
    clips = sum(1 for _ in store.clips_dir.glob("*.m4a")) if store.clips_dir.is_dir() else 0
    span = f"{days[0]}..{days[-1]}" if len(days) > 1 else (days[0] if days else "no days")
    return f"{span}, {len(days)} day{'s' if len(days) != 1 else ''}, {clips} clips, {store.total_bytes() / 1e6:.1f} MB"


def commit_and_push(dest: Path, branch: str = "audio", remote: str = "origin", push: bool = True) -> dict[str, Any]:
    """Snapshot the worktree as a parentless commit and force-push it to `branch`. Nothing is pushed when the tree equals the remote's."""
    dest = Path(dest)
    git(dest, "add", "-A")
    tree = git(dest, "write-tree").stdout.decode().strip()
    if tree == EMPTY_TREE:
        return {"committed": False, "pushed": False, "reason": "the audio store is empty"}
    probe = git(dest, "rev-parse", "--verify", "-q", f"refs/remotes/{remote}/{branch}^{{tree}}", check=False)
    if probe.returncode == 0 and probe.stdout.decode().strip() == tree:
        return {"committed": False, "pushed": False, "reason": "unchanged since the last publish"}
    message = f"chore(audio): {summarize(dest)}"
    env = {"GIT_AUTHOR_NAME": BOT_NAME, "GIT_AUTHOR_EMAIL": BOT_EMAIL, "GIT_COMMITTER_NAME": BOT_NAME, "GIT_COMMITTER_EMAIL": BOT_EMAIL}
    commit = git(dest, "commit-tree", tree, "-m", message, env=env).stdout.decode().strip()
    if push:
        git(dest, "push", "--force", remote, f"{commit}:refs/heads/{branch}")
    return {"committed": True, "pushed": push, "commit": commit, "tree": tree, "message": message}
