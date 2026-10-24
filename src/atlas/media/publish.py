"""Getting the store out: copy it into site/media for the Pages deploy (thumbnails only for the retention window), and
commit/push it to the append-only `media` branch (the branch keeps everything; the site publishes the window)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .catalogue import first_asset
from .keys import entity_keys
from .store import MediaStore

log = logging.getLogger("atlas.media.publish")

BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


def publish_to_site(media: MediaStore, site_dir: Path, keep_papers: set[str] | None, now: str) -> dict[str, Any]:
    """Copy the store to <site>/media. With `keep_papers` (arxiv ids in the retention window) the catalogue entry and the JPEG
    of every other paper are left out of the site, so the published catalogue never points at bytes that are not published.
    Returns counts; writes the published manifest."""
    dest = site_dir / "media"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    stats = {"entries": 0, "papers_left_out": 0, "blobs": 0, "bytes": 0}
    jpg_keep: set[str] = set()
    for kind, key, doc in media.iter_entries():
        if kind == "paper" and keep_papers is not None and (doc.get("ids") or {}).get("arxiv") not in keep_papers:
            stats["papers_left_out"] += 1
            continue
        if kind == "paper":
            a = first_asset(doc, "thumbnail")
            if a and a.get("hosted") and a.get("sha256"):
                jpg_keep.add(a["sha256"])
        (dest / "c" / kind).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(media.entry_path(kind, key), dest / "c" / kind / f"{key}.json")
        stats["entries"] += 1
    blobs = media.root / "b"
    for f in sorted(blobs.rglob("*")) if blobs.is_dir() else []:
        if not f.is_file() or f.name.startswith(".tmp-"):
            continue
        if f.suffix == ".jpg" and f.stem not in jpg_keep:
            continue
        target = dest / "b" / f.relative_to(blobs)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
        stats["blobs"] += 1
        stats["bytes"] += f.stat().st_size
    MediaStore(dest, media.base_url).write_manifest(now)
    return stats


# ------------------------------------------------------------------ git


def git(cwd: Path, *args: str, check: bool = True, env: dict[str, str] | None = None,
        input: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    full_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})}
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, env=full_env, input=input)


def checkout_branch(repo: Path, dest: Path, branch: str = "media", remote: str = "origin") -> str:
    """Make `dest` a git worktree of `branch` (a shared repo config keeps the push credentials working).
    Creates an orphan branch on the first run. Returns 'present', 'existing' or 'orphan'."""
    dest = Path(dest)
    if (dest / ".git").exists():
        return "present"
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"{dest} exists and is not a checkout of {branch}")
    probe = git(repo, "ls-remote", "--exit-code", "--heads", remote, branch, check=False)
    if probe.returncode not in (0, 2):   # 2 = "no such branch"; anything else (network, auth) must not be mistaken for a first run
        raise RuntimeError(f"cannot list {remote}: {probe.stderr.decode(errors='replace').strip()[:200]}")
    if probe.returncode == 0:
        git(repo, "fetch", "--depth", "1", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}")
        git(repo, "worktree", "add", "--detach", str(dest), f"refs/remotes/{remote}/{branch}")
        return "existing"
    git(repo, "worktree", "add", "--detach", str(dest), "HEAD")
    git(dest, "checkout", "--orphan", branch)
    git(dest, "rm", "-rf", "--quiet", "--ignore-unmatch", ".")
    return "orphan"


def _summary(dest: Path) -> str:
    """'+3 authors, +2 institutions, +5 papers, +9 images' from what is staged (alias files of one entity count once)."""
    names = git(dest, "diff", "--cached", "--name-only", "--diff-filter=A").stdout.decode().splitlines()
    entities: dict[str, set[str]] = {"author": set(), "institution": set(), "paper": set()}
    images = 0
    for n in names:
        parts = n.split("/")
        if parts[0] == "b" and parts[-1].rsplit(".", 1)[-1] in ("webp", "jpg"):
            images += 1
        elif parts[0] == "c" and len(parts) == 3 and parts[1] in entities:
            try:
                ids = json.loads((dest / n).read_text(encoding="utf-8")).get("ids", {})
            except (OSError, ValueError):
                ids = {}
            key = parts[2].removesuffix(".json")
            entities[parts[1]].add((entity_keys(parts[1], ids) or [key])[0] if parts[1] != "paper" else key)
    bits = [f"+{len(v)} {k if len(v) == 1 else k + 's'}" for k, v in (("author", entities["author"]), ("institution", entities["institution"]),
                                                                       ("paper", entities["paper"])) if v]
    bits += [f"+{images} image{'s' if images != 1 else ''}"] if images else []
    return ", ".join(bits)


def commit_and_push(dest: Path, branch: str = "media", remote: str = "origin", push: bool = True) -> dict[str, Any]:
    """Commit new files as `chore(media): <counts>` and push HEAD to the branch (fast-forward only: history is append-only)."""
    git(dest, "add", "-A")
    if not git(dest, "diff", "--cached", "--name-only").stdout.strip():
        return {"committed": False, "pushed": False, "reason": "nothing new"}
    summary = _summary(dest) or "update catalogue"
    ident = ["-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}"]
    git(dest, *ident, "commit", "-q", "-m", f"chore(media): {summary}")
    sha = git(dest, "rev-parse", "HEAD").stdout.decode().strip()
    pushed = False
    if push:
        git(dest, "push", remote, f"HEAD:refs/heads/{branch}")
        pushed = True
    return {"committed": True, "pushed": pushed, "commit": sha, "message": f"chore(media): {summary}"}
