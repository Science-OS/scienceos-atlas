"""Durability of everything scraped: state/atlas.sqlite, gzipped and encrypted (openssl aes-256-cbc, pbkdf2) with the Actions secret
ATLAS_STATE_KEY, force-pushed as a SINGLE commit to the `state` branch. On an Actions-cache miss the workflow restores from it.
A missing key, an absent branch or a bad file only ever produce a warning: this must never fail the run."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .media.publish import BOT_EMAIL, BOT_NAME, git

log = logging.getLogger("atlas.snapshot")

KEY_ENV = "ATLAS_STATE_KEY"
FILE_NAME = "atlas.sqlite.gz.enc"


def _openssl(decrypt: bool, src: Path, dst: Path, key_env: str) -> None:
    cmd = ["openssl", "enc", *(["-d"] if decrypt else []), "-aes-256-cbc", "-pbkdf2", "-salt", "-pass", f"env:{key_env}", "-in", str(src), "-out", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, env=os.environ.copy())


def encrypt_db(db: Path, key_env: str = KEY_ENV) -> bytes:
    """A consistent copy of the SQLite file (backup API), gzipped, then encrypted."""
    with tempfile.TemporaryDirectory() as td:
        copy, gz, enc = Path(td) / "atlas.sqlite", Path(td) / "atlas.sqlite.gz", Path(td) / FILE_NAME
        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        with open(copy, "rb") as fin, gzip.open(gz, "wb", compresslevel=9) as fout:
            shutil.copyfileobj(fin, fout)
        _openssl(False, gz, enc, key_env)
        return enc.read_bytes()


def decrypt_to_db(blob: bytes, dest: Path, key_env: str = KEY_ENV) -> None:
    with tempfile.TemporaryDirectory() as td:
        enc, gz = Path(td) / FILE_NAME, Path(td) / "atlas.sqlite.gz"
        enc.write_bytes(blob)
        _openssl(True, enc, gz, key_env)
        tmp = dest.with_name(dest.name + ".restoring")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with gzip.open(gz, "rb") as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            check = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
            try:
                check.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            finally:
                check.close()
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)


def snapshot(db: Path, repo: Path, *, branch: str = "state", remote: str = "origin", push: bool = True, key_env: str = KEY_ENV) -> dict[str, Any]:
    """Encrypt the DB and replace branch `state` with one commit holding just that file (history is dropped on purpose)."""
    if not os.environ.get(key_env):
        log.warning("%s is not set: skipping the state snapshot", key_env)
        return {"skipped": f"{key_env} not set"}
    if not Path(db).exists():
        log.warning("%s does not exist: skipping the state snapshot", db)
        return {"skipped": "no database"}
    blob = encrypt_db(Path(db), key_env)
    sha_blob = git(repo, "hash-object", "-w", "--stdin", input=blob).stdout.decode().strip()
    tree = git(repo, "mktree", input=f"100644 blob {sha_blob}\t{FILE_NAME}\n".encode()).stdout.decode().strip()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = {"GIT_AUTHOR_NAME": BOT_NAME, "GIT_AUTHOR_EMAIL": BOT_EMAIL, "GIT_COMMITTER_NAME": BOT_NAME, "GIT_COMMITTER_EMAIL": BOT_EMAIL}
    commit = git(repo, "commit-tree", tree, "-m", f"chore(state): encrypted snapshot {stamp}", env=env).stdout.decode().strip()
    if push:
        git(repo, "push", "--force", remote, f"{commit}:refs/heads/{branch}")
    return {"snapshot": True, "bytes": len(blob), "commit": commit, "pushed": push}


def restore(db: Path, repo: Path, *, branch: str = "state", remote: str = "origin", force: bool = False, key_env: str = KEY_ENV) -> dict[str, Any]:
    """Bring the DB back from branch `state` when it is missing (a cache miss). An existing DB is left alone unless force."""
    db = Path(db)
    if db.exists() and not force:
        return {"restored": False, "reason": "database already present"}
    if not os.environ.get(key_env):
        log.warning("%s is not set: cannot restore the state snapshot", key_env)
        return {"restored": False, "reason": f"{key_env} not set"}
    try:
        probe = git(repo, "ls-remote", "--exit-code", "--heads", remote, branch, check=False)
        if probe.returncode == 2:
            log.warning("no %s branch yet: nothing to restore", branch)
            return {"restored": False, "reason": f"no {branch} branch"}
        if probe.returncode != 0:
            raise OSError(probe.stderr.decode(errors="replace").strip()[:200])
        git(repo, "fetch", "--depth", "1", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}")
        blob = git(repo, "cat-file", "blob", f"refs/remotes/{remote}/{branch}:{FILE_NAME}").stdout
        decrypt_to_db(blob, db, key_env)
    except (subprocess.CalledProcessError, OSError, sqlite3.Error) as exc:
        detail = (exc.stderr.decode(errors="replace").strip()[:200] if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc))
        log.warning("state restore failed (%s); the run will cold-start", detail)
        return {"restored": False, "reason": "restore failed"}
    return {"restored": True, "bytes": db.stat().st_size}
