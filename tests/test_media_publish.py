from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from atlas import snapshot as snap
from atlas.cli import main
from atlas.media import publish as pub
from atlas.media.catalogue import asset
from atlas.media.store import MediaStore

BASE = "https://science-os.github.io/scienceos-atlas"
T0 = "2026-10-24T06:00:00Z"
needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")


# ---------------------------------------------------------------- git sandbox


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A bare 'origin' plus a working clone, isolated from the developer's git config (signing, hooks, identity)."""
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Test\n\temail = test@example.org\n[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    (work / "README.md").write_text("atlas\n")
    for args in (["add", "-A"], ["commit", "-q", "-m", "init"], ["remote", "add", "origin", str(origin)], ["push", "-q", "origin", "main"]):
        subprocess.run(["git", *args], cwd=work, check=True)
    return origin, work


def rgit(origin: Path, *args: str) -> str:
    return subprocess.run(["git", "--git-dir", str(origin), *args], check=True, capture_output=True, text=True).stdout.strip()


def fill(store: MediaStore, n_authors=1, n_inst=1, n_images=2, tag="a") -> None:
    for i in range(n_authors):
        sha, _ = store.put_bytes(f"photo-{tag}-{i}".encode(), "webp")
        oa = f"A{1 if tag == 'a' else 2}{i}"
        orcid = f"0000-0002-1992-{'268' if tag == 'a' else '269'}{i}"
        store.write_entry("author", [f"orcid-{orcid}", f"openalex-{oa}"], {"orcid": orcid, "openalex": oa},
                          [asset("photo", store.blob_url(sha, "webp"), True, "wikidata_commons", T0, sha256=sha, mime="image/webp", width=9, height=9,
                                 sourceUrl="https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg", license="CC BY 4.0", attribution="x",
                                 attributionUrl="https://commons.wikimedia.org/wiki/File:X.jpg")], T0)
    for i in range(n_inst):
        sha, _ = store.put_bytes(f"logo-{tag}-{i}".encode(), "webp")
        store.write_entry("institution", [f"ror-0190ak57{i}"], {"ror": f"0190ak57{i}"},
                          [asset("logo", store.blob_url(sha, "webp"), True, "wikidata_commons", T0, sha256=sha, mime="image/webp", width=9, height=9,
                                 sourceUrl="https://commons.wikimedia.org/wiki/Special:FilePath/L.svg", license="Public domain",
                                 attribution="y", attributionUrl="https://commons.wikimedia.org/wiki/File:L.svg")], T0)


# ---------------------------------------------------------------- site copy with the retention window


def paper_entry(store: MediaStore, aid: str) -> str:
    sha, _ = store.put_bytes(f"jpeg-{aid}".encode(), "jpg")
    store.write_entry("paper", [f"arxiv-{aid}"], {"arxiv": aid}, [asset(
        "thumbnail", store.blob_url(sha, "jpg"), True, "arxiv_pdf", T0, sha256=sha, mime="image/jpeg", width=600, height=800,
        sourceUrl=f"https://arxiv.org/pdf/{aid}", license="CC BY 4.0", attribution="A et al., arXiv:" + aid,
        attributionUrl=f"https://arxiv.org/abs/{aid}")], T0)
    return sha


def test_site_gets_everything_except_thumbnails_of_papers_outside_the_window(tmp_path):
    store = MediaStore(tmp_path / "ms", BASE)
    fill(store)
    keep, drop = paper_entry(store, "2610.00001"), paper_entry(store, "2509.00002")
    stats = pub.publish_to_site(store, tmp_path / "site", {"2610.00001"}, T0)
    media = tmp_path / "site" / "media"
    assert stats["papers_left_out"] == 1 and stats["entries"] == 4            # 2 author alias files + 1 institution + 1 paper
    assert (media / "b" / keep[:2] / f"{keep}.jpg").is_file() and not (media / "b" / drop[:2] / f"{drop}.jpg").exists()
    assert (media / "c/paper/arxiv-2610.00001.json").is_file() and not (media / "c/paper/arxiv-2509.00002.json").exists()
    assert len(list((media / "b").rglob("*.webp"))) == 2                       # photos and logos are always published
    # the branch keeps everything
    assert store.blob_path(drop, "jpg").is_file() and store.entry_path("paper", "arxiv-2509.00002").is_file()
    # the published manifest describes what is published
    m = json.loads((media / "manifest.json").read_text())
    assert m["counts"] == {"author": 1, "institution": 1, "paper": 1, "hostedFiles": 3, "linkOnly": 0}


def test_without_a_window_everything_is_published_and_files_are_byte_identical(tmp_path):
    store = MediaStore(tmp_path / "ms", BASE)
    fill(store)
    paper_entry(store, "2509.00002")
    pub.publish_to_site(store, tmp_path / "site", None, T0)
    for p in (store.root / "c").rglob("*.json"):
        assert (tmp_path / "site/media" / p.relative_to(store.root)).read_bytes() == p.read_bytes()
    assert len(list((tmp_path / "site/media/b").rglob("*.jpg"))) == 1


def test_republishing_replaces_the_previous_copy(tmp_path):
    store = MediaStore(tmp_path / "ms", BASE)
    fill(store)
    (tmp_path / "site/media/c/author").mkdir(parents=True)
    (tmp_path / "site/media/c/author/stale.json").write_text("{}")
    pub.publish_to_site(store, tmp_path / "site", None, T0)
    assert not (tmp_path / "site/media/c/author/stale.json").exists()


# ---------------------------------------------------------------- the media branch


def test_first_run_creates_an_orphan_branch_and_pushes_it(sandbox, tmp_path):
    origin, work = sandbox
    dest = work / "media-store"
    assert pub.checkout_branch(work, dest, "media") == "orphan"
    assert not (dest / "README.md").exists()                                   # the orphan branch shares nothing with main
    store = MediaStore(dest, BASE)
    fill(store, n_authors=2, n_inst=1)
    store.write_manifest(T0)
    out = pub.commit_and_push(dest, "media")
    assert out["committed"] and out["pushed"] and out["message"] == "chore(media): +2 authors, +1 institution, +3 images"
    assert rgit(origin, "log", "-1", "--format=%s", "media") == out["message"]
    assert rgit(origin, "rev-list", "--count", "media") == "1"
    files = rgit(origin, "ls-tree", "-r", "--name-only", "media").splitlines()
    assert "manifest.json" in files and sum(f.startswith("b/") for f in files) == 3 and "README.md" not in files
    assert rgit(origin, "log", "-1", "--format=%an", "media") == pub.BOT_NAME


def test_later_runs_append_never_rewrite(sandbox):
    origin, work = sandbox
    dest = work / "media-store"
    pub.checkout_branch(work, dest, "media")
    fill(MediaStore(dest, BASE))
    first = pub.commit_and_push(dest, "media")["commit"]
    shutil.rmtree(dest)
    subprocess.run(["git", "worktree", "prune"], cwd=work, check=True)
    assert pub.checkout_branch(work, dest, "media") == "existing"             # second run: the branch is there now
    assert (dest / "b").is_dir() and (dest / "c").is_dir()
    fill(MediaStore(dest, BASE), n_authors=1, n_inst=0, tag="b")
    out = pub.commit_and_push(dest, "media")
    assert out["message"] == "chore(media): +1 author, +1 image"
    assert rgit(origin, "rev-list", "--count", "media") == "2"
    assert rgit(origin, "merge-base", "--is-ancestor", first, "media") == ""   # the first commit is still in the history (fast-forward push)


def test_ci_flow_with_shallow_clones_appends_to_the_branch(sandbox, tmp_path):
    """GitHub's checkout is depth 1: the first run creates the orphan branch, the next run fetches it shallow and fast-forwards."""
    origin, work = sandbox
    url = f"file://{origin}"
    for n, label in enumerate(("one", "two")):
        ci = tmp_path / f"ci{n}"
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(ci)], check=True)
        assert pub.checkout_branch(ci, ci / "media-store", "media") == ("orphan" if n == 0 else "existing")
        MediaStore(ci / "media-store", BASE).put_bytes(label.encode(), "webp")
        assert pub.commit_and_push(ci / "media-store", "media")["pushed"]
    assert rgit(origin, "rev-list", "--count", "media") == "2"
    assert len(rgit(origin, "ls-tree", "-r", "--name-only", "media").splitlines()) == 2


def test_an_unreachable_remote_is_not_mistaken_for_a_first_run(sandbox, tmp_path):
    origin, work = sandbox
    subprocess.run(["git", "remote", "set-url", "origin", str(tmp_path / "gone.git")], cwd=work, check=True)
    with pytest.raises(RuntimeError, match="cannot list"):
        pub.checkout_branch(work, work / "media-store", "media")        # an orphan branch here would later be rejected as non-fast-forward
    assert not (work / "media-store").exists()


def test_a_quiet_run_commits_nothing(sandbox):
    origin, work = sandbox
    dest = work / "media-store"
    pub.checkout_branch(work, dest, "media")
    fill(MediaStore(dest, BASE))
    pub.commit_and_push(dest, "media")
    assert pub.commit_and_push(dest, "media") == {"committed": False, "pushed": False, "reason": "nothing new"}
    assert rgit(origin, "rev-list", "--count", "media") == "1"


def test_checkout_is_a_no_op_when_already_present_and_refuses_a_foreign_directory(sandbox):
    origin, work = sandbox
    dest = work / "media-store"
    pub.checkout_branch(work, dest, "media")
    assert pub.checkout_branch(work, dest, "media") == "present"
    foreign = work / "other"
    foreign.mkdir()
    (foreign / "x.txt").write_text("x")
    with pytest.raises(RuntimeError):
        pub.checkout_branch(work, foreign, "media")


def test_cli_publish_reports_a_rejected_push_briefly(sandbox, capsys):
    origin, work = sandbox
    dest = work / "media-store"
    pub.checkout_branch(work, dest, "media")
    fill(MediaStore(dest, BASE))
    rival = work.parent / "rival"                                              # somebody else pushed to the branch meanwhile
    subprocess.run(["git", "clone", "-q", str(origin), str(rival)], check=True)
    subprocess.run(["git", "checkout", "-q", "--orphan", "media"], cwd=rival, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "rival"], cwd=rival, check=True)
    subprocess.run(["git", "push", "-q", "origin", "media"], cwd=rival, check=True)
    assert main(["media", "publish", "--media-dir", str(dest)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["committed"] and not out["pushed"] and "rejected" in out["error"]


def test_cli_publish_commits_without_pushing_on_request_and_refuses_a_plain_directory(sandbox, tmp_path, capsys):
    origin, work = sandbox
    dest = work / "media-store"
    pub.checkout_branch(work, dest, "media")
    fill(MediaStore(dest, BASE))
    assert main(["media", "publish", "--media-dir", str(dest), "--no-push"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["committed"] and not out["pushed"] and subprocess.run(["git", "--git-dir", str(origin), "rev-parse", "--verify", "-q", "media"]).returncode != 0
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(["media", "publish", "--media-dir", str(plain)]) == 1


# ---------------------------------------------------------------- state snapshot and restore


KEY = "test-key-not-a-secret"


def make_db(path: Path, rows=("alpha", "beta")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE papers(arxiv_id TEXT)")
    db.executemany("INSERT INTO papers VALUES(?)", [(r,) for r in rows])
    db.commit()
    db.close()


def read_rows(path: Path) -> list[str]:
    db = sqlite3.connect(path)
    try:
        return [r[0] for r in db.execute("SELECT arxiv_id FROM papers ORDER BY 1")]
    finally:
        db.close()


@needs_openssl
def test_snapshot_restore_round_trip_with_a_test_key(sandbox, tmp_path, monkeypatch):
    origin, work = sandbox
    monkeypatch.setenv(snap.KEY_ENV, KEY)
    db = tmp_path / "state" / "atlas.sqlite"
    make_db(db)
    out = snap.snapshot(db, work)
    assert out["pushed"] and out["bytes"] > 0
    assert rgit(origin, "ls-tree", "-r", "--name-only", "state").splitlines() == ["atlas.sqlite.gz.enc"]
    # a fresh checkout (CI after a cache miss) brings the file back
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    restored = tmp_path / "restored" / "atlas.sqlite"
    assert snap.restore(restored, clone) == {"restored": True, "bytes": restored.stat().st_size}
    assert read_rows(restored) == ["alpha", "beta"]


@needs_openssl
def test_the_blob_is_what_the_contract_says_aes256_cbc_pbkdf2_over_gzip(sandbox, tmp_path, monkeypatch):
    origin, work = sandbox
    monkeypatch.setenv(snap.KEY_ENV, KEY)
    db = tmp_path / "atlas.sqlite"
    make_db(db, rows=("needle-in-the-state",))
    snap.snapshot(db, work)
    blob = subprocess.run(["git", "--git-dir", str(origin), "cat-file", "blob", "state:atlas.sqlite.gz.enc"], check=True, capture_output=True).stdout
    assert blob.startswith(b"Salted__") and b"SQLite" not in blob and b"needle" not in blob          # encrypted, not just compressed
    (tmp_path / "blob.enc").write_bytes(blob)
    plain = subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-pass", "env:K", "-in", str(tmp_path / "blob.enc")],
                           check=True, capture_output=True, env={"K": KEY, "PATH": "/usr/bin:/bin:/opt/homebrew/bin"}).stdout
    assert gzip.decompress(plain).startswith(b"SQLite format 3")


@needs_openssl
def test_state_branch_is_always_a_single_commit_force_pushed(sandbox, tmp_path, monkeypatch):
    origin, work = sandbox
    monkeypatch.setenv(snap.KEY_ENV, KEY)
    db = tmp_path / "atlas.sqlite"
    make_db(db)
    first = snap.snapshot(db, work)["commit"]
    db.unlink()
    make_db(db, rows=("alpha", "beta", "gamma"))
    second = snap.snapshot(db, work)["commit"]
    assert first != second and rgit(origin, "rev-list", "--count", "state") == "1" and rgit(origin, "rev-parse", "state") == second


@needs_openssl
def test_snapshot_reads_a_consistent_copy_of_a_database_that_is_open_elsewhere(sandbox, tmp_path, monkeypatch):
    origin, work = sandbox
    monkeypatch.setenv(snap.KEY_ENV, KEY)
    db = tmp_path / "atlas.sqlite"
    make_db(db)
    live = sqlite3.connect(db)
    live.execute("INSERT INTO papers VALUES('uncommitted')")                   # an open, uncommitted transaction
    snap.snapshot(db, work)
    live.rollback()
    live.close()
    restored = tmp_path / "r.sqlite"
    snap.restore(restored, work)
    assert read_rows(restored) == ["alpha", "beta"]


def test_without_the_secret_both_directions_skip_with_a_warning_and_never_fail(sandbox, tmp_path, monkeypatch, capsys, caplog):
    origin, work = sandbox
    monkeypatch.delenv(snap.KEY_ENV, raising=False)
    db = tmp_path / "state" / "atlas.sqlite"
    make_db(db)
    assert snap.snapshot(db, work) == {"skipped": "ATLAS_STATE_KEY not set"}
    assert "ATLAS_STATE_KEY" in caplog.text
    assert main(["state", "snapshot", "--state", str(db.parent), "--repo", str(work)]) == 0
    db.unlink()
    assert main(["state", "restore", "--state", str(db.parent), "--repo", str(work)]) == 0
    assert not db.exists()
    assert subprocess.run(["git", "--git-dir", str(origin), "rev-parse", "--verify", "-q", "state"]).returncode != 0      # nothing was pushed


@needs_openssl
def test_restore_leaves_an_existing_database_alone_and_survives_a_missing_branch_or_a_wrong_key(sandbox, tmp_path, monkeypatch):
    origin, work = sandbox
    monkeypatch.setenv(snap.KEY_ENV, KEY)
    target = tmp_path / "t" / "atlas.sqlite"
    assert snap.restore(target, work) == {"restored": False, "reason": "no state branch"}          # first ever run
    src = tmp_path / "src.sqlite"
    make_db(src)
    snap.snapshot(src, work)
    make_db(target, rows=("local",))
    assert snap.restore(target, work)["reason"] == "database already present" and read_rows(target) == ["local"]
    assert snap.restore(target, work, force=True)["restored"] and read_rows(target) == ["alpha", "beta"]
    monkeypatch.setenv(snap.KEY_ENV, "the-wrong-key")
    bad = tmp_path / "bad" / "atlas.sqlite"
    assert snap.restore(bad, work) == {"restored": False, "reason": "restore failed"} and not bad.exists() and not list(bad.parent.glob("*"))


@needs_openssl
def test_restore_from_an_unreachable_remote_warns_and_does_not_raise(sandbox, tmp_path, monkeypatch):
    origin, work = sandbox
    monkeypatch.setenv(snap.KEY_ENV, KEY)
    subprocess.run(["git", "remote", "set-url", "origin", str(tmp_path / "gone.git")], cwd=work, check=True)
    assert snap.restore(tmp_path / "x" / "atlas.sqlite", work) == {"restored": False, "reason": "restore failed"}


def test_snapshot_of_a_missing_database_is_skipped(sandbox, tmp_path, monkeypatch):
    monkeypatch.setenv(snap.KEY_ENV, KEY)
    assert snap.snapshot(tmp_path / "nope.sqlite", sandbox[1]) == {"skipped": "no database"}
