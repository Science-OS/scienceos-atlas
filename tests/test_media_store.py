from __future__ import annotations

import json
import os

import pytest

from atlas.media import catalogue as cat
from atlas.media.catalogue import asset
from atlas.media.store import MediaStore

BASE = "https://science-os.github.io/scienceos-atlas"
T0, T1 = "2026-10-24T06:00:00Z", "2026-10-25T06:00:00Z"
SHA_A, SHA_B, SHA_C, SHA_D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def hosted(role="photo", sha=SHA_A, source="wikidata_commons", at=T0, **kw):
    return asset(role, f"{BASE}/media/b/{sha[:2]}/{sha}.webp", True, source, at, sha256=sha, mime="image/webp", width=200, height=256,
                 sourceUrl="https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg", license="CC BY-SA 2.0",
                 attribution="Someone", attributionUrl="https://commons.wikimedia.org/wiki/File:X.jpg", confidence=0.95, **kw)


def linked(source="github", role="photo", url="https://github.com/ylecun.png?size=256", at=T0, **kw):
    return asset(role, url, False, source, at, attribution="GitHub profile @ylecun", confidence=0.9, **kw)


@pytest.fixture
def store(tmp_path) -> MediaStore:
    return MediaStore(tmp_path / "media-store", BASE)


# ---------------------------------------------------------------- bytes


def test_same_bytes_same_path_and_existing_files_are_never_rewritten(store):
    sha, created = store.put_bytes(b"first", "webp")
    path = store.blob_path(sha, "webp")
    assert created and path == store.root / "b" / sha[:2] / f"{sha}.webp"
    os.utime(path, (1_000_000, 1_000_000))
    sha2, created2 = store.put_bytes(b"first", "webp")
    assert sha2 == sha and not created2
    assert path.stat().st_mtime == 1_000_000 and path.read_bytes() == b"first"


def test_an_existing_file_wins_even_if_its_content_was_altered(store):
    sha, _ = store.put_bytes(b"orig", "webp")
    store.blob_path(sha, "webp").write_bytes(b"tampered")
    store.put_bytes(b"orig", "webp")
    assert store.blob_path(sha, "webp").read_bytes() == b"tampered"      # append-only: never rewritten, validate.py catches it


def test_blob_url_and_extensions(store):
    sha, _ = store.put_bytes(b"jpegbytes", "jpg")
    assert store.blob_url(sha, "jpg") == f"{BASE}/media/b/{sha[:2]}/{sha}.jpg"
    with pytest.raises(ValueError):
        store.put_bytes(b"x", "svg")
    assert not list(store.root.rglob(".tmp-*"))


# ---------------------------------------------------------------- catalogue files


def test_entry_files_one_per_key_identical_apart_from_key(store):
    keys = ["orcid-0000-0002-1992-2684", "wikidata-Q3571662", "openalex-A5071252540"]
    ids = {"openalex": "A5071252540", "orcid": "0000-0002-1992-2684", "wikidata": "Q3571662"}
    out = store.write_entry("author", keys, ids, [hosted()], T0)
    assert out == dict.fromkeys(keys, "created")
    docs = [json.loads((store.root / "c" / "author" / f"{k}.json").read_text()) for k in keys]
    assert [d["key"] for d in docs] == keys
    assert all({**d, "key": ""} == {**docs[0], "key": ""} for d in docs)
    assert docs[0]["ids"] == ids and list(docs[0]["ids"]) == ["orcid", "wikidata", "openalex"]    # contract order, not input order
    assert docs[0]["version"] == 1 and docs[0]["kind"] == "author" and docs[0]["updatedAt"] == T0


def test_unchanged_content_does_not_rewrite_the_file_even_with_new_timestamps(store):
    key = "orcid-0000-0002-1992-2684"
    store.write_entry("author", [key], {"orcid": "0000-0002-1992-2684"}, [hosted(), linked()], T0)
    path = store.root / "c" / "author" / f"{key}.json"
    os.utime(path, (1_000_000, 1_000_000))
    out = store.write_entry("author", [key], {"orcid": "0000-0002-1992-2684"}, [hosted(at=T1), linked(at=T1)], T1)
    assert out == {key: "unchanged"} and path.stat().st_mtime == 1_000_000
    assert json.loads(path.read_text())["updatedAt"] == T0


def test_changed_content_updates_the_file_and_its_timestamp(store):
    key = "orcid-0000-0002-1992-2684"
    store.write_entry("author", [key], {"orcid": "0000-0002-1992-2684"}, [linked()], T0)
    out = store.write_entry("author", [key], {"orcid": "0000-0002-1992-2684"}, [hosted(), linked()], T1)
    assert out == {key: "updated"}
    doc = store.read_entry("author", key)
    assert doc["updatedAt"] == T1 and [a["hosted"] for a in doc["assets"]] == [True, False]


def test_a_new_alias_joins_an_existing_entity_and_all_files_agree(store):
    store.write_entry("author", ["openalex-A1"], {"openalex": "A1"}, [hosted()], T0)
    out = store.write_entry("author", ["openalex-A1", "wikidata-Q9"], {"openalex": "A1", "wikidata": "Q9"}, [hosted()], T1)
    assert out == {"openalex-A1": "updated", "wikidata-Q9": "created"}
    a, b = store.read_entry("author", "openalex-A1"), store.read_entry("author", "wikidata-Q9")
    assert a["assets"] == b["assets"] and a["ids"] == b["ids"] == {"wikidata": "Q9", "openalex": "A1"}


def test_invalid_keys_never_touch_the_filesystem(store):
    with pytest.raises(ValueError):
        store.entry_path("author", "../../outside")
    assert store.write_entry("author", ["../../outside", "nonsense"], {}, [hosted()], T0) == {}
    assert not (store.root / "c").exists() and not (store.root.parent / "outside.json").exists()


def test_lookup_returns_first_existing_key(store):
    store.write_entry("author", ["openalex-A1"], {"openalex": "A1"}, [hosted()], T0)
    assert store.lookup("author", ["orcid-0000-0002-1992-2684", "openalex-A1"])["key"] == "openalex-A1"
    assert store.lookup("author", ["wikidata-Q404"]) is None


# ---------------------------------------------------------------- ordering and merging


def test_assets_are_ordered_best_first_hosted_before_link_only_within_a_role():
    shuffled = [linked("bluesky", url="https://cdn.bsky.app/a"), linked("github"), hosted("logo", SHA_B),
                hosted("photo", SHA_A), linked("favicon", "logo", "https://t3.gstatic.com/f"), hosted("thumbnail", SHA_C, "arxiv_pdf"),
                linked("homepage_icon", "logo", "https://x.org/i.png")]
    got = [(a["role"], a["hosted"], a["source"]) for a in cat.order_assets(shuffled)]
    assert got == [("photo", True, "wikidata_commons"), ("photo", False, "github"), ("photo", False, "bluesky"),
                   ("logo", True, "wikidata_commons"), ("logo", False, "homepage_icon"), ("logo", False, "favicon"),
                   ("thumbnail", True, "arxiv_pdf")]


def test_hosted_beats_link_only_even_with_lower_confidence():
    low = hosted(); low["confidence"] = 0.1
    high = linked(); high["confidence"] = 1.0
    assert cat.order_assets([high, low])[0]["hosted"] is True


def test_merge_never_drops_a_hosted_file_and_replaces_stale_links():
    old = [hosted(sha=SHA_A), linked(url="https://github.com/old.png?size=256")]
    new = [hosted(sha=SHA_B), linked(url="https://github.com/new.png?size=256")]
    merged = cat.merge_assets(old, new)
    assert [a.get("sha256") or a["url"] for a in merged] == [SHA_B, SHA_A, "https://github.com/new.png?size=256"]


def test_merge_keeps_roles_the_new_resolution_found_nothing_for():
    old = [hosted("photo", SHA_A), linked(), hosted("logo", SHA_B)]
    merged = cat.merge_assets(old, [hosted("logo", SHA_C)])
    assert {a["role"] for a in merged} == {"photo", "logo"} and len([a for a in merged if a["role"] == "logo"]) == 2
    assert any(a["source"] == "github" for a in merged)


def test_merge_caps_old_hosted_files_per_role():
    old = [hosted(sha=c * 64) for c in "abc"]
    merged = cat.merge_assets(old, [hosted(sha="d" * 64)])
    assert len(merged) == cat.MAX_HOSTED_PER_ROLE and merged[0]["sha256"] == "d" * 64


def test_first_asset_and_photo_mapping():
    e = {"assets": [hosted("logo", SHA_B), hosted("photo", SHA_A)]}
    assert cat.first_asset(e, "photo")["sha256"] == SHA_A and cat.first_asset(e, "thumbnail") is None and cat.first_asset(None, "photo") is None
    photo = cat.as_photo(hosted())
    assert set(photo) == {"url", "source", "license", "attribution", "attributionUrl", "confidence", "fetchedAt"}


# ---------------------------------------------------------------- manifest


def test_manifest_counts_entities_once_not_alias_files(store):
    store.write_entry("author", ["orcid-0000-0002-1992-2684", "wikidata-Q3571662"], {"orcid": "0000-0002-1992-2684", "wikidata": "Q3571662"},
                      [hosted(sha=SHA_A), linked()], T0)
    store.write_entry("author", ["openalex-A2"], {"openalex": "A2"}, [linked("bluesky", url="https://cdn.bsky.app/x")], T0)
    store.write_entry("institution", ["ror-0190ak572"], {"ror": "0190ak572"}, [hosted("logo", SHA_B)], T0)
    store.write_entry("paper", ["arxiv-2610.30108"], {"arxiv": "2610.30108"}, [hosted("thumbnail", SHA_C, "arxiv_pdf")], T0)
    store.put_bytes(b"a", "webp"), store.put_bytes(b"bb", "jpg")
    m = store.compute_manifest(T0)
    assert m == {"version": 1, "generatedAt": T0, "counts": {"author": 2, "institution": 1, "paper": 1, "hostedFiles": 2, "linkOnly": 2},
                 "hostedBytes": 3}


def test_manifest_is_only_rewritten_when_something_changed(store):
    store.write_entry("author", ["openalex-A1"], {"openalex": "A1"}, [linked()], T0)
    first = store.write_manifest(T0)
    path = store.root / "manifest.json"
    os.utime(path, (1_000_000, 1_000_000))
    assert store.write_manifest(T1) == first and path.stat().st_mtime == 1_000_000       # quiet run: nothing to commit
    store.write_entry("author", ["openalex-A2"], {"openalex": "A2"}, [linked()], T1)
    assert store.write_manifest(T1)["generatedAt"] == T1 and json.loads(path.read_text())["counts"]["author"] == 2
