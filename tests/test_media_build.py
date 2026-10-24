from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from atlas.build import SiteBuilder, build_author_json
from atlas.config import load_config
from atlas.media.catalogue import asset
from atlas.media.store import MediaStore
from atlas.store import Store
from atlas.validate import validate_site
from media_fakes import make_image
from test_build_validate import NC_ID, NEW_ID, NOW, oa_work_for, seed

BASE = "https://science-os.github.io/scienceos-atlas"
T0 = "2026-10-22T04:00:00Z"
ORCID, ROR = "0000-0002-3067-9590", "022zdgq74"          # first author of NEW_ID in the test world, and its institution
PHOTO_KEYS = [f"orcid-{ORCID}", "openalex-A900"]


def photo_asset(store: MediaStore, role="photo", seed_bytes=b"p", **kw) -> dict:
    sha, _ = store.put_bytes(make_image(40, 40, "PNG") + seed_bytes, "webp")           # bytes need not be a real WebP for the builder
    return asset(role, store.blob_url(sha, "webp"), True, "wikidata_commons", T0, sha256=sha, mime="image/webp", width=40, height=40,
                 sourceUrl="https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg", license="CC BY-SA 2.0", licenseUrl="https://creativecommons.org/licenses/by-sa/2.0",
                 attribution="Ecole polytechnique", attributionUrl="https://commons.wikimedia.org/wiki/File:X.jpg", confidence=0.95, **kw)


def thumb_asset(store: MediaStore, aid: str) -> dict:
    sha, _ = store.put_bytes(b"\xff\xd8\xff jpeg " + aid.encode(), "jpg")
    return asset("thumbnail", store.blob_url(sha, "jpg"), True, "arxiv_pdf", T0, sha256=sha, mime="image/jpeg", width=600, height=800,
                 sourceUrl=f"https://arxiv.org/pdf/{aid}", license="CC BY 4.0", licenseUrl="https://creativecommons.org/licenses/by/4.0",
                 attribution=f"A et al., arXiv:{aid}", attributionUrl=f"https://arxiv.org/abs/{aid}", confidence=1.0)


@pytest.fixture
def world(tmp_path):
    state = Store(":memory:")
    recs = seed(state)
    target = next(r for r in recs if r.arxiv_id == NEW_ID)
    state.set_oa_work(NEW_ID, "found", oa_work_for(target), "2026-10-22T04:05:00Z")
    state.upsert_authors({f"A{900 + i}": {"id": f"A{900 + i}", "name": a["name"], "alternatives": [], "orcid": ORCID if i == 0 else None, "works_count": 10,
                                          "cited_by_count": 100, "h_index": 5, "i10_index": 2,
                                          "institutions": [{"ror": ROR, "openalex": "I1", "name": "ENAC", "country": "FR", "type": "education"}],
                                          "topics": [], "updated_date": "2026-10-20"} for i, a in enumerate(target.authors)}, "2026-10-22T04:06:00Z")
    thumbs = tmp_path / "state" / "thumbs"
    thumbs.mkdir(parents=True)
    media = MediaStore(tmp_path / "media-store", BASE)
    site = tmp_path / "site"
    return state, media, thumbs, site


def build(world, now=NOW, cfg=None):
    state, media, thumbs, site = world
    b = SiteBuilder(state, cfg or load_config(), site, thumbs, now, media)
    stats = b.build()
    b.write_status({"duration_s": 1})
    return stats


def load(site: Path, rel: str):
    return json.loads((site / rel).read_text())


def first_author_file(site: Path) -> dict:
    return load(site, "authors/A900.json")


# ---------------------------------------------------------------- the feed is filled from the catalogue


def test_author_file_gets_photo_and_logos_from_the_catalogue(world):
    state, media, thumbs, site = world
    photo = photo_asset(media)
    logo = photo_asset(media, "logo", b"l")
    media.write_entry("author", PHOTO_KEYS, {"orcid": ORCID, "openalex": "A900"}, [photo], T0)
    media.write_entry("institution", [f"ror-{ROR}"], {"ror": ROR}, [logo], T0)
    stats = build(world)
    doc = first_author_file(site)
    a = doc["author"]
    assert a["photo"] == {"url": photo["url"], "source": "wikidata_commons", "license": "CC BY-SA 2.0", "attribution": "Ecole polytechnique",
                          "attributionUrl": "https://commons.wikimedia.org/wiki/File:X.jpg", "confidence": 0.95, "fetchedAt": T0}
    assert a["enrichment"]["pending"] == ["experience", "coauthors"]                 # "photo" goes away once a photo is known
    assert a["affiliations"][0]["logoUrl"] == logo["url"]
    assert doc["institutions"][0]["logo"]["url"] == logo["url"] and doc["institutions"][0]["logo"]["license"] == "CC BY-SA 2.0"
    assert stats.authors_with_photo >= 1 and stats.institutions_with_logo >= 1
    assert validate_site(site).ok


def test_authors_without_a_catalogue_entry_stay_as_before(world):
    build(world)
    a = load(world[3], "authors/A901.json")["author"]
    assert "photo" not in a and a["enrichment"]["pending"] == ["photo", "experience", "coauthors"]
    assert "logoUrl" not in a["affiliations"][0] and "logo" not in load(world[3], "authors/A901.json")["institutions"][0]


def test_a_link_only_photo_fills_the_field_when_nothing_is_hosted(world):
    state, media, thumbs, site = world
    gh = asset("photo", "https://github.com/someone.png?size=256", False, "github", T0, attribution="GitHub profile @someone",
               attributionUrl="https://github.com/someone", confidence=0.9)
    media.write_entry("author", ["openalex-A900"], {"openalex": "A900"}, [gh], T0)
    build(world)
    assert first_author_file(site)["author"]["photo"]["url"] == "https://github.com/someone.png?size=256"


def test_orcid_key_is_preferred_over_openalex(world):
    state, media, thumbs, site = world
    media.write_entry("author", ["openalex-A900"], {"openalex": "A900"}, [photo_asset(media, seed_bytes=b"oa")], T0)
    best = photo_asset(media, seed_bytes=b"orcid")
    media.write_entry("author", [f"orcid-{ORCID}"], {"orcid": ORCID}, [best], T0)
    build(world)
    assert first_author_file(world[3])["author"]["photo"]["url"] == best["url"]


def test_build_author_json_without_media_is_unchanged():
    author, insts = build_author_json({"id": "A1", "name": "N", "alternatives": [], "orcid": None, "institutions": [
        {"ror": "022zdgq74", "openalex": "I1", "name": "ENAC", "country": "FR", "type": "education"}], "topics": []}, "2026-10-22T00:00:00Z")
    assert "photo" not in author and author["enrichment"]["pending"] == ["photo", "experience", "coauthors"] and "logo" not in insts[0]


# ---------------------------------------------------------------- thumbnails


def test_thumbnail_link_points_at_the_hosted_file_and_the_legacy_path_is_still_published(world):
    state, media, thumbs, site = world
    state.set_thumb(NEW_ID, "ok", 10, "2026-10-22T04:07:00Z")
    (thumbs / f"{NEW_ID}.jpg").write_bytes(b"\xff\xd8\xff legacy")
    t = thumb_asset(media, NEW_ID)
    media.write_entry("paper", [f"arxiv-{NEW_ID}"], {"arxiv": NEW_ID}, [t], T0)
    stats = build(world)
    links = load(site, f"papers/{NEW_ID}.json")["paper"]["links"]
    assert {"kind": "other", "label": "thumbnail", "url": t["url"]} in links
    assert (site / "thumbs" / f"{NEW_ID}.jpg").read_bytes() == b"\xff\xd8\xff legacy"        # old app builds build this path themselves
    assert (site / "media" / "b" / t["sha256"][:2] / f"{t['sha256']}.jpg").is_file()
    assert stats.papers_with_hosted_thumb == 1 and stats.thumbs == 1
    assert validate_site(site).ok


def test_without_a_catalogue_entry_the_link_is_the_legacy_one(world):
    state, media, thumbs, site = world
    state.set_thumb(NEW_ID, "ok", 10, "2026-10-22T04:07:00Z")
    (thumbs / f"{NEW_ID}.jpg").write_bytes(b"\xff\xd8\xff legacy")
    build(world)
    links = load(site, f"papers/{NEW_ID}.json")["paper"]["links"]
    assert {"kind": "other", "label": "thumbnail", "url": f"{BASE}/thumbs/{NEW_ID}.jpg"} in links


def test_a_hosted_thumbnail_without_a_local_file_still_links_and_publishes(world):
    state, media, thumbs, site = world
    t = thumb_asset(media, NEW_ID)                                   # the Actions cache lost state/thumbs, the store still has it
    media.write_entry("paper", [f"arxiv-{NEW_ID}"], {"arxiv": NEW_ID}, [t], T0)
    build(world)
    assert any(l.get("label") == "thumbnail" and l["url"] == t["url"] for l in load(site, f"papers/{NEW_ID}.json")["paper"]["links"])
    assert not (site / "thumbs").exists() and validate_site(site).ok


def test_non_redistributable_papers_never_link_a_thumbnail_even_if_the_catalogue_has_one(world):
    state, media, thumbs, site = world
    media.write_entry("paper", [f"arxiv-{NC_ID}"], {"arxiv": NC_ID}, [thumb_asset(media, NC_ID)], T0)
    build(world)
    assert not any(l.get("label") == "thumbnail" for l in load(site, f"papers/{NC_ID}.json")["paper"]["links"])


def test_papers_outside_the_retention_window_keep_their_bytes_on_the_branch_but_not_on_the_site(world):
    state, media, thumbs, site = world
    old = "2509.00002"
    t_old, t_new = thumb_asset(media, old), thumb_asset(media, NEW_ID)
    media.write_entry("paper", [f"arxiv-{old}"], {"arxiv": old}, [t_old], T0)
    media.write_entry("paper", [f"arxiv-{NEW_ID}"], {"arxiv": NEW_ID}, [t_new], T0)
    media.write_entry("author", PHOTO_KEYS, {"orcid": ORCID, "openalex": "A900"}, [photo_asset(media)], T0)
    build(world)
    pub = site / "media"
    assert (pub / "b" / t_new["sha256"][:2] / f"{t_new['sha256']}.jpg").is_file()
    assert not (pub / "b" / t_old["sha256"][:2] / f"{t_old['sha256']}.jpg").exists() and not (pub / "c/paper/arxiv-2509.00002.json").exists()
    assert media.blob_path(t_old["sha256"], "jpg").is_file()                         # still in the store
    assert len(list((pub / "b").rglob("*.webp"))) == 1                               # the author photo
    assert validate_site(site).ok


def test_site_without_a_media_store_has_no_media_directory(world):
    state, media, thumbs, site = world
    SiteBuilder(state, load_config(), site, thumbs, NOW).build()
    assert not (site / "media").exists()


# ---------------------------------------------------------------- the validator on a media site


@pytest.fixture
def good_site(world):
    state, media, thumbs, site = world
    media.write_entry("author", PHOTO_KEYS, {"orcid": ORCID, "openalex": "A900"}, [photo_asset(media)], T0)
    media.write_entry("paper", [f"arxiv-{NEW_ID}"], {"arxiv": NEW_ID}, [thumb_asset(media, NEW_ID)], T0)
    build(world)
    assert validate_site(site).ok
    return site


def entry_file(site: Path, kind="author") -> Path:
    return sorted((site / "media" / "c" / kind).glob("*.json"))[0]


def mutate(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def errors(site: Path, **kw) -> str:
    return " | ".join(validate_site(site, **kw).errors)


def test_validator_counts_the_media_entries(good_site):
    rep = validate_site(good_site)
    assert rep.checked["media_entries"] == 3 and rep.ok


def test_media_size_counts_toward_the_site_budget_and_has_its_own_limit(good_site):
    media_bytes = sum(p.stat().st_size for p in (good_site / "media").rglob("*") if p.is_file())
    assert media_bytes > 0
    assert "media budget" in errors(good_site, media_max_mb=0)
    assert "over the" in errors(good_site, max_site_mb=0)
    assert validate_site(good_site, max_site_mb=450, media_max_mb=250).ok
    (good_site / "media" / "ballast.bin").write_bytes(b"0" * (2 * 1024 * 1024))
    assert "media/ is" in errors(good_site, media_max_mb=1) and validate_site(good_site, media_max_mb=250).ok


def test_catalogue_entries_are_validated_against_the_schema(good_site):
    f = entry_file(good_site)
    mutate(f, lambda d: d["assets"][0].pop("sha256"))
    assert "sha256" in errors(good_site)
    mutate(f, lambda d: d["assets"][0].update(sha256="a" * 64, role="avatar"))
    assert "role" in errors(good_site) or "avatar" in errors(good_site)
    mutate(f, lambda d: d.update(surprise=1))
    assert "surprise" in errors(good_site) or "Additional properties" in errors(good_site)


def test_link_only_assets_must_not_claim_bytes(good_site):
    f = entry_file(good_site)
    mutate(f, lambda d: d["assets"].append({"role": "photo", "url": "https://github.com/x.png", "hosted": False, "source": "github",
                                            "fetchedAt": T0, "sha256": "a" * 64}))
    assert errors(good_site)


def test_hosted_assets_must_carry_licence_and_credit(good_site):
    f = entry_file(good_site)
    mutate(f, lambda d: d["assets"][0].pop("license"))
    assert "license" in errors(good_site)


def test_hosted_file_must_exist_and_match_its_hash(good_site):
    f = entry_file(good_site)
    sha = json.loads(f.read_text())["assets"][0]["sha256"]
    blob = good_site / "media" / "b" / sha[:2] / f"{sha}.webp"
    blob.write_bytes(blob.read_bytes() + b"x")
    assert "do not match the sha256" in errors(good_site)
    blob.unlink()
    assert "is missing" in errors(good_site)


def test_hosted_url_must_match_hash_and_mime(good_site):
    f = entry_file(good_site)
    mutate(f, lambda d: d["assets"][0].update(url=d["assets"][0]["url"].replace("/media/b/", "/media/x/")))
    assert "does not match its sha256" in errors(good_site)


def test_kind_key_and_path_must_agree(good_site):
    f = entry_file(good_site)
    mutate(f, lambda d: d.update(key="openalex-A12345"))
    assert "do not match the file path" in errors(good_site)
    f2 = good_site / "media" / "c" / "author" / "not-a-key.json"
    f2.write_text(f.read_text())
    assert "not-a-key.json" in errors(good_site)


def test_manifest_must_exist_and_tell_the_truth(good_site):
    m = good_site / "media" / "manifest.json"
    mutate(m, lambda d: d["counts"].update(author=99))
    assert "counts/hostedBytes do not match" in errors(good_site)
    m.unlink()
    assert "manifest.json is missing" in errors(good_site)


def test_published_json_may_not_point_at_unpublished_media(good_site):
    author = good_site / "authors" / "A900.json"
    mutate(author, lambda d: d["author"]["photo"].update(url=f"{BASE}/media/b/ab/{'ab' * 32}.webp"))
    assert "is referenced by published JSON but missing" in errors(good_site)
    mutate(author, lambda d: d["author"].pop("photo"))
    shard = good_site / "feeds" / "2026-10-21.json"
    shard_doc = json.loads(shard.read_text())
    assert any(l.get("label") == "thumbnail" and "/media/b/" in l["url"] for it in shard_doc["items"] for l in it["paper"]["links"])
    for p in (good_site / "media" / "b").rglob("*.jpg"):
        p.unlink()
    assert "referenced by published JSON but missing" in errors(good_site)
