from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from atlas.media.sources import SourceError
from media_fakes import BASE, ETH, PAGE, cascade_world, imageinfo_payload, make_image


def by_source(res):
    return {a["source"]: a for a in res.assets}


# ---------------------------------------------------------------- authors


def test_orcid_chain_hosts_the_commons_photo_and_links_github_and_bluesky(tmp_path):
    cfg, net, http, store, c = cascade_world(tmp_path)
    res = c.resolve_author({"orcid": PAGE["orcid"], "openalex": PAGE["openalex"]}, PAGE["name"], [])
    assert res.found and not res.partial and res.notes == []
    assert res.ids == {"orcid": PAGE["orcid"], "openalex": PAGE["openalex"], "wikidata": "Q7356570", "github": "rdmpage",
                       "bluesky": "rdmpage.bsky.social"}
    assert [(a["source"], a["hosted"]) for a in res.assets] == [("wikidata_commons", True), ("github", False), ("bluesky", False)]
    photo = res.assets[0]
    assert photo["role"] == "photo" and photo["license"] == "CC BY-SA 3.0" and photo["licenseUrl"] == "https://creativecommons.org/licenses/by-sa/3.0"
    assert photo["attribution"] and photo["attributionUrl"] == "https://commons.wikimedia.org/wiki/File:Roderic_D_M_Page.jpg"
    assert photo["sourceUrl"] == "https://commons.wikimedia.org/wiki/Special:FilePath/Roderic_D_M_Page.jpg" and photo["confidence"] == 0.95
    assert photo["mime"] == "image/webp" and max(photo["width"], photo["height"]) == 256 and (photo["width"], photo["height"]) == (183, 256)
    assert photo["url"] == f"{BASE}/media/b/{photo['sha256'][:2]}/{photo['sha256']}.webp"
    blob = store.read_blob(photo["sha256"], "webp")
    assert blob and Image.open(io.BytesIO(blob)).size == (183, 256)           # resize only, never crop
    gh, bs = by_source(res)["github"], by_source(res)["bluesky"]
    assert gh["url"] == "https://github.com/rdmpage.png?size=256" and gh["attributionUrl"] == "https://github.com/rdmpage" and "sha256" not in gh
    assert bs["url"].startswith("https://cdn.bsky.app/") and bs["attribution"] == "Bluesky profile @rdmpage.bsky.social"
    assert [r.url.host for r in net.hits(method="GET")].count("github.com") == 0      # avatar bytes of link-only sources are never fetched


def test_orcid_only_finds_wikidata_and_learns_the_other_ids(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    res = c.resolve_author({"orcid": PAGE["orcid"]}, None)
    assert res.ids["wikidata"] == "Q7356570" and res.ids["openalex"] == "A5068055706"
    assert any("P496" in r.url.params.get("srsearch", "") for r in net.hits("www.wikidata.org"))


def test_openalex_only_uses_the_wikidata_openalex_property(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    res = c.resolve_author({"openalex": PAGE["openalex"]}, "R. Page")
    assert res.found and res.ids["orcid"] == PAGE["orcid"] and res.ids["wikidata"] == "Q7356570"
    assert [r.url.params["srsearch"] for r in net.hits("www.wikidata.org") if "srsearch" in r.url.params] == [f"haswbstatement:P10283={PAGE['openalex']}"]


def test_a_wikidata_key_skips_the_reverse_lookup(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    res = c.resolve_author({"wikidata": "Q7356570"})
    assert res.found and not [r for r in net.hits("www.wikidata.org") if r.url.params.get("action") == "query"]


def test_never_a_name_search(tmp_path):
    """Whatever we know about a person, the only Wikidata search we ever issue is an external-id statement lookup."""
    _, net, _, _, c = cascade_world(tmp_path)
    c.resolve_author({"openalex": "A9999999999"}, "Roderic D. M. Page", ["Rod Page"])
    c.resolve_author({"orcid": "0000-0000-0000-0001"}, "Yann LeCun")
    searches = [r.url.params["srsearch"] for r in net.requests if "srsearch" in r.url.params]
    assert searches and all(s.startswith("haswbstatement:P") and "=" in s for s in searches)
    assert not [r for r in net.requests if "Page" in str(r.url.params.get("srsearch", "")) or "LeCun" in str(r.url.params.get("srsearch", ""))]


def test_no_wikidata_item_means_no_assets_and_no_guessing(tmp_path):
    _, net, _, store, c = cascade_world(tmp_path)
    res = c.resolve_author({"orcid": "0000-0000-0000-0001", "openalex": "A1"}, "Some Researcher")
    assert not res.found and res.notes == ["no_wikidata"] and not net.hits("commons.wikimedia.org")


def test_reverse_hit_with_a_different_name_is_rejected(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    res = c.resolve_author({"orcid": PAGE["orcid"]}, "Totally Different Person", ["T. Different"])
    assert not res.found and res.notes == ["name_mismatch"] and "wikidata" not in res.ids
    assert not net.hits("commons.wikimedia.org") and c.stats["name_mismatch"] == 1


def test_name_check_accepts_the_usual_variants_via_aliases(tmp_path):
    _, _, _, _, c = cascade_world(tmp_path)
    assert c.resolve_author({"orcid": PAGE["orcid"]}, "R. D. M. Page", ["Rod Page", "Roderic Page"]).found


def test_person_without_p18_still_gets_link_only_avatars(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.entities["Q7356570"] = {**net.entities["Q7356570"], "claims": {k: v for k, v in net.entities["Q7356570"]["claims"].items() if k != "P18"}}
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert [a["source"] for a in res.assets] == ["github", "bluesky"] and "no_p18" in res.notes


def test_non_free_commons_licence_is_never_hosted_but_other_sources_remain(tmp_path):
    _, net, _, store, c = cascade_world(tmp_path)
    net.imageinfo["Roderic D M Page.jpg"] = imageinfo_payload("Roderic D M Page.jpg", "CC BY-NC 4.0")
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert [a["source"] for a in res.assets] == ["github", "bluesky"] and "license_nonfree" in res.notes
    assert not net.hits("upload.wikimedia.org") and not (store.root / "b").exists()


def test_missing_licence_information_is_never_hosted(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.imageinfo["Roderic D M Page.jpg"] = imageinfo_payload("Roderic D M Page.jpg", None)
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert not any(a["hosted"] for a in res.assets) and "license_missing" in res.notes and not net.hits("upload.wikimedia.org")


def test_commons_file_that_does_not_exist_counts_as_missing_licence(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.imageinfo.pop("Roderic D M Page.jpg")
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert "license_missing" in res.notes and not any(a["hosted"] for a in res.assets)


def test_free_licence_without_an_attributable_author_is_not_hosted(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.imageinfo["Roderic D M Page.jpg"] = imageinfo_payload("Roderic D M Page.jpg", "CC BY 4.0", artist=None, credit=None)
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert not any(a["hosted"] for a in res.assets) and "no_attribution" in res.notes


def test_public_domain_photo_without_artist_is_hosted_with_a_generic_credit(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.imageinfo["Roderic D M Page.jpg"] = imageinfo_payload("Roderic D M Page.jpg", "Public domain", artist=None)
    photo = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"]).assets[0]
    assert photo["hosted"] and photo["attribution"] == "Wikimedia Commons" and "licenseUrl" not in photo


def test_dead_github_account_gets_no_avatar(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.github.clear()
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert [a["source"] for a in res.assets] == ["wikidata_commons", "bluesky"] and "github" not in res.ids


def test_a_failing_optional_source_keeps_the_rest_and_marks_the_result_partial(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.down["public.api.bsky.app"] = 503
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert res.partial and [a["source"] for a in res.assets] == ["wikidata_commons", "github"] and c.stats["error_bluesky"] == 1


def test_a_failing_core_source_raises_so_nothing_is_cached_as_negative(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.down["commons.wikimedia.org"] = 503
    with pytest.raises(SourceError):
        c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])


def test_undecodable_image_bytes_are_skipped(tmp_path):
    _, net, _, store, c = cascade_world(tmp_path)
    net.images["Roderic_D_M_Page.jpg"] = (b"<html>rate limited</html>", "text/html")
    res = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert not any(a["hosted"] for a in res.assets) and "image_undecodable" in res.notes and not (store.root / "b").exists()


def test_resolving_again_reuses_the_blob_and_a_304_reuses_the_asset(tmp_path):
    _, net, _, store, c = cascade_world(tmp_path)
    first = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    store.write_entry("author", ["orcid-" + PAGE["orcid"]], first.ids, first.assets, "2026-10-24T06:00:00Z")
    assert c.stats["blobs_created"] == 1 and c.etags                                   # ETag of the render remembered
    again = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"])
    assert again.assets[0]["sha256"] == first.assets[0]["sha256"] and c.stats["not_modified"] == 1 and c.stats["blobs_created"] == 1
    sent = [r.headers.get("if-none-match") for r in net.hits("upload.wikimedia.org")]
    assert sent == [None, 'W/"v1"']


def test_changed_remote_image_without_prior_etag_gives_the_same_hash_when_bytes_equal(tmp_path):
    _, net, _, store, c = cascade_world(tmp_path)
    a = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"]).assets[0]
    b = c.resolve_author({"orcid": PAGE["orcid"]}, PAGE["name"]).assets[0]           # no catalogue entry written: no 304 path
    assert a["sha256"] == b["sha256"] and c.stats["blobs_created"] == 1 and c.stats["blobs_reused"] == 1
    assert len(list((store.root / "b").rglob("*.webp"))) == 1


# ---------------------------------------------------------------- institutions


def test_ror_to_wikidata_to_p154_hosts_the_logo_via_the_commons_png_render(tmp_path):
    cfg, net, _, store, c = cascade_world(tmp_path)
    res = c.resolve_institution({"ror": ETH["ror"]})
    assert res.ids == {"ror": ETH["ror"], "wikidata": ETH["qid"]} and res.notes == []
    logo = res.assets[0]
    assert len(res.assets) == 1 and logo["role"] == "logo" and logo["hosted"] and logo["source"] == "wikidata_commons"
    assert logo["license"] == "Public domain" and logo["attributionUrl"] == "https://commons.wikimedia.org/wiki/File:ETH_Z%C3%BCrich_Logo_black.svg"
    assert logo["confidence"] == 0.9 and logo["width"] == 256 and logo["height"] == 43       # 330x56 render shrunk to 256 px, ratio kept
    asked = [str(r.url) for r in net.hits("commons.wikimedia.org") if "Special:FilePath" in r.url.path]
    assert asked == ["https://commons.wikimedia.org/wiki/Special:FilePath/ETH_Z%C3%BCrich_Logo_black.svg?width=256"]
    img = Image.open(io.BytesIO(store.read_blob(logo["sha256"], "webp")))
    assert img.mode == "RGBA" and img.format == "WEBP"
    assert not net.hits("ethz.ch") and not net.hits("t3.gstatic.com")                         # a hosted logo ends the cascade


def test_wikidata_key_only_institution_learns_its_ror_from_p6782(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    res = c.resolve_institution({"wikidata": ETH["qid"]})
    assert res.ids["ror"] == ETH["ror"] and res.found


def test_non_free_logo_falls_back_to_the_homepage_icon_link_only(tmp_path):
    _, net, _, store, c = cascade_world(tmp_path)
    net.imageinfo["ETH Zürich Logo black.svg"] = imageinfo_payload("ETH Zürich Logo black.svg", "CC BY-ND 4.0")
    res = c.resolve_institution({"ror": ETH["ror"]})
    assert "license_nonfree" in res.notes and len(res.assets) == 1
    icon = res.assets[0]
    assert icon["source"] == "homepage_icon" and not icon["hosted"] and icon["url"].endswith("192-xxxhpdi.png") and icon["confidence"] == 0.7
    assert not (store.root / "b").exists()


def test_homepage_without_a_big_icon_falls_back_to_faviconv2(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.imageinfo.clear()
    net.pages["https://ethz.ch/"] = '<link rel="icon" href="/small.png" sizes="32x32">'
    net.favicon = make_image(128, 128)
    res = c.resolve_institution({"ror": ETH["ror"]})
    assert [a["source"] for a in res.assets] == ["favicon"] and res.assets[0]["confidence"] == 0.5 and "size=128" in res.assets[0]["url"]


def test_tiny_favicon_is_rejected_so_the_institution_has_no_logo(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    net.imageinfo.clear()
    net.pages["https://ethz.ch/"] = "<html></html>"
    net.favicon = make_image(16, 16)
    assert not c.resolve_institution({"ror": ETH["ror"]}).found


def test_unknown_ror_id_resolves_to_nothing(tmp_path):
    _, net, _, _, c = cascade_world(tmp_path)
    res = c.resolve_institution({"ror": "000000000"})
    assert not res.found and res.notes == ["no_wikidata"]


def test_institution_logo_resolution_is_json_serialisable_and_contract_shaped(tmp_path):
    _, _, _, store, c = cascade_world(tmp_path)
    res = c.resolve_institution({"ror": ETH["ror"]})
    store.write_entry("institution", ["ror-05a28rw58", "wikidata-Q11942"], res.ids, res.assets, "2026-10-24T06:00:00Z")
    doc = json.loads((store.root / "c" / "institution" / "ror-05a28rw58.json").read_text())
    assert list(doc) == ["version", "kind", "key", "ids", "updatedAt", "assets"]
    assert list(doc["assets"][0])[:5] == ["role", "url", "hosted", "sha256", "mime"]
