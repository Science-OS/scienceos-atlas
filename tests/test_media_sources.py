from __future__ import annotations

import pytest

from atlas.media import sources as S
from atlas.media import web as W
from media_fakes import FIX, FakeNet, fx, make_cfg, make_http, make_image


@pytest.fixture
def world():
    net = FakeNet()
    return net, make_http(make_cfg(), net)


# ---------------------------------------------------------------- the licence gate (recorded Commons responses + variants)


def test_recorded_commons_response_parses_licence_artist_and_file_page():
    info = S.parse_imageinfo(fx("commons_info_lecun.json"))
    assert info["license"] == "CC BY-SA 2.0" and info["licenseUrl"] == "https://creativecommons.org/licenses/by-sa/2.0"
    assert info["artist"].startswith("Ecole polytechnique")
    assert info["descriptionUrl"] == "https://commons.wikimedia.org/wiki/File:Yann_LeCun_-_2025_(cropped).jpg"
    assert S.license_verdict(info) == "free" and S.credit_for(info)


@pytest.mark.parametrize("short", ["CC BY-SA 4.0", "CC BY 2.0", "CC BY 4.0", "CC BY-SA 3.0 de", "CC-BY-SA-3.0", "CC0", "CC0 1.0 Universal",
                                   "Public domain", "PD-textlogo", "PD-old-70", "PD-USGov", "pd"])
def test_free_licences_may_be_hosted(short):
    assert S.license_verdict({"license": short}) == "free"


@pytest.mark.parametrize("short", ["CC BY-NC 4.0", "CC BY-ND 2.0", "CC BY-NC-SA 3.0", "CC BY-NC-ND 4.0", "GFDL", "GFDL 1.2", "Copyrighted free use",
                                   "OGL 3.0", "Attribution", "Fair use", "All rights reserved", "PDM"])
def test_non_free_or_unlisted_licences_are_never_hosted(short):
    assert S.license_verdict({"license": short}) == "nonfree"


def test_non_free_flag_wins_over_a_free_looking_licence():
    assert S.license_verdict({"license": "CC BY 4.0", "nonFree": True}) == "nonfree"


@pytest.mark.parametrize("info", [None, {}, {"license": None}, {"license": "   "}])
def test_missing_licence_information_is_its_own_verdict(info):
    assert S.license_verdict(info) == "missing"


def test_unknown_commons_file_has_no_info(world):
    net, http = world
    assert S.license_info(http, "No such file.jpg") is None


def test_credit_for_artist_credit_and_public_domain_fallback():
    assert S.credit_for({"license": "CC BY 4.0", "artist": "Ada", "credit": "Flickr"}) == "Ada"
    assert S.credit_for({"license": "CC BY 4.0", "artist": None, "credit": "Flickr"}) == "Flickr"
    assert S.credit_for({"license": "Public domain"}) == "Wikimedia Commons"
    assert S.credit_for({"license": "CC0"}) == "Wikimedia Commons"
    assert S.credit_for({"license": "CC BY-SA 4.0"}) is None            # attribution is the licence: no artist, no hosting


# ---------------------------------------------------------------- Wikidata


def test_person_and_org_facts_from_recorded_entities():
    p = S.person_facts(fx("wikidata_entity_page.json")["entities"]["Q7356570"])
    assert p["image"] == "Roderic D M Page.jpg" and p["github"] == "rdmpage" and p["bluesky"] == "rdmpage.bsky.social"
    assert p["orcid"] == "0000-0002-7101-9767" and p["openalex"] == ["A5068055706"] and p["label"] == "Roderic D. M. Page"
    o = S.org_facts(fx("wikidata_entity_eth.json")["entities"]["Q11942"])
    assert o["logos"][0] == "ETH Zürich Logo black.svg" and o["website"] and o["ror"] == "05a28rw58"


def test_claim_values_prefer_preferred_rank_and_put_retired_values_last():
    def claim(v, rank="normal", end=None):
        c = {"rank": rank, "mainsnak": {"snaktype": "value", "datavalue": {"type": "string", "value": v}}}
        if end:
            c["qualifiers"] = {"P582": [{}]}
        return c
    ent = {"claims": {"P154": [claim("old.svg", end=True), claim("normal.svg"), claim("preferred.svg", "preferred"), claim("gone.svg", "deprecated")]}}
    assert S.claim_values(ent, "P154") == ["preferred.svg", "normal.svg", "old.svg"]


def test_reverse_lookup_by_external_id(world):
    net, http = world
    assert S.find_qid(http, "P496", "0000-0002-7101-9767") == "Q7356570"
    assert S.find_qid(http, "P10283", "A5068055706") == "Q7356570"
    assert S.find_qid(http, "P496", "0000-0000-0000-0001") is None
    assert all("haswbstatement:" in r.url.params["srsearch"] for r in net.hits("www.wikidata.org"))


def test_ambiguous_reverse_lookup_attaches_nothing(world):
    net, http = world
    two = {"query": {"search": [{"title": "Q1"}, {"title": "Q2"}]}}
    net.search[("P496", "0000-0002-7101-9767")] = two
    assert S.find_qid(http, "P496", "0000-0002-7101-9767") is None


def test_wikidata_outage_is_a_source_error_not_an_empty_answer(world):
    net, http = world
    net.down["www.wikidata.org"] = 503
    with pytest.raises(S.SourceError):
        S.find_qid(http, "P496", "0000-0002-7101-9767")


# ---------------------------------------------------------------- Commons URL, ROR


def test_filepath_url_quotes_and_asks_for_a_render_width():
    assert S.filepath_url("ETH Zürich Logo black.svg", 256) == "https://commons.wikimedia.org/wiki/Special:FilePath/ETH_Z%C3%BCrich_Logo_black.svg?width=256"
    assert S.filepath_url("A (b).jpg") == "https://commons.wikimedia.org/wiki/Special:FilePath/A_%28b%29.jpg"


def test_recorded_ror_record(world):
    net, http = world
    org = S.ror_record(http, "05a28rw58")
    assert org["name"] == "ETH Zurich" and org["wikidata"] == "Q11942" and org["website"].startswith("https://")
    assert S.ror_record(http, "000000000") is None


# ---------------------------------------------------------------- GitHub and Bluesky (link-only)


def test_github_avatar_uses_head_without_following_the_redirect(world):
    net, http = world
    assert S.github_avatar(http, "rdmpage") == "https://github.com/rdmpage.png?size=256"
    req = net.hits("github.com")[0]
    assert req.method == "HEAD" and len(net.hits("github.com")) == 1            # redirect not followed, bytes never fetched
    assert S.github_avatar(http, "no-such-user") is None


@pytest.mark.parametrize("login", ["", "-bad", "a/b", "x" * 40, "with space", "../etc"])
def test_github_logins_are_validated_before_any_request(world, login):
    net, http = world
    assert S.github_avatar(http, login) is None and not net.hits("github.com")


def test_bluesky_avatar_from_recorded_profile(world):
    net, http = world
    got = S.bluesky_avatar(http, "@RDMPage.bsky.social")
    assert got["handle"] == "rdmpage.bsky.social" and got["avatar"].startswith("https://cdn.bsky.app/")
    assert S.bluesky_avatar(http, "nobody.bsky.social") is None               # Bluesky answers 400 for unknown actors
    assert S.bluesky_avatar(http, "not a handle") is None


def test_bluesky_avatar_must_live_on_the_bluesky_cdn(world):
    net, http = world
    net.bluesky["rdmpage.bsky.social"] = {**net.bluesky["rdmpage.bsky.social"], "avatar": "https://evil.example/a.png"}
    assert S.bluesky_avatar(http, "rdmpage.bsky.social") is None


# ---------------------------------------------------------------- homepages and favicons (SSRF guard, size checks)


@pytest.mark.parametrize("url,ok", [
    ("https://ethz.ch/", True), ("http://example.org/x", True),
    ("http://localhost/", False), ("http://127.0.0.1/", False), ("http://10.0.0.5/", False), ("http://192.168.1.1/", False),
    ("http://169.254.169.254/latest/meta-data", False), ("http://[::1]/", False), ("http://intranet/", False),
    ("http://printer.local/", False), ("http://svc.internal/", False), ("ftp://example.org/", False), ("file:///etc/passwd", False),
    ("javascript:alert(1)", False), ("", False),
])
def test_is_public_http_url(url, ok):
    assert W.is_public_http_url(url) is ok


def test_icons_parsed_from_recorded_homepage_largest_first():
    html = (FIX / "homepage_eth_head.html").read_text(encoding="utf-8")
    icons = W.parse_icons("https://ethz.ch/en.html", html)
    assert icons == [{"href": "https://ethz.ch/etc/designs/ethz/img/icons/ETH-APP-Icons-Theme-white/192-xxxhpdi.png", "size": 192, "rel": "icon"}]


def test_icon_sizes_come_from_sizes_attribute_with_apple_touch_default():
    page = ('<link rel="icon" href="/a.ico" sizes="16x16 32x32"><link rel="apple-touch-icon" href="/t.png">'
            '<link rel="shortcut icon" href="https://cdn.x/i.svg" sizes="any"><link rel="stylesheet" href="/s.css"><link rel="icon">')
    got = {i["href"]: i["size"] for i in W.parse_icons("https://x.org/", page)}
    assert got == {"https://x.org/a.ico": 32, "https://x.org/t.png": 180, "https://cdn.x/i.svg": 512}


def test_homepage_icon_is_the_real_size_not_just_the_declared_one(world):
    net, http = world
    assert W.homepage_icon(http, "https://ethz.ch/", 120) == (
        "https://ethz.ch/etc/designs/ethz/img/icons/ETH-APP-Icons-Theme-white/192-xxxhpdi.png", 192)
    net.icons[next(iter(net.icons))] = make_image(32, 32)                       # page claims 192x192, file is 32x32
    assert W.homepage_icon(http, "https://ethz.ch/", 120) is None


def test_homepage_icon_svg_counts_as_scalable(world):
    net, http = world
    net.pages["https://u.example.org/"] = '<link rel="icon" href="/logo.svg" sizes="any">'
    net.icons["https://u.example.org/logo.svg"] = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'
    assert W.homepage_icon(http, "https://u.example.org/", 120) == ("https://u.example.org/logo.svg", None)


def test_homepage_redirect_to_a_private_host_is_not_followed(world):
    net, http = world
    net.redirects["https://u.example.org/"] = "http://169.254.169.254/latest/meta-data"
    assert W.homepage_icon(http, "https://u.example.org/", 120) is None
    assert [r.url.host for r in net.requests] == ["u.example.org"]            # the private hop was refused before any request


def test_homepage_redirects_to_public_hosts_are_followed(world):
    net, http = world
    net.redirects["http://ethz.ch/"] = "https://ethz.ch/"
    assert W.homepage_icon(http, "http://ethz.ch/", 120)[1] == 192


def test_private_homepage_is_never_requested(world):
    net, http = world
    assert W.homepage_icon(http, "http://10.1.2.3/", 120) is None and W.homepage_icon(http, "http://localhost:8080/", 120) is None
    assert not net.requests


def test_favicon_accepted_only_when_big_enough(world):
    net, http = world
    assert W.favicon(http, "https://ethz.ch/") is None                          # 404 from the service
    net.favicon = make_image(16, 16)
    assert W.favicon(http, "https://ethz.ch/") is None                          # the 16 px globe placeholder
    net.favicon = make_image(128, 128)
    url, px = W.favicon(http, "https://ethz.ch/")
    assert px == 128 and url.startswith("https://t3.gstatic.com/faviconV2?") and "size=128" in url and "url=https%3A%2F%2Fethz.ch%2F" in url
