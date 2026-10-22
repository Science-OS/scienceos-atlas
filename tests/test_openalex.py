from __future__ import annotations

import json

import httpx

from atlas import openalex
from atlas.config import OpenAlexCfg
from conftest import data_bytes, data_json, mock_transport


def _client(handler, **cfg):
    return openalex.OpenAlexClient(OpenAlexCfg(min_interval_s=0, **cfg), "ua", transport=mock_transport(handler),
                                   api_key="", sleep=lambda s: None)


def test_id_helpers():
    assert openalex.short_id("https://openalex.org/A5029353080") == "A5029353080"
    assert openalex.short_id(None) is None
    assert openalex.short_orcid("https://orcid.org/0000-0002-3067-9590") == "0000-0002-3067-9590"
    assert openalex.short_ror("https://ror.org/022zdgq74") == "022zdgq74"
    assert openalex.short_ror("not a ror!") is None


def test_compact_work_sample():
    w = openalex.compact_work(data_json("openalex_work_sample.json"))
    assert w["work_id"] == "W7220654000" and w["cited_by_count"] == 0
    assert w["primary_topic"] == "Game Theory and Voting Systems"
    au = w["authorships"][0]
    assert au["id"] == "A5029353080" and au["name"] == "Steve Lawford" and au["orcid"] == "0000-0002-3067-9590"
    assert au["institutions"][0] == {"ror": "022zdgq74", "openalex": "I100296615", "name": "École Nationale de l’Aviation Civile",
                                     "country": "FR", "type": "education"}


def test_compact_author_sample():
    results = data_json("openalex_authors_sample.json")["results"]
    cards = {c["id"]: c for c in map(openalex.compact_author, results)}
    a = cards["A5029353080"]
    assert (a["h_index"], a["i10_index"], a["works_count"], a["cited_by_count"]) == (6, 4, 29, 179)
    assert a["institutions"][0]["ror"] == "022zdgq74"
    assert a["topics"][0] == "Advanced Statistical Methods and Models"  # most frequent topic first
    assert len(a["topics"]) <= 5


def test_get_work_found_missing_and_urls():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        if "2610.99999" in str(req.url):
            return httpx.Response(404, text="<html>nope</html>", headers={"x-ratelimit-remaining": "1000"})
        return httpx.Response(200, content=data_bytes("openalex_work_sample.json"),
                              headers={"x-ratelimit-remaining": "1000", "x-ratelimit-limit": "1000", "x-ratelimit-credits-used": "0"})

    c = _client(handler)
    ok = c.get_work("2610.04688")
    miss = c.get_work("2610.99999")
    assert ok.status == "found" and ok.work["cited_by_count"] == 0
    assert miss.status == "missing"
    assert "works/doi:10.48550/arXiv.2610.04688" in seen[0] and "select=" in seen[0]
    assert "api_key" not in seen[0]
    assert c.credits_used == 0 and c.remaining == 1000 and c.active


def test_429_stops_enrichment_without_raising():
    c = _client(lambda r: httpx.Response(429, headers={"Retry-After": "3600"}))
    assert c.get_work("2610.00001").status == "stopped"
    assert not c.active and "429" in c.stopped_reason
    assert c.get_work("2610.00002").status == "stopped"  # no further requests
    assert c.requests == 1


def test_low_budget_stops_after_the_response():
    def handler(req):
        return httpx.Response(200, content=data_bytes("openalex_work_sample.json"), headers={"x-ratelimit-remaining": "40"})

    c = _client(handler, min_remaining_credits=100)
    assert c.get_work("2610.00001").status == "found"  # this response is still used
    assert not c.active and "budget_low" in c.stopped_reason
    assert c.get_work("2610.00002").status == "stopped"


def test_get_authors_batches_100_and_uses_pipe_filter():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.params["filter"])
        return httpx.Response(200, content=data_bytes("openalex_authors_sample.json"),
                              headers={"x-ratelimit-remaining": "900", "x-ratelimit-credits-used": "1"})

    c = _client(handler)
    ids = [f"A{n}" for n in range(1, 251)]
    out = c.get_authors(ids)
    assert len(calls) == 3 and calls[0].count("|") == 99 and calls[0].startswith("ids.openalex:A1|A2")
    assert "A5029353080" in out
    assert c.credits_used == 3


def test_api_key_is_sent_as_param_when_configured():
    seen = []
    c = openalex.OpenAlexClient(OpenAlexCfg(min_interval_s=0), "ua",
                                transport=mock_transport(lambda r: (seen.append(str(r.url)), httpx.Response(404))[1]),
                                api_key="SECRET", sleep=lambda s: None)
    c.get_work("2610.00001")
    assert "api_key=SECRET" in seen[0]
