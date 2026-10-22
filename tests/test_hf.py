from __future__ import annotations

from datetime import date

import httpx

from atlas import hf
from atlas.http import PoliteClient
from conftest import data_json, mock_transport


def test_parse_daily_sample():
    sigs = {s.arxiv_id: s for s in hf.parse_daily(data_json("hf_daily_sample.json"))}
    assert set(sigs) == {"2610.26203", "2610.26855", "2610.27999", "2610.27628"}
    s = sigs["2610.26203"]
    assert s.upvotes == 227 and s.github_stars == 12
    assert s.github_repo == "https://github.com/fyzhang1/FocusGUI"
    assert s.ai_summary and s.ai_summary.startswith("FocusGUI")
    assert s.daily_date == "2026-10-21"
    assert sigs["2610.27999"].github_repo is None and sigs["2610.27999"].github_stars is None


def test_parse_daily_is_tolerant():
    junk = [None, 3, {"paper": None}, {"paper": {"id": "not-an-id"}}, {"paper": {"id": "2610.00001", "upvotes": "x"}}]
    out = hf.parse_daily(junk)
    assert [s.arxiv_id for s in out] == ["2610.00001"] and out[0].upvotes is None
    assert hf.parse_daily({"error": "nope"}) == []


def test_fetch_recent_merges_days_keeping_highest_upvotes(clock):
    def handler(req: httpx.Request) -> httpx.Response:
        day = req.url.params["date"]
        up = {"2026-10-22": 5, "2026-10-21": 9}.get(day)
        body = [{"paper": {"id": "2610.00001", "upvotes": up}}] if up is not None else []
        return httpx.Response(200, json=body)

    c = PoliteClient("ua", 0.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    out = hf.fetch_recent(c, "https://hf.test/api/daily_papers", date(2026, 10, 22), days=3, limit=100, max_pages=2)
    assert len(out) == 1 and out[0].upvotes == 9
    assert c.requests_made == 3


def test_fetch_day_pages_until_short_page(clock):
    pages = []

    def handler(req: httpx.Request) -> httpx.Response:
        p = int(req.url.params.get("p", 0))
        pages.append(p)
        n = 2 if p == 0 else 1
        return httpx.Response(200, json=[{"paper": {"id": f"2610.{p}{i:04d}"}} for i in range(n)])

    c = PoliteClient("ua", 0.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    out = hf.fetch_day(c, "https://hf.test/x", date(2026, 10, 22), limit=2, max_pages=5)
    assert pages == [0, 1] and len(out) == 3


def test_fetch_recent_survives_a_failing_day(clock):
    def handler(req):
        if req.url.params["date"] == "2026-10-21":
            return httpx.Response(500)
        return httpx.Response(200, json=[{"paper": {"id": "2610.00002", "upvotes": 1}}])

    c = PoliteClient("ua", 0.0, max_retries=0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    out = hf.fetch_recent(c, "https://hf.test/x", date(2026, 10, 22), days=2, limit=100, max_pages=1)
    assert [s.arxiv_id for s in out] == ["2610.00002"]
