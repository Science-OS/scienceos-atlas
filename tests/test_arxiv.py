from __future__ import annotations

from datetime import date

import httpx
import pytest

from atlas import arxiv
from atlas.http import PoliteClient
from conftest import atom_feed, data_bytes, mock_transport


def test_parse_oai_sample_fields():
    records, token = arxiv.parse_list_records(data_bytes("oai_econ_sample.xml"))
    assert token is None
    assert len(records) == 6
    by_id = {r.arxiv_id: r for r in records}
    r = by_id["2610.27576"]
    assert r.datestamp == "2026-10-21"
    assert r.license == "http://creativecommons.org/licenses/by/4.0/"
    assert r.title.startswith("Who Gains When the Umpire Is a Machine")
    assert "\n" not in r.abstract and "  " not in r.abstract  # whitespace normalised
    assert r.authors and all(a["name"] for a in r.authors)
    assert any(c.startswith("econ.") for c in r.categories)
    assert "econ:econ:GN" in r.set_specs or any(s.startswith("econ") for s in r.set_specs)
    # keyname/forenames are joined "forenames keyname"
    old = by_id["2109.14850"]
    assert [a["name"] for a in old.authors][:2] == ["Bruce Changlong Xu", "Andrew Yang"]


def test_parse_resumption_token():
    records, token = arxiv.parse_list_records(data_bytes("oai_cs_page_token.xml"))
    assert len(records) == 3
    assert token and token.startswith("verb%3DListRecords")


def test_parse_no_records_match_and_errors():
    assert arxiv.parse_list_records(data_bytes("oai_no_records.xml")) == ([], None)
    with pytest.raises(arxiv.OaiError) as ei:
        arxiv.parse_list_records(data_bytes("oai_error_badargument.xml"))
    assert ei.value.code == "badArgument"


def test_harvest_follows_resumption_token_and_throttles(clock):
    seen: list[dict] = []
    page1, page2 = data_bytes("oai_cs_page_token.xml"), data_bytes("oai_econ_sample.xml")

    def handler(req: httpx.Request) -> httpx.Response:
        params = dict(req.url.params)
        seen.append(params)
        return httpx.Response(200, content=page1 if "resumptionToken" not in params else page2)

    client = PoliteClient("ua", 3.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    recs = list(arxiv.harvest(client, "https://oaipmh.arxiv.org/oai", "cs", "2026-10-22"))
    assert len(recs) == 3 + 6
    assert seen[0] == {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": "cs", "from": "2026-10-22"}
    # follow-up request carries ONLY verb + resumptionToken (OAI-PMH rule)
    assert set(seen[1]) == {"verb", "resumptionToken"}
    assert clock.sleeps == [3.0]  # one throttle wait between the two pages


def test_harvest_stop_callback_halts_paging(clock):
    client = PoliteClient("ua", 0.0, transport=mock_transport(lambda r: httpx.Response(200, content=data_bytes("oai_cs_page_token.xml"))),
                          sleep=clock.sleep, clock=clock.now)
    recs = list(arxiv.harvest(client, "https://x/oai", "cs", "2026-10-22", stop=lambda: True))
    assert len(recs) == 3 and client.requests_made == 1


def test_is_recent_id():
    assert arxiv.is_recent_id("2610.27576", "2026-10-21")
    assert arxiv.is_recent_id("2609.10000", "2026-10-02")      # previous month still counts
    assert not arxiv.is_recent_id("2109.14850", "2026-10-21")  # old paper revised today
    assert not arxiv.is_recent_id("cs/0701001", "2026-10-21")  # old-style id
    assert arxiv.is_recent_id("2601.00001", "2026-01-02")
    assert arxiv.is_recent_id("2512.00001", "2026-01-02")      # December -> January boundary


def test_matches_categories():
    pats = ["cs.*", "stat.ML", "econ.*"]
    assert arxiv.matches_categories(["math.OC", "cs.LG"], pats)
    assert arxiv.matches_categories(["stat.ML"], pats)
    assert not arxiv.matches_categories(["stat.ME"], pats)
    assert not arxiv.matches_categories(["hep-th"], pats)


def test_parse_api_feed_and_is_new_paper():
    entries = arxiv.parse_api_feed(data_bytes("arxiv_api_sample.xml"))
    assert set(entries) == {"2610.27576", "2610.04688", "2109.14850"}
    new, rev, old = entries["2610.27576"], entries["2610.04688"], entries["2109.14850"]
    assert new.version == 1 and new.published == "2026-10-20T03:42:43Z" and new.primary_category == "econ.GN"
    assert rev.version == 3
    assert arxiv.is_new_paper(new, "2026-10-21", 6)
    assert not arxiv.is_new_paper(rev, "2026-10-21", 6)  # v1 was 18 days ago: a revision
    assert not arxiv.is_new_paper(old, "2026-10-21", 6)  # v1 from 2021


def test_verify_entries_batches_ids(clock):
    requested: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        ids = req.url.params["id_list"].split(",")
        requested.append(req.url.params["id_list"])
        return httpx.Response(200, content=atom_feed([{"id": i, "published": "2026-10-20T01:00:00Z"} for i in ids]))

    client = PoliteClient("ua", 3.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    ids = [f"2610.{n:05d}" for n in range(1, 251)]
    out = arxiv.verify_entries(client, "https://export.arxiv.org/api/query", ids, batch=100)
    assert len(out) == 250 and len(requested) == 3
    assert clock.sleeps == [3.0, 3.0]


def test_api_error_entry_is_ignored():
    xml = (b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#bad</id>'
           b'<title>Error</title></entry></feed>')
    assert arxiv.parse_api_feed(xml) == {}
