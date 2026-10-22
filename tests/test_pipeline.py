from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import httpx
import pytest

from atlas import arxiv
from atlas.config import load_config
from atlas.http import PoliteClient
from atlas.openalex import OpenAlexClient
from atlas.pipeline import RunOptions, run
from atlas.store import Store
from conftest import atom_feed, data_bytes, mock_transport

NOW = datetime(2026, 10, 22, 5, 30, tzinfo=timezone.utc)
RECORDS, _ = arxiv.parse_list_records(data_bytes("oai_econ_sample.xml"))
BY_ID = {r.arxiv_id: r for r in RECORDS}
PUBLISHED = {"2610.04688": ("2026-10-03T18:04:38Z", 3), "2109.14850": ("2021-09-30T05:13:27Z", 2)}


def pdf_bytes() -> bytes:
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument.new()
    doc.new_page(300, 400)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class World:
    """Fake arXiv / HF / OpenAlex endpoints. Records what was requested."""

    def __init__(self, clock, *, oai_fail=False, api_fail=False, unknown_authors=()):
        self.clock, self.oai_fail, self.api_fail = clock, oai_fail, api_fail
        self.unknown_authors = set(unknown_authors)  # author ids the /authors endpoint cannot resolve
        self.arxiv_urls: list[str] = []
        self.oa_urls: list[str] = []
        self.pdf = pdf_bytes()

    def arxiv_handler(self, req: httpx.Request) -> httpx.Response:
        self.arxiv_urls.append(str(req.url))
        host, path, p = req.url.host, req.url.path, dict(req.url.params)
        if host == "oaipmh.arxiv.org":
            if self.oai_fail:
                return httpx.Response(500)
            if p.get("set") == "econ":
                return httpx.Response(200, content=data_bytes("oai_econ_sample.xml"))
            return httpx.Response(200, content=data_bytes("oai_no_records.xml"))
        if path.endswith("/api/query"):
            if self.api_fail:
                return httpx.Response(500)
            ids = p["id_list"].split(",")
            return httpx.Response(200, content=atom_feed([
                {"id": i, "published": PUBLISHED.get(i, ("2026-10-20T03:42:43Z", 1))[0], "v": PUBLISHED.get(i, (0, 1))[1],
                 "primary": BY_ID[i].categories[0]} for i in ids if i in BY_ID]))
        if "/pdf/" in path:
            return httpx.Response(200, content=self.pdf)
        return httpx.Response(404)

    def hf_handler(self, req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"paper": {"id": "2610.27576", "upvotes": 12, "githubRepo": "https://github.com/o/r",
                                                    "githubStars": 3, "ai_summary": "Umpires."},
                                          "submittedOnDailyAt": "2026-10-22T00:00:00.000Z"}])

    def oa_handler(self, req: httpx.Request) -> httpx.Response:
        self.oa_urls.append(str(req.url))
        hdr = {"x-ratelimit-remaining": "990", "x-ratelimit-limit": "1000"}
        path = req.url.path
        if "/works/" in path:
            aid = path.rsplit("arXiv.", 1)[1]
            if aid != "2610.27576":
                return httpx.Response(404, headers=hdr)
            body = {"id": "https://openalex.org/W5", "cited_by_count": 2, "publication_date": "2026-10-20",
                    "primary_topic": {"display_name": "Econ"},
                    "authorships": [{"author": {"id": f"https://openalex.org/A{70 + i}", "display_name": a["name"]},
                                     "institutions": [{"id": "https://openalex.org/I1", "ror": "https://ror.org/022zdgq74",
                                                       "display_name": "ENAC", "country_code": "FR", "type": "education"}]}
                                    for i, a in enumerate(BY_ID["2610.27576"].authors)]}
            return httpx.Response(200, json=body, headers=hdr)
        wanted = [w for w in req.url.params["filter"].split(":", 1)[1].split("|") if w not in self.unknown_authors]
        res = [{"id": f"https://openalex.org/{w}", "display_name": f"Author {w}", "works_count": 3, "cited_by_count": 9,
                "summary_stats": {"h_index": 2, "i10_index": 1}, "last_known_institutions": [], "topics": []} for w in wanted]
        return httpx.Response(200, json={"results": res}, headers={**hdr, "x-ratelimit-credits-used": "1"})

    def clients(self, cfg):
        cfg.openalex.min_interval_s = 0
        a = PoliteClient("ua", 3.0, transport=mock_transport(self.arxiv_handler), sleep=self.clock.sleep, clock=self.clock.now)
        h = PoliteClient("ua", 0.0, transport=mock_transport(self.hf_handler), sleep=self.clock.sleep, clock=self.clock.now)
        o = OpenAlexClient(cfg.openalex, "ua", transport=mock_transport(self.oa_handler), api_key="", sleep=lambda s: None)
        return a, h, o


def opts(tmp_path, **kw) -> RunOptions:
    return RunOptions(site_dir=tmp_path / "site", state_dir=tmp_path / "state", now=NOW, **kw)


def go(cfg, tmp_path, world, **kw):
    a, h, o = world.clients(cfg)
    return run(cfg, opts(tmp_path, **kw), arxiv_client=a, hf_client=h, oa_client=o), a


def test_full_run_ingests_enriches_thumbnails_and_validates(tmp_path, clock):
    pytest.importorskip("pypdfium2")
    cfg, world = load_config(), None
    world = World(clock)
    res, a = go(cfg, tmp_path, world)
    st = res.stats["stages"]
    assert res.ok, res.stats["validation"]
    # harvest: 5 candidates (2109 filtered by id), 2610.04688 is a v3 revision -> rejected, 4 new papers stored
    assert st["harvest"]["candidates"] == 5 and st["harvest"]["rejected_revisions"] == 1
    assert st["harvest"]["added"] == 4 and st["harvest"]["complete"]
    assert st["harvest"]["from"] == "2026-10-19"                     # cold start: yesterday - 2 backfill days
    assert st["hf"]["matched_in_state"] == 1
    assert st["openalex"]["found"] == 1 and st["openalex"]["missing"] == 3
    assert st["openalex"]["authors_fetched"] == len(BY_ID["2610.27576"].authors)
    assert st["openalex"]["targets_found_pct"] == 25.0
    # thumbnails: only the CC BY papers (2610.27576, 2610.28965); NC-ND and non-exclusive are skipped
    assert st["thumbs"]["made"] == 2 and st["thumbs"]["eligible"] == 2
    site = tmp_path / "site"
    assert sorted(p.name for p in (site / "thumbs").glob("*.jpg")) == ["2610.27576.jpg", "2610.28965.jpg"]
    assert (site / "index.json").exists() and (site / "feeds/latest.json").exists() and (site / "status.json").exists()
    status = json.loads((site / "status.json").read_text())
    assert status["days"] == ["2026-10-22", "2026-10-21"] and status["counts"]["papersInWindow"] == 4
    # OAI cursor advanced only because the run was complete
    s = Store(tmp_path / "state" / "atlas.sqlite")
    assert s.get_meta("harvest_cursor") == "2026-10-22"
    s.close()
    # every arXiv-host request (OAI + API + PDF) was spaced by the shared 3 s throttle
    assert a.requests_made == len(world.arxiv_urls)
    assert clock.sleeps.count(3.0) == a.requests_made - 1
    # meta sidecar material
    assert res.stats["totals"]["papers_in_window"] == 4 and res.stats["totals"]["with_openalex_work"] == 1


def test_second_run_is_incremental(tmp_path, clock):
    pytest.importorskip("pypdfium2")
    cfg = load_config()
    world = World(clock)
    go(cfg, tmp_path, world)
    oa_calls, pdf_calls = len(world.oa_urls), sum("/pdf/" in u for u in world.arxiv_urls)
    res, _ = go(cfg, tmp_path, world)
    st = res.stats["stages"]
    assert res.ok
    assert st["harvest"]["added"] == 0 and st["harvest"]["from"] == "2026-10-22"  # cursor, not cold start
    assert st["harvest"]["rejected_revisions"] == 0                               # rejected ids are remembered
    assert st["openalex"]["lookups"] == 0                                         # found: done; missing: retry window not elapsed
    assert len(world.oa_urls) == oa_calls
    assert sum("/pdf/" in u for u in world.arxiv_urls) == pdf_calls               # thumbnails not re-fetched
    assert st["thumbs"]["skipped_existing"] == 2


def test_missing_openalex_works_are_retried_after_the_window(tmp_path, clock):
    cfg = load_config()
    cfg.thumbs.enabled = False
    world = World(clock)
    go(cfg, tmp_path, world)
    a, h, o = world.clients(cfg)
    later = RunOptions(site_dir=tmp_path / "site", state_dir=tmp_path / "state", now=datetime(2026, 10, 23, 5, 30, tzinfo=timezone.utc))
    res = run(cfg, later, arxiv_client=a, hf_client=h, oa_client=o)
    assert res.stats["stages"]["openalex"]["lookups"] == 3                         # the 3 missing ones, 24 h later


def test_limit_caps_papers_and_does_not_advance_cursor(tmp_path, clock):
    cfg = load_config()
    cfg.thumbs.enabled = False
    res, _ = go(cfg, tmp_path, World(clock), limit=2)
    assert res.ok and res.stats["stages"]["harvest"]["added"] == 2
    assert res.stats["stages"]["harvest"]["complete"] is False
    s = Store(tmp_path / "state" / "atlas.sqlite")
    assert s.get_meta("harvest_cursor") is None and s.count_papers() == 2
    s.close()


def test_source_outage_degrades_instead_of_crashing(tmp_path, clock):
    cfg = load_config()
    cfg.thumbs.enabled = False
    cfg.arxiv.max_retries = 0
    res, _ = go(cfg, tmp_path, World(clock, oai_fail=True))
    assert res.ok                                                                  # site still builds (empty) and validates
    h = res.stats["stages"]["harvest"]
    assert h["added"] == 0 and h["complete"] is False and len(h["errors"]) == 5    # every set failed, none crashed the run
    s = Store(tmp_path / "state" / "atlas.sqlite")
    assert s.get_meta("harvest_cursor") is None                                    # outage: retry the same window tomorrow
    s.close()


def test_api_verification_failure_adds_nothing_and_keeps_going(tmp_path, clock):
    cfg = load_config()
    cfg.thumbs.enabled = False
    cfg.arxiv.max_retries = 0
    res, _ = go(cfg, tmp_path, World(clock, api_fail=True))
    h = res.stats["stages"]["harvest"]
    assert res.ok and h["added"] == 0 and h["complete"] is False and any("verify" in e for e in h["errors"])


def test_openalex_rate_limit_stops_enrichment_but_not_the_run(tmp_path, clock):
    cfg = load_config()
    cfg.thumbs.enabled = False
    world = World(clock)
    world.oa_handler = lambda req: httpx.Response(429, headers={"Retry-After": "86400"})  # type: ignore[method-assign]
    res, _ = go(cfg, tmp_path, world)
    oa = res.stats["stages"]["openalex"]
    assert res.ok and oa["found"] == 0 and "429" in oa["stopped_reason"]
    assert res.stats["stages"]["harvest"]["added"] == 4                            # ingestion unaffected


def test_offline_rebuild_from_state(tmp_path, clock):
    cfg = load_config()
    cfg.thumbs.enabled = False
    go(cfg, tmp_path, World(clock))
    import shutil
    shutil.rmtree(tmp_path / "site")
    res = run(cfg, opts(tmp_path, skip_harvest=True, skip_hf=True, skip_openalex=True, skip_thumbs=True, skip_briefs=True))
    assert res.ok and (tmp_path / "site" / "index.json").exists()
    assert res.stats["totals"]["papers_in_window"] == 4


def test_briefs_stay_off_without_a_secret(tmp_path, clock, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config()
    cfg.thumbs.enabled = False
    res, _ = go(cfg, tmp_path, World(clock))
    assert res.stats["stages"]["briefs"]["enabled"] is False
    # HF ai_summary is not republished unless briefs.hf_summary_fallback is on
    paper = json.loads((tmp_path / "site" / "papers" / "2610.27576.json").read_text())["paper"]
    assert "brief" not in paper
    other = json.loads((tmp_path / "site" / "papers" / "2610.27224.json").read_text())["paper"]
    assert "brief" not in other


def test_llm_briefs_are_published_cached_and_cost_guarded(tmp_path, clock, monkeypatch):
    from atlas import pipeline
    from atlas.briefs import BriefResult

    class Fake:
        name, model = "openai", "gpt-6-luna"

        def __init__(self):
            self.calls: list[str] = []

        def generate(self, title, abstract, max_output_tokens):
            self.calls.append(title)
            return BriefResult(f"Brief of {title}", 400, 50)

    fake = Fake()
    cfg = load_config()
    cfg.thumbs.enabled = False
    monkeypatch.setattr(pipeline, "resolve_provider", lambda c: (fake, c.providers[c.provider]))

    res, _ = go(cfg, tmp_path, World(clock))
    b1 = res.stats["stages"]["briefs"]
    assert b1["enabled"] and b1["generated"] == 4 and b1["failed"] == 0 and b1["stopped_reason"] is None
    assert b1["cost_usd"] == round(4 * (400 * 0.10 + 50 * 0.50) / 1e6, 5)
    assert res.stats["totals"]["with_brief"] == 4
    paper = json.loads((tmp_path / "site" / "papers" / "2610.27576.json").read_text())["paper"]
    assert paper["brief"]["source"] == "llm" and paper["brief"]["model"] == "gpt-6-luna"
    assert paper["brief"]["text"].startswith("Brief of ")

    # second run: every paper is cached, the provider is never called, the cost is zero
    n_calls = len(fake.calls)
    res2, _ = go(cfg, tmp_path, World(clock))
    b2 = res2.stats["stages"]["briefs"]
    assert (b2["generated"], b2["cached"], b2["cost_usd"]) == (0, 4, 0)
    assert len(fake.calls) == n_calls

    # a fresh state with a tiny cap: the guard stops the stage and the run still succeeds
    cfg.briefs.max_cost_usd_per_run = 0.0002
    res3, _ = go(cfg, tmp_path / "capped", World(clock))
    b3 = res3.stats["stages"]["briefs"]
    assert res3.ok and b3["generated"] < 4 and "cost_guard" in b3["stopped_reason"]


def author_calls(world) -> int:
    return sum("/authors" in u for u in world.oa_urls)


def test_unresolved_authors_are_negative_cached_and_retried_daily(tmp_path, clock):
    cfg = load_config()
    cfg.thumbs.enabled = False
    world = World(clock, unknown_authors={"A70"})
    res, _ = go(cfg, tmp_path, world)
    oa = res.stats["stages"]["openalex"]
    n_authors = len(BY_ID["2610.27576"].authors)
    assert res.ok and oa["authors_fetched"] == n_authors - 1 and oa["authors_unresolved"] == 1
    paper = json.loads((tmp_path / "site" / "papers" / "2610.27576.json").read_text())["paper"]
    assert "authorId" not in paper["authors"][0]                                   # no card -> no dangling id
    assert all(a["authorId"] for a in paper["authors"][1:])
    assert not (tmp_path / "site" / "authors" / "A70.json").exists()
    before = author_calls(world)
    go(cfg, tmp_path, world)                                                       # same day: nothing to ask again
    assert author_calls(world) == before
    a, h, o = world.clients(cfg)                                                   # two days later: only the unresolved id is retried
    later = RunOptions(site_dir=tmp_path / "site", state_dir=tmp_path / "state", now=datetime(2026, 10, 24, 5, 30, tzinfo=timezone.utc))
    run(cfg, later, arxiv_client=a, hf_client=h, oa_client=o)
    retried = [u for u in world.oa_urls[before:] if "/authors" in u]
    assert any("A70" in u for u in retried)
