"""`atlas run` with a media store: thumbnails move in, entities get resolved, the site publishes the window, a second run is quiet."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import httpx

from atlas.http import PoliteClient
from atlas.media.store import MediaStore
from atlas.openalex import OpenAlexClient
from atlas.pipeline import RunOptions, run
from atlas.store import Store
from media_fakes import BASE, ETH, PAGE, FakeNet, make_cfg, make_http, make_image
from test_build_validate import NC_ID, NEW_ID, NOW, seed

SECOND_CC_BY = "2610.28965"


def seed_state(state_dir: Path) -> None:
    """Papers, one OpenAlex work whose authors include Rod Page (Wikidata photo) at ETH, rendered thumbnails for the CC BY papers."""
    store = Store(state_dir / "atlas.sqlite")
    recs = seed(store)
    target = next(r for r in recs if r.arxiv_id == NEW_ID)
    target.authors[0]["name"] = PAGE["name"]                       # make Rod Page the first author of the paper
    store.conn.execute("UPDATE papers SET record=? WHERE arxiv_id=?", (json.dumps(target.to_json()), NEW_ID))
    store.conn.commit()
    authorships = [{"id": "A5068055706", "name": PAGE["name"], "orcid": PAGE["orcid"],
                    "institutions": [{"ror": ETH["ror"], "openalex": "I1", "name": "ETH Zurich", "country": "CH", "type": "education"}]}]
    authorships += [{"id": f"A{900 + i}", "name": a["name"], "orcid": None, "institutions": []} for i, a in enumerate(target.authors[1:])]
    store.set_oa_work(NEW_ID, "found", {"work_id": "W1", "cited_by_count": 1, "publication_date": "2026-10-20", "primary_topic": "T",
                                        "authorships": authorships}, "2026-10-22T04:05:00Z")
    store.upsert_authors({au["id"]: {"id": au["id"], "name": au["name"], "alternatives": [], "orcid": au["orcid"], "works_count": 1, "cited_by_count": 1,
                                     "h_index": 1, "i10_index": 0, "institutions": au["institutions"], "topics": []} for au in authorships},
                         "2026-10-22T04:06:00Z")
    (state_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    for aid in (NEW_ID, SECOND_CC_BY, NC_ID):
        (state_dir / "thumbs" / f"{aid}.jpg").write_bytes(make_image(600, 780, "JPEG"))
        store.set_thumb(aid, "ok", 1000, "2026-10-22T04:07:00Z")
    store.close()


class Arxiv:
    """An arXiv endpoint that fails the test if anything is requested (no harvest, no PDF re-render)."""

    def __init__(self, clock):
        self.urls: list[str] = []
        self.client = PoliteClient("ua", 0.0, max_retries=0, sleep=clock.sleep, clock=clock.now,
                                   transport=httpx.MockTransport(lambda r: (self.urls.append(str(r.url)), httpx.Response(500))[1]))


def go(tmp_path, clock, *, now=NOW, net=None, cfg=None, **opts):
    cfg = cfg or make_cfg()
    net = net or FakeNet()
    arxiv = Arxiv(clock)
    oa = OpenAlexClient(cfg.openalex, "ua", transport=httpx.MockTransport(lambda r: httpx.Response(500)), api_key="", sleep=lambda s: None)
    res = run(cfg, RunOptions(site_dir=tmp_path / "site", state_dir=tmp_path / "state", media_dir=tmp_path / "media-store", now=now,
                              skip_harvest=True, skip_hf=True, skip_openalex=True, skip_briefs=True, **opts),
              arxiv_client=arxiv.client, hf_client=arxiv.client, oa_client=oa, media_http=make_http(cfg, net))
    return res, net, arxiv


def test_first_run_ingests_thumbnails_resolves_entities_and_publishes_the_window(tmp_path, clock):
    seed_state(tmp_path / "state")
    res, net, arxiv = go(tmp_path, clock, skip_thumbs=True)
    assert res.ok, res.stats["validation"]
    st = res.stats["stages"]
    assert {k: v for k, v in st["media_thumbs"].items() if k != "seconds"} == {"ingested": 2, "already": 0, "not_licensed": 1, "no_paper": 0, "bad_image": 0}
    assert st["media"]["processed_author"] == 2 and st["media"]["processed_institution"] == 1       # Rod Page, his co-author, ETH
    assert st["media"]["found"] == 2 and st["media"]["negative"] == 1                                # the co-author has no Wikidata item
    store = MediaStore(tmp_path / "media-store", BASE)
    page = store.read_entry("author", f"orcid-{PAGE['orcid']}")
    assert page["assets"][0]["hosted"] and store.read_entry("institution", "ror-05a28rw58")["assets"][0]["role"] == "logo"
    assert store.read_entry("paper", f"arxiv-{NEW_ID}") and not store.read_entry("paper", f"arxiv-{NC_ID}")
    site = tmp_path / "site"
    papers = json.loads((site / f"papers/{NEW_ID}.json").read_text())["paper"]
    thumb = next(l for l in papers["links"] if l.get("label") == "thumbnail")
    assert "/media/b/" in thumb["url"] and thumb["url"].endswith(".jpg")
    assert (site / "thumbs" / f"{NEW_ID}.jpg").is_file()                                  # still published for older app builds
    author = json.loads((site / "authors/A5068055706.json").read_text())
    assert author["author"]["photo"]["url"].endswith(".webp") and "photo" not in author["author"]["enrichment"]["pending"]
    assert author["author"]["affiliations"][0]["logoUrl"].endswith(".webp")
    assert (site / "media" / "manifest.json").is_file() and json.loads((site / "media/manifest.json").read_text())["counts"]["paper"] == 2
    assert res.stats["totals"]["authors_with_photo"] == 1 and res.stats["totals"]["hosted_thumbnails"] == 2
    assert not arxiv.urls                                                                  # nothing was fetched from arXiv


def test_second_run_is_quiet_nothing_rewritten_no_source_requests(tmp_path, clock):
    seed_state(tmp_path / "state")
    go(tmp_path, clock, skip_thumbs=True)
    root = tmp_path / "media-store"
    files = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    assert files
    for p in files:
        os.utime(p, ns=(1_000_000_000, 1_000_000_000))
    net = FakeNet()
    res, net, _ = go(tmp_path, clock, now=NOW + timedelta(hours=24), net=net, skip_thumbs=True)
    assert res.ok and res.stats["stages"]["media"]["fresh"] == 3 and res.stats["stages"]["media_thumbs"]["ingested"] == 0
    assert [r for r in net.requests if not r.url.path.startswith("/rest/v1/")] == []
    assert {p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == {1_000_000_000}      # catalogue, bytes and manifest untouched


def test_lost_thumbnail_cache_is_rebuilt_from_the_store_without_rendering_again(tmp_path, clock):
    seed_state(tmp_path / "state")
    go(tmp_path, clock, skip_thumbs=True)
    for f in (tmp_path / "state" / "thumbs").glob("*.jpg"):
        f.unlink()                                                                          # the Actions cache lost state/thumbs
    res, _, arxiv = go(tmp_path, clock, now=NOW + timedelta(hours=24))                      # thumbs stage enabled this time
    t = res.stats["stages"]["thumbs"]
    assert t["restored_from_media"] == 2 and t["made"] == 0 and not any("/pdf/" in u for u in arxiv.urls)
    assert (tmp_path / "state" / "thumbs" / f"{NEW_ID}.jpg").is_file() and res.ok


def test_budget_from_the_config_limits_a_run_and_the_next_run_continues(tmp_path, clock):
    seed_state(tmp_path / "state")
    cfg = make_cfg(max_new_per_run=2)
    r1, _, _ = go(tmp_path, clock, cfg=cfg, skip_thumbs=True)
    m1 = r1.stats["stages"]["media"]
    assert m1.get("processed_author", 0) + m1.get("processed_institution", 0) == 2 and m1["budget_exhausted"]
    r2, _, _ = go(tmp_path, clock, cfg=cfg, now=NOW + timedelta(hours=1), skip_thumbs=True)
    m2 = r2.stats["stages"]["media"]
    assert m2.get("processed_author", 0) + m2.get("processed_institution", 0) == 1 and m2["fresh"] == 2 and not m2["budget_exhausted"]
    r3, _, _ = go(tmp_path, clock, cfg=cfg, now=NOW + timedelta(hours=2), skip_thumbs=True)
    assert r3.stats["stages"]["media"]["fresh"] == 3


def test_media_failure_degrades_the_run_instead_of_failing_it(tmp_path, clock):
    seed_state(tmp_path / "state")
    net = FakeNet()
    net.down.update({"www.wikidata.org": 503, "api.ror.org": 503})
    res, _, _ = go(tmp_path, clock, net=net, skip_thumbs=True)
    assert res.ok and res.stats["stages"]["media"]["errors"] == 3                       # each entity failed, none was cached as "nothing found"
    state = Store(tmp_path / "state" / "atlas.sqlite")
    assert {r["status"] for r in state.conn.execute("SELECT status FROM media_checks")} == {"error"}
    state.close()


def test_no_media_dir_means_no_media_stage(tmp_path, clock):
    seed_state(tmp_path / "state")
    cfg = make_cfg()
    arxiv = Arxiv(clock)
    res = run(cfg, RunOptions(site_dir=tmp_path / "site", state_dir=tmp_path / "state", now=NOW, skip_harvest=True, skip_hf=True,
                              skip_openalex=True, skip_briefs=True, skip_thumbs=True),
              arxiv_client=arxiv.client, hf_client=arxiv.client,
              oa_client=OpenAlexClient(cfg.openalex, "ua", transport=httpx.MockTransport(lambda r: httpx.Response(500)), api_key="", sleep=lambda s: None))
    assert res.ok and "media" not in res.stats["stages"] and not (tmp_path / "site" / "media").exists()


def test_media_disabled_in_config_skips_everything(tmp_path, clock):
    seed_state(tmp_path / "state")
    cfg = make_cfg(enabled=False)
    res, net, _ = go(tmp_path, clock, cfg=cfg, skip_thumbs=True)
    assert res.ok and "media" not in res.stats["stages"] and not net.requests and not (tmp_path / "media-store").exists()


def test_skip_media_flag(tmp_path, clock):
    seed_state(tmp_path / "state")
    res, net, _ = go(tmp_path, clock, skip_thumbs=True, skip_media=True)
    assert res.ok and "media" not in res.stats["stages"] and not net.requests
