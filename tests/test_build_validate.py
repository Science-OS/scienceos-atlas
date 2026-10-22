from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atlas import arxiv
from atlas.build import SiteBuilder, build_author_json, github_repo_slug
from atlas.config import load_config
from atlas.hf import HfSignal
from atlas.models import ts
from atlas.store import Store
from atlas.validate import make_validators, validate_site
from conftest import FIXTURES, data_bytes

NOW = datetime(2026, 10, 22, 5, 30, tzinfo=timezone.utc)
NEW_ID = "2610.27576"   # CC BY 4.0
NC_ID = "2610.18625"    # CC BY-NC-ND 4.0


def seed(store: Store) -> list[arxiv.OaiRecord]:
    records, _ = arxiv.parse_list_records(data_bytes("oai_econ_sample.xml"))
    kept = []
    for r in records:
        if r.arxiv_id.startswith("2610") and r.arxiv_id != "2610.04688":
            api = arxiv.ApiEntry(r.arxiv_id, "2026-10-20T03:42:43Z", "2026-10-20T03:42:43Z", 1, r.categories[0])
            store.add_paper(r, api, r.datestamp, ts(NOW))
            kept.append(r)
    return kept


def oa_work_for(rec: arxiv.OaiRecord) -> dict:
    return {"work_id": "W1", "cited_by_count": 3, "publication_date": "2026-10-20", "primary_topic": "T",
            "authorships": [{"id": f"A{900 + i}", "name": a["name"], "orcid": None,
                             "institutions": [{"ror": "022zdgq74", "openalex": "I1", "name": "ENAC", "country": "FR", "type": "education"}]}
                            for i, a in enumerate(rec.authors)]}


@pytest.fixture
def world(tmp_path):
    store = Store(":memory:")
    recs = seed(store)
    target = next(r for r in recs if r.arxiv_id == NEW_ID)
    store.upsert_hf([HfSignal(NEW_ID, upvotes=7, github_repo="https://github.com/o/r", github_stars=11,
                              ai_summary="Machine umpires change outcomes.", project_page="https://p.example/x")], "2026-10-22T04:00:00Z")
    store.set_oa_work(NEW_ID, "found", oa_work_for(target), "2026-10-22T04:05:00Z")
    cards = {f"A{900 + i}": {"id": f"A{900 + i}", "name": a["name"], "alternatives": ["Alt " + a["name"]], "orcid": "0000-0002-3067-9590" if i == 0 else None,
                             "works_count": 10, "cited_by_count": 100, "h_index": 5, "i10_index": 2,
                             "institutions": [{"ror": "022zdgq74", "openalex": "I1", "name": "ENAC", "country": "FR", "type": "education"}],
                             "topics": ["Economics"], "updated_date": "2026-10-20"}
             for i, a in enumerate(target.authors)}
    store.upsert_authors(cards, "2026-10-22T04:06:00Z")
    thumbs = tmp_path / "state" / "thumbs"
    thumbs.mkdir(parents=True)
    for aid in (NEW_ID, NC_ID):
        (thumbs / f"{aid}.jpg").write_bytes(b"\xff\xd8\xff fake")
        store.set_thumb(aid, "ok", 10, "2026-10-22T04:07:00Z")
    site = tmp_path / "site"
    return store, SiteBuilder(store, load_config(), site, thumbs, NOW), site


def load(site: Path, rel: str):
    return json.loads((site / rel).read_text())


def test_end_to_end_site_is_schema_valid(world):
    store, builder, site = world
    stats = builder.build()
    builder.write_status({"duration_s": 1})
    report = validate_site(site)
    assert report.ok, report.errors
    assert stats.shards >= 3 and stats.paper_files == 4
    idx = load(site, "index.json")
    assert idx["schema"] == "scienceos.atlas.feed-manifest/1" and idx["latestDate"] == "2026-10-22"
    assert set(idx) == {"schema", "generatedAt", "latestDate", "shards"}  # strict manifest, no extras
    for ref in idx["shards"]:  # hashes in the manifest match the bytes on disk
        data = (site / ref["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == ref["sha256"] and len(data) == ref["bytes"]
    assert idx["shards"][0]["date"] == "2026-10-22"  # newest day first
    feeds = {(s["feed"], s["date"]) for s in idx["shards"]}
    assert ("today", "2026-10-21") in feeds and ("trending", "2026-10-22") in feeds


def test_latest_aliases_equal_newest_dated_shards(world):
    _, builder, site = world
    builder.build()
    assert (site / "feeds/latest.json").read_bytes() == (site / "feeds/2026-10-22.json").read_bytes()
    assert (site / "feeds/trending/latest.json").read_bytes() == (site / "feeds/trending/2026-10-22.json").read_bytes()
    cat_latest = list((site / "feeds/categories").glob("*/latest.json"))
    assert cat_latest, "category shards missing"


def test_hf_signal_boosts_rank_and_fills_paper_fields(world):
    _, builder, site = world
    builder.build()
    shard = load(site, "feeds/2026-10-21.json")
    first = shard["items"][0]
    assert first["paper"]["arxivId"] == NEW_ID and first["rank"] == 1 and first["reason"] == "trending"
    p = first["paper"]
    assert p["signals"]["hfUpvotes"] == {"value": 7, "source": "hf", "fetchedAt": "2026-10-22T04:00:00Z"}
    assert p["signals"]["githubStars"]["value"] == 11 and p["signals"]["githubRepo"] == "o/r"
    assert {"kind": "code", "url": "https://github.com/o/r", "label": "o/r"} in p["links"]
    assert p["brief"]["source"] == "hf_ai_summary" and p["brief"]["text"].startswith("Machine umpires")
    assert p["citations"] == {"value": 3, "source": "openalex", "fetchedAt": "2026-10-22T04:05:00Z"}
    others = [i for i in shard["items"][1:]]
    assert all(i["reason"] == "new" for i in others)
    assert [i["rank"] for i in shard["items"]] == list(range(1, len(shard["items"]) + 1))
    assert [i["score"] for i in shard["items"]] == sorted((i["score"] for i in shard["items"]), reverse=True)


def test_author_ids_cards_and_files(world):
    _, builder, site = world
    builder.build()
    paper = load(site, f"papers/{NEW_ID}.json")["paper"]
    ids = [a["authorId"] for a in paper["authors"]]
    assert all(i and i.startswith("oa:A9") for i in ids)
    af = load(site, f"authors/{ids[0][3:]}.json")
    a = af["author"]
    assert a["id"] == ids[0] and a["ids"]["openalex"] == ids[0][3:] and a["ids"]["orcid"] == "0000-0002-3067-9590"
    assert a["metrics"]["hIndex"] == [{"value": 5, "source": "openalex", "fetchedAt": "2026-10-22T04:06:00Z"}]
    assert a["enrichment"] == {"stage": "A", "complete": False, "pending": ["photo", "experience", "coauthors"], "updatedAt": "2026-10-22T04:06:00Z"}
    assert a["affiliations"] == [{"id": "ror:022zdgq74", "name": "ENAC", "country": "FR", "type": "education"}]
    assert {"kind": "orcid", "url": "https://orcid.org/0000-0002-3067-9590"} in a["links"]
    assert af["institutions"][0]["ids"] == {"ror": "022zdgq74", "openalex": "I1"}
    # papers without an OpenAlex work carry no author ids
    other = next(p for p in (site / "papers").glob("*.json") if NEW_ID not in p.name)
    assert not any(a.get("authorId") for a in json.loads(other.read_text())["paper"]["authors"])


def test_thumbnails_only_for_redistributable_licences(world):
    _, builder, site = world
    builder.build()
    assert (site / "thumbs" / f"{NEW_ID}.jpg").exists()
    assert not (site / "thumbs" / f"{NC_ID}.jpg").exists()  # CC BY-NC-ND: even though a file exists in state
    ok = load(site, f"papers/{NEW_ID}.json")["paper"]["links"]
    assert {"kind": "other", "label": "thumbnail", "url": f"https://science-os.github.io/scienceos-atlas/thumbs/{NEW_ID}.jpg"} in ok
    nc = load(site, f"papers/{NC_ID}.json")["paper"]
    assert nc["license"].endswith("by-nc-nd/4.0/")
    assert not any(l.get("label") == "thumbnail" for l in nc["links"])


def test_unchanged_shards_keep_their_bytes_across_runs(world):
    store, builder, site = world
    builder.build()
    before = (site / "feeds/2026-10-21.json").read_bytes()
    later = SiteBuilder(store, load_config(), site, builder.thumbs_dir, datetime(2026, 10, 23, 5, 30, tzinfo=timezone.utc))
    later.build()
    assert (site / "feeds/2026-10-21.json").read_bytes() == before  # same content -> same generatedAt -> same sha
    assert load(site, "index.json")["generatedAt"] == "2026-10-23T05:30:00Z"


def test_llm_brief_wins_over_hf_summary(world):
    store, builder, site = world
    store.set_brief(NEW_ID, "LLM brief.", "llm", "m-1", "2026-10-22T04:30:00Z", 0.0001)
    builder.build()
    b = load(site, f"papers/{NEW_ID}.json")["paper"]["brief"]
    assert b == {"text": "LLM brief.", "source": "llm", "model": "m-1", "generatedAt": "2026-10-22T04:30:00Z"}


def test_retention_window_limits_days(world):
    store, _, site = world
    cfg = load_config()
    cfg.feeds.retention_days = 1
    SiteBuilder(store, cfg, site, Path("/nonexistent"), NOW).build()
    assert {s["date"] for s in load(site, "index.json")["shards"]} == {"2026-10-22"}


# ---------------------------------------------------------------- validator


def test_validator_accepts_the_upstream_fixtures():
    vs = make_validators()
    for name, key in [("feed-manifest", "manifest"), ("feed-shard", "shard"), ("paper-file", "paper"), ("author-file", "author")]:
        doc = json.loads((FIXTURES / f"{name}.json").read_text())
        assert not list(vs[key].iter_errors(doc)), name


def test_validator_flags_schema_violations(world):
    _, builder, site = world
    builder.build()
    assert validate_site(site).ok
    shard = site / "feeds/2026-10-22.json"
    doc = json.loads(shard.read_text())
    doc["items"][0]["paper"]["publishedAt"] = "2026-10-22"          # not an RFC 3339 timestamp
    doc["items"][0]["paper"]["surprise"] = 1                        # additionalProperties: false
    shard.write_text(json.dumps(doc))
    rep = validate_site(site)
    assert not rep.ok
    msgs = " ".join(rep.errors)
    assert "publishedAt" in msgs or "surprise" in msgs
    assert any("sha256" in e for e in rep.errors)                   # manifest hash no longer matches


def test_validator_flags_dangling_references(world):
    _, builder, site = world
    builder.build()
    next((site / "authors").glob("*.json")).unlink()
    (site / "thumbs" / f"{NEW_ID}.jpg").unlink()
    rep = validate_site(site)
    assert any("authors/" in e and "missing" in e for e in rep.errors)
    assert any("thumbs/" in e for e in rep.errors)


def test_validator_enforces_size_budget(world):
    _, builder, site = world
    builder.build()
    assert not validate_site(site, max_site_mb=0).ok


def test_github_repo_slug():
    assert github_repo_slug("https://github.com/facebookresearch/vjepa2") == "facebookresearch/vjepa2"
    assert github_repo_slug("https://github.com/o/r.git") == "o/r"
    assert github_repo_slug("https://gitlab.com/o/r") is None
    assert github_repo_slug(None) is None


def test_build_author_json_without_orcid_or_metrics():
    author, insts = build_author_json({"id": "A1", "name": "N", "alternatives": ["N"], "orcid": None, "institutions": [], "topics": []}, "2026-10-22T00:00:00Z")
    assert author["metrics"] == {"citations": [], "hIndex": [], "i10Index": []} and author["aliases"] == [] and insts == []
