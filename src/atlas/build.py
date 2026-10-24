"""Assemble the static site from the SQLite state. Pure function of (state, config, now): no network."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .arxiv import ApiEntry
from .config import Config
from .hf import HfSignal
from .licenses import allows_redistribution
from .matching import match_authors
from .media import keys as media_keys
from .media.catalogue import as_photo, first_asset
from .media.papers import thumbnail_url
from .media.publish import publish_to_site
from .media.store import MediaStore
from .models import (
    Author, AuthorExternalIds, AuthorMetrics, Brief, Enrichment, Institution, InstitutionExternalIds, InstitutionRef,
    LinkOut, MetricValue, Paper, PaperAuthor, PaperSignals, Photo, dump, ts,
)
from .scoring import score_paper
from .store import Store, StoredPaper
from .thumbs import thumb_path

log = logging.getLogger("atlas.build")

SOURCES = ["arxiv", "openalex", "huggingface"]
STATIC_DIR = Path(__file__).resolve().parents[2] / "site_static"
_GH_REPO = re.compile(r"^https?://github\.com/([^/\s]+/[^/\s#?]+?)(?:\.git)?/?$", re.IGNORECASE)


def write_json(path: Path, obj: Any, *, pretty: bool = False) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=1 if pretty else None, separators=None if pretty else (",", ":"))
    data = (text + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def iso_to_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class PaperView:
    """Everything the ranking needs about one paper, plus its schema-ready JSON."""
    arxiv_id: str
    feed_date: str
    published: datetime
    categories: list[str]
    hf_upvotes: int | None
    github_stars: int | None
    has_code: bool
    json: dict[str, Any]
    license: str | None


@dataclass
class BuildStats:
    dates: list[str] = field(default_factory=list)
    papers_in_window: int = 0
    shards: int = 0
    paper_files: int = 0
    author_files: int = 0
    thumbs: int = 0
    site_bytes: int = 0
    papers_with_hf: int = 0
    papers_with_openalex: int = 0
    papers_with_brief: int = 0
    papers_with_author_ids: int = 0
    papers_with_hosted_thumb: int = 0
    authors_with_photo: int = 0
    institutions_with_logo: int = 0
    media_entries: int = 0
    media_blobs: int = 0
    media_bytes: int = 0


def github_repo_slug(url: str | None) -> str | None:
    if not url:
        return None
    m = _GH_REPO.match(url.strip())
    return m.group(1) if m else None


def _logo_photo(media: MediaStore | None, ror: str | None) -> Photo | None:
    """First logo asset of an institution in the central catalogue (hosted first), as a Link `Photo`."""
    if media is None or not ror:
        return None
    a = first_asset(media.lookup("institution", media_keys.institution_keys({"ror": ror})), "logo")
    return Photo(**as_photo(a)) if a else None


def build_author_json(card: dict[str, Any], fetched_at: str, media: MediaStore | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """OpenAlex author card -> (schema Author, institutions[]). Light card: the Mac resolver does the rest.
    With a media store, photo and logos come from its catalogue (identity chains only; nothing is scraped here)."""
    oa_id = card["id"]
    name = card.get("name") or oa_id
    insts = card.get("institutions") or []
    logos = {i["ror"]: _logo_photo(media, i["ror"]) for i in insts}
    affiliations = [
        InstitutionRef(id=f"ror:{i['ror']}", name=i["name"], country=i.get("country"), type=i.get("type"),
                       logoUrl=logos[i["ror"]].url if logos[i["ror"]] else None)
        for i in insts
    ]
    photo = None
    if media is not None:
        a = first_asset(media.lookup("author", media_keys.author_keys({"orcid": card.get("orcid"), "openalex": oa_id})), "photo")
        photo = Photo(**as_photo(a)) if a else None
    ids = AuthorExternalIds(openalex=oa_id, orcid=card.get("orcid"))
    links = [LinkOut(kind="other", url=f"https://openalex.org/{oa_id}", label="OpenAlex")]
    if card.get("orcid"):
        links.insert(0, LinkOut(kind="orcid", url=f"https://orcid.org/{card['orcid']}"))

    def metric(v: Any) -> list[MetricValue]:
        return [MetricValue(value=v, source="openalex", fetchedAt=fetched_at)] if isinstance(v, (int, float)) else []

    metrics = AuthorMetrics(
        citations=metric(card.get("cited_by_count")),
        hIndex=metric(card.get("h_index")),
        i10Index=metric(card.get("i10_index")),
        worksCount=metric(card.get("works_count")) or None,
    )
    # OpenAlex sometimes merges a whole author list into one "alternative" ("Che, Yifeng [Idaho …]; Calvin, …"): that is not a
    # name, and its length (> the schema's 256) failed validation and stopped the deploy. Keep only name-shaped aliases.
    aliases = [a for a in dict.fromkeys(card.get("alternatives") or [])
               if a != name and len(a) <= 120 and ";" not in a and "[" not in a][:10]
    author = Author(
        id=f"oa:{oa_id}", name=name, aliases=aliases, ids=ids, photo=photo, affiliations=affiliations, experience=[], links=links,
        metrics=metrics, citationsByYear=[], topics=card.get("topics") or [], coauthors=[],
        claimed=False,
        enrichment=Enrichment(stage="A", complete=False, pending=(["photo"] if photo is None else []) + ["experience", "coauthors"],
                              updatedAt=fetched_at),
        updatedAt=fetched_at,
    )
    institutions = [
        dump(Institution(
            id=f"ror:{i['ror']}", name=i["name"], country=i.get("country"), type=i.get("type"), logo=logos[i["ror"]],
            ids=InstitutionExternalIds(ror=i["ror"], openalex=i.get("openalex")), updatedAt=fetched_at,
        ))
        for i in insts
    ]
    return dump(author), institutions


class SiteBuilder:
    def __init__(self, store: Store, cfg: Config, site_dir: Path, thumbs_dir: Path, now: datetime, media: MediaStore | None = None):
        self.store, self.cfg, self.site, self.thumbs_dir, self.now = store, cfg, site_dir, thumbs_dir, now
        self.media = media
        self.now_ts = ts(now)
        self.stats = BuildStats()

    # ------------------------------------------------------------------ papers
    def _paper_view(self, sp: StoredPaper, hf: tuple[HfSignal, str] | None, oa: dict[str, Any] | None,
                    card_ids: set[str], brief: dict[str, Any] | None, thumb_url: str | None) -> PaperView:
        rec, api = sp.record, sp.api
        aid = rec.arxiv_id
        cats = list(dict.fromkeys(rec.categories))
        primary = api.primary_category or (cats[0] if cats else None)
        if primary:
            cats = [primary] + [c for c in cats if c != primary]
        author_ids: dict[int, str] = {}
        if oa and oa.get("work"):
            for pos, au in match_authors([a["name"] for a in rec.authors], oa["work"]["authorships"]).items():
                if au["id"] in card_ids:
                    author_ids[pos] = f"oa:{au['id']}"
        authors = [
            PaperAuthor(name=a["name"], authorId=author_ids.get(i), position=i, affiliation=a.get("affiliation"))
            for i, a in enumerate(rec.authors)
        ]
        links = [LinkOut(kind="abs", url=f"https://arxiv.org/abs/{aid}"), LinkOut(kind="pdf", url=f"https://arxiv.org/pdf/{aid}")]
        signals = None
        stars = upvotes = None
        has_code = False
        if hf:
            sig, hf_at = hf
            upvotes, stars = sig.upvotes, sig.github_stars
            slug = github_repo_slug(sig.github_repo)
            has_code = bool(sig.github_repo)
            if sig.github_repo:
                links.append(LinkOut(kind="code", url=sig.github_repo, label=slug))
            if sig.project_page:
                links.append(LinkOut(kind="project", url=sig.project_page))
            parts = PaperSignals(
                hfUpvotes=MetricValue(value=upvotes, source="hf", fetchedAt=hf_at) if upvotes is not None else None,
                githubStars=MetricValue(value=stars, source="hf", fetchedAt=hf_at) if stars is not None else None,
                githubRepo=slug,
            )
            signals = parts if dump(parts) else None
        if thumb_url:
            links.append(LinkOut(kind="other", url=thumb_url, label="thumbnail"))
        citations = None
        if oa and oa.get("work") and isinstance(oa["work"].get("cited_by_count"), int):
            citations = MetricValue(value=oa["work"]["cited_by_count"], source="openalex", fetchedAt=oa["fetched_at"])
        brief_model: Brief | None = None
        if brief:
            brief_model = Brief(text=brief["text"][:2000], source=brief["source"], model=brief["model"], generatedAt=brief["generated_at"])
        elif hf and hf[0].ai_summary and self.cfg.briefs.hf_summary_fallback:
            brief_model = Brief(text=hf[0].ai_summary[:2000], source="hf_ai_summary", generatedAt=hf[1])
        paper = Paper(
            id=f"arxiv:{aid}", arxivId=aid, doi=rec.doi or f"10.48550/arXiv.{aid}", title=rec.title[:1024],
            abstract=rec.abstract[:20000] or None, authors=authors, categories=cats, primaryCategory=primary,
            publishedAt=api.published, updatedAt=api.updated if api.updated != api.published else None,
            version=api.version, license=rec.license, venue=rec.journal_ref[:256] if rec.journal_ref else None,
            links=links, citations=citations, signals=signals, brief=brief_model,
        )
        if oa and oa.get("work"):
            self.stats.papers_with_openalex += 1
        if author_ids:
            self.stats.papers_with_author_ids += 1
        if hf:
            self.stats.papers_with_hf += 1
        if brief_model:
            self.stats.papers_with_brief += 1
        return PaperView(aid, sp.feed_date, iso_to_dt(api.published), cats, upvotes, stars, has_code, dump(paper), rec.license)

    def _score(self, v: PaperView, shard_date: str) -> float:
        age = max(0.0, (date.fromisoformat(shard_date) - v.published.date()).days)
        return score_paper(self.cfg.scoring, hf_upvotes=v.hf_upvotes, github_stars=v.github_stars,
                           n_categories=len(v.categories), age_days=age, has_code=v.has_code)

    def _ranked(self, views: list[PaperView], shard_date: str, limit: int) -> list[tuple[PaperView, float]]:
        scored = [(v, self._score(v, shard_date)) for v in views]
        scored.sort(key=lambda t: (-t[1], -t[0].published.timestamp(), t[0].arxiv_id))
        return scored[:limit]

    # ------------------------------------------------------------------ build
    def build(self) -> BuildStats:
        cfg, site = self.cfg, self.site
        if site.exists():
            shutil.rmtree(site)
        site.mkdir(parents=True)
        all_dates = self.store.feed_dates()
        if not all_dates:
            log.warning("No papers in state; writing an empty manifest")
        latest = all_dates[-1] if all_dates else self.now.date().isoformat()
        window_start = (date.fromisoformat(latest) - timedelta(days=cfg.feeds.retention_days - 1)).isoformat()
        dates = [d for d in all_dates if d >= window_start]
        self.stats.dates = dates
        stored = self.store.papers_between(window_start, latest) if all_dates else []
        self.stats.papers_in_window = len(stored)

        hf, oa_rows, author_rows = self.store.hf_signals(), self.store.oa_work_rows(), self.store.author_rows()
        briefs, thumb_rows = self.store.brief_rows(), self.store.thumb_rows()
        author_rows = {k: v for k, v in author_rows.items() if not v[0].get("missing")}  # negative-cache markers
        card_ids = set(author_rows)
        views: dict[str, PaperView] = {}
        legacy_thumbs: set[str] = set()   # thumbs/<id>.jpg is still published: app builds that predate the media store build that path
        base = cfg.site_base_url.rstrip("/")
        for sp in stored:
            aid = sp.record.arxiv_id
            licensed = allows_redistribution(sp.record.license, cfg.thumbs.allowed_license_prefixes)
            thumb_ok = thumb_rows.get(aid, {}).get("status") == "ok" and licensed and thumb_path(self.thumbs_dir, aid).exists()
            if thumb_ok:
                legacy_thumbs.add(aid)
            hosted = thumbnail_url(self.media, aid) if self.media is not None and licensed else None
            thumb_url = hosted or (f"{base}/thumbs/{thumb_path(Path('.'), aid).name}" if thumb_ok else None)
            if hosted:
                self.stats.papers_with_hosted_thumb += 1
            oa = oa_rows.get(aid)
            views[aid] = self._paper_view(sp, hf.get(aid), oa if oa and oa["status"] == "found" else None,
                                          card_ids, briefs.get(aid), thumb_url)
        by_date: dict[str, list[PaperView]] = {d: [] for d in dates}
        for v in views.values():
            by_date[v.feed_date].append(v)

        stamps: dict[str, dict[str, str]] = json.loads(self.store.get_meta("shard_stamps") or "{}")
        new_stamps: dict[str, dict[str, str]] = {}
        manifest_shards: list[dict[str, Any]] = []
        file_paper_ids: set[str] = set()
        latest_for_feed: dict[str, str] = {}  # feed -> relative path of newest dated shard

        def emit(feed: str, d: str, rel: str, ranked: list[tuple[PaperView, float]], reason: str | None) -> None:
            items = [
                {"paper": v.json, "rank": i, "reason": reason or ("trending" if (v.hf_upvotes or 0) > 0 else "new"), "score": s}
                for i, (v, s) in enumerate(ranked, start=1)
            ]
            body = {"schema": "scienceos.atlas.feed-shard/1", "feed": feed, "date": d, "generatedAt": "", "sources": SOURCES, "items": items}
            content = hashlib.sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            stamp = stamps.get(rel)
            generated = stamp["generatedAt"] if stamp and stamp.get("content") == content else self.now_ts
            body["generatedAt"] = generated
            new_stamps[rel] = {"content": content, "generatedAt": generated}
            sha, nbytes = write_json(site / rel, body)
            manifest_shards.append({"feed": feed, "date": d, "path": rel, "count": len(items), "sha256": sha, "bytes": nbytes})
            latest_for_feed[feed] = rel  # dates are iterated ascending, so the last write wins
            self.stats.shards += 1

        for d in dates:
            day_views = by_date[d]
            today = self._ranked(day_views, d, cfg.feeds.today_max_items)
            emit("today", d, f"feeds/{d}.json", today, None)
            for v, _ in self._ranked(day_views, d, cfg.feeds.paper_files_top_n):
                file_paper_ids.add(v.arxiv_id)
            window = [x for dd in dates if (date.fromisoformat(d) - date.fromisoformat(dd)).days in range(0, cfg.feeds.trending_window_days)
                      for x in by_date[dd]]
            trending = self._ranked(window, d, cfg.feeds.trending_max_items)
            emit("trending", d, f"feeds/trending/{d}.json", trending, "trending")
            for v, _ in trending:
                file_paper_ids.add(v.arxiv_id)
            for cat in cfg.feeds.categories:
                in_cat = [v for v in day_views if cat in v.categories]
                if in_cat:
                    emit(f"cat:{cat}", d, f"feeds/categories/{cat}/{d}.json", self._ranked(in_cat, d, cfg.feeds.category_max_items), "category")

        self.store.set_meta("shard_stamps", json.dumps(new_stamps))
        for feed, rel in latest_for_feed.items():  # latest.json aliases (not listed in the manifest)
            alias = Path(rel).parent / "latest.json" if feed != "today" else Path("feeds/latest.json")
            shutil.copyfile(site / rel, site / alias)

        # papers/<id>.json
        for aid in sorted(file_paper_ids):
            write_json(site / "papers" / f"{aid.replace('/', '_')}.json", {"schema": "scienceos.atlas.paper/1", "paper": views[aid].json})
            self.stats.paper_files += 1

        # authors/<openalexId>.json for every card referenced by a paper in the window
        referenced: set[str] = set()
        for v in views.values():
            for a in v.json["authors"]:
                if a.get("authorId"):
                    referenced.add(a["authorId"].split(":", 1)[1])
        for oa_id in sorted(referenced):
            card, fetched = author_rows[oa_id]
            author, insts = build_author_json(card, fetched, self.media)
            write_json(site / "authors" / f"{oa_id}.json", {"schema": "scienceos.atlas.author/1", "author": author, "institutions": insts})
            self.stats.author_files += 1
            self.stats.authors_with_photo += 1 if "photo" in author else 0
            self.stats.institutions_with_logo += sum(1 for i in insts if "logo" in i)

        # thumbs: copy only what is still in the window AND still licence-permitted
        for aid in sorted(legacy_thumbs):
            dst = thumb_path(site / "thumbs", aid)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(thumb_path(self.thumbs_dir, aid), dst)
            self.stats.thumbs += 1

        # the central media store: everything it holds, except the JPEGs (and entries) of papers that left the window
        if self.media is not None:
            ms = publish_to_site(self.media, site, set(views), self.now_ts)
            self.stats.media_entries, self.stats.media_blobs, self.stats.media_bytes = ms["entries"], ms["blobs"], ms["bytes"]

        # manifest (newest first; today, trending, then categories) + status
        feed_order = {"today": 0, "trending": 1}
        manifest_shards.sort(key=lambda s: (feed_order.get(s["feed"], 2), s["feed"]))
        manifest_shards.sort(key=lambda s: s["date"], reverse=True)  # stable: newest day first, then today/trending/cats
        manifest = {"schema": "scienceos.atlas.feed-manifest/1", "generatedAt": self.now_ts, "latestDate": latest, "shards": manifest_shards}
        write_json(site / "index.json", manifest, pretty=True)

        if STATIC_DIR.exists():
            for f in STATIC_DIR.iterdir():
                if f.is_file():
                    shutil.copyfile(f, site / f.name)
        self.stats.site_bytes = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
        return self.stats

    def write_status(self, run: dict[str, Any]) -> None:
        """status.json: the extras that FeedManifest (additionalProperties:false) cannot carry."""
        cats = sorted({s.split(":", 1)[1] for s in self._feeds_in_manifest() if s.startswith("cat:")})
        per_date: dict[str, int] = {}
        for sp in self.store.papers_between(self.stats.dates[0], self.stats.dates[-1]) if self.stats.dates else []:
            per_date[sp.feed_date] = per_date.get(sp.feed_date, 0) + 1
        status = {
            "schema": "scienceos.atlas.status/1",
            "generatedAt": self.now_ts,
            "latestDate": self.stats.dates[-1] if self.stats.dates else None,
            "days": sorted(self.stats.dates, reverse=True),
            "categories": cats,
            "counts": {
                "papersInWindow": self.stats.papers_in_window,
                "papersByDate": dict(sorted(per_date.items(), reverse=True)),
                "shards": self.stats.shards,
                "paperFiles": self.stats.paper_files,
                "authorFiles": self.stats.author_files,
                "thumbs": self.stats.thumbs,
            },
            "sources": SOURCES,
            "license": "Data: arXiv metadata (CC0), OpenAlex (CC0), Hugging Face Daily Papers signals and links. "
                       "Thumbnails only for papers licensed CC BY, CC BY-SA or CC0. No PDFs are served.",
            "run": run,
        }
        write_json(self.site / "status.json", status, pretty=True)

    def _feeds_in_manifest(self) -> list[str]:
        return [s["feed"] for s in json.loads((self.site / "index.json").read_text())["shards"]]
