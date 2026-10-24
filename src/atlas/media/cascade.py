"""The two identity cascades, ported from the Mac resolver. Authors: ORCID / Wikidata / OpenAlex -> Wikidata -> P18 -> Commons
licence gate -> hosted photo, plus link-only GitHub and Bluesky avatars. Institutions: ROR -> Wikidata -> P154 -> licence gate
-> hosted logo, else a homepage icon, else faviconV2 (both link-only). Nothing is ever attached from a name search."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..matching import name_similarity
from . import keys as K
from .catalogue import asset
from .images import ImageError, normalise_image
from .sources import (
    MediaHttp, SourceError, bluesky_avatar, credit_for, fetch_image, filepath_url, find_qid, github_avatar, license_info,
    license_verdict, org_facts, person_facts, ror_record, wikidata_entity,
)
from .store import MediaStore
from .web import favicon, homepage_icon, is_public_http_url

log = logging.getLogger("atlas.media.cascade")

NAME_MIN = 0.6          # a Wikidata label found through a reverse id lookup must resemble the name we hold
CONF_COMMONS = 0.95
CONF_GITHUB = 0.9
CONF_BLUESKY = 0.85
CONF_LOGO = 0.9
CONF_HOMEPAGE = 0.7
CONF_FAVICON = 0.5


@dataclass
class Resolved:
    kind: str
    ids: dict[str, str]
    assets: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    partial: bool = False       # an optional source failed: keep the result but look again soon

    @property
    def found(self) -> bool:
        return bool(self.assets)


class Cascade:
    def __init__(self, cfg: Config, http: MediaHttp, store: MediaStore, now: str, etags: dict[str, str] | None = None):
        self.cfg, self.http, self.store, self.now = cfg, http, store, now
        self.etags = etags if etags is not None else {}   # image fetch url -> ETag; the caller persists it
        self.stats: Counter[str] = Counter()

    # -- shared pieces ------------------------------------------------------------
    def _optional(self, res: Resolved, label: str, fn):
        """Run a link-only source; its failure keeps the rest of the result (and shortens its lifetime)."""
        try:
            return fn()
        except SourceError as exc:
            log.warning("%s failed for %s: %s", label, res.ids, exc)
            res.partial = True
            self.stats[f"error_{label}"] += 1
            return None

    def _prior(self, kind: str, ids: dict[str, str]) -> list[dict[str, Any]]:
        e = self.store.lookup(kind, K.entity_keys(kind, ids))
        return list((e or {}).get("assets", []))

    def _commons_asset(self, res: Resolved, role: str, file_name: str, confidence: float) -> dict[str, Any] | None:
        """Licence gate, then fetch the 256 px render and host it. Returns None (with a note) when it may not or cannot be hosted."""
        info = license_info(self.http, file_name)
        verdict = license_verdict(info)
        credit = credit_for(info) if info else None
        if verdict != "free" or not credit:
            note = f"license_{verdict}" if verdict != "free" else "no_attribution"
            res.notes.append(note)
            self.stats[note] += 1
            return None
        cfg = self.cfg.media
        source_url = filepath_url(file_name)                    # no width: the file itself (contract sourceUrl)
        fetch_url = filepath_url(file_name, cfg.max_px)         # Commons serves a scaled render (a PNG for SVG files)
        prior = next((a for a in self._prior(res.kind, res.ids)
                      if a.get("hosted") and a.get("role") == role and a.get("sourceUrl") == source_url
                      and self.store.blob_path(a.get("sha256", ""), "webp").exists()), None)
        got = fetch_image(self.http.wikimedia, fetch_url, cfg.max_download_mb * 1024 * 1024, self.etags.get(fetch_url) if prior else None)
        if got.status == "not_modified" and prior:
            self.stats["not_modified"] += 1
            fresh = dict(prior)
            fresh.update(license=info["license"][:64], fetchedAt=self.now)
            return fresh
        if got.status != "ok":
            res.notes.append("image_missing")
            return None
        try:
            norm = normalise_image(got.data, cfg.max_px, cfg.webp_quality)
        except ImageError as exc:
            res.notes.append("image_undecodable")
            log.info("skip %s: %s", file_name, exc)
            return None
        sha, created = self.store.put_bytes(norm.data, norm.ext)
        self.stats["blobs_created" if created else "blobs_reused"] += 1
        if got.etag:
            self.etags[fetch_url] = got.etag
        return asset(role, self.store.blob_url(sha, norm.ext), True, "wikidata_commons", self.now, sha256=sha, mime=norm.mime,
                     width=norm.width, height=norm.height, sourceUrl=source_url, license=info["license"][:64],
                     licenseUrl=info.get("licenseUrl"), attribution=credit, attributionUrl=info.get("descriptionUrl"),
                     confidence=confidence)

    # -- authors --------------------------------------------------------------------
    def resolve_author(self, ids: dict[str, str | None], name: str | None = None, alt_names: list[str] | None = None) -> Resolved:
        clean = {"orcid": K.orcid_id(ids.get("orcid")), "wikidata": K.wikidata_id(ids.get("wikidata")),
                 "openalex": K.openalex_id(ids.get("openalex"))}
        res = Resolved("author", {k: v for k, v in clean.items() if v})
        qid, reverse = clean["wikidata"], False
        if not qid and clean["orcid"]:
            qid, reverse = find_qid(self.http, "P496", clean["orcid"]), True
        if not qid and clean["openalex"]:
            qid, reverse = find_qid(self.http, "P10283", clean["openalex"]), True
        if not qid:
            res.notes.append("no_wikidata")
            return res
        ent = wikidata_entity(self.http, qid)
        if ent is None:
            res.notes.append("wikidata_missing")
            return res
        facts = person_facts(ent)
        names = [n for n in [name, *(alt_names or [])] if n]
        if reverse and names and facts["label"] and max(name_similarity(facts["label"], n) for n in names) < NAME_MIN:
            res.notes.append("name_mismatch")   # the id chain says one thing, the label another: attach nothing
            self.stats["name_mismatch"] += 1
            return res
        res.ids["wikidata"] = qid
        if K.orcid_id(facts["orcid"]):
            res.ids.setdefault("orcid", K.orcid_id(facts["orcid"]))
        if facts["openalex"] and K.openalex_id(facts["openalex"][0]):
            res.ids.setdefault("openalex", K.openalex_id(facts["openalex"][0]))

        if facts["image"]:
            photo = self._commons_asset(res, "photo", facts["image"], CONF_COMMONS)
            if photo:
                res.assets.append(photo)
                self.stats["photo_hosted"] += 1
        else:
            res.notes.append("no_p18")

        login = facts["github"]
        if login:
            url = self._optional(res, "github", lambda: github_avatar(self.http, login, self.cfg.media.max_px))
            if url:
                res.ids["github"] = login
                res.assets.append(asset("photo", url, False, "github", self.now, attribution=f"GitHub profile @{login}",
                                        attributionUrl=f"https://github.com/{login}", confidence=CONF_GITHUB))
                self.stats["photo_github"] += 1
        handle = (facts["bluesky"] or "").lstrip("@").lower()
        if handle:
            prof = self._optional(res, "bluesky", lambda: bluesky_avatar(self.http, handle))
            if prof:
                res.ids["bluesky"] = prof["handle"]
                res.assets.append(asset("photo", prof["avatar"], False, "bluesky", self.now,
                                        attribution=f"Bluesky profile @{prof['handle']}",
                                        attributionUrl=f"https://bsky.app/profile/{prof['handle']}", confidence=CONF_BLUESKY))
                self.stats["photo_bluesky"] += 1
        return res

    # -- institutions ------------------------------------------------------------------
    def resolve_institution(self, ids: dict[str, str | None], website: str | None = None) -> Resolved:
        ror, qid = K.ror_id(ids.get("ror")), K.wikidata_id(ids.get("wikidata"))
        res = Resolved("institution", {k: v for k, v in {"ror": ror, "wikidata": qid}.items() if v})
        if ror:
            org = ror_record(self.http, ror)
            if org:
                qid = qid or K.wikidata_id(org["wikidata"])
                website = website or org["website"]
        facts: dict[str, Any] = {}
        if qid:
            ent = wikidata_entity(self.http, qid)
            if ent:
                facts = org_facts(ent)
                res.ids["wikidata"] = qid
                if not ror and K.ror_id(facts["ror"]):
                    ror = K.ror_id(facts["ror"])
                    res.ids["ror"] = ror
                website = website or facts["website"]
        if not facts:
            res.notes.append("no_wikidata")

        for file_name in facts.get("logos", [])[:3]:   # preferred rank first; the first one that passes the gate wins
            logo = self._commons_asset(res, "logo", file_name, CONF_LOGO)
            if logo:
                res.assets.append(logo)
                self.stats["logo_hosted"] += 1
                break
        if not res.found and website and is_public_http_url(website):
            hp = self._optional(res, "homepage", lambda: homepage_icon(self.http, website, self.cfg.media.homepage_min_icon_px))
            if hp:
                res.assets.append(asset("logo", hp[0], False, "homepage_icon", self.now, attribution=f"Website icon of {website}"[:512],
                                        attributionUrl=website, confidence=CONF_HOMEPAGE))
                self.stats["logo_homepage"] += 1
            else:
                fav = self._optional(res, "favicon", lambda: favicon(self.http, website, self.cfg.media.favicon_min_px))
                if fav:
                    res.assets.append(asset("logo", fav[0], False, "favicon", self.now, attribution="Google favicon service",
                                            attributionUrl=website, confidence=CONF_FAVICON))
                    self.stats["logo_favicon"] += 1
        return res
