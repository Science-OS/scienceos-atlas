"""Identity-chain sources, ported from the Mac resolver: Wikidata (CC0), Wikimedia Commons, ROR (CC0), GitHub and Bluesky.
Synchronous and polite (one PoliteClient per source group). Nothing here searches by name: every lookup starts from an id."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from ..config import Config
from ..http import HttpError, PoliteClient

log = logging.getLogger("atlas.media.sources")

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
FILE_PATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
ROR_API = "https://api.ror.org/v2/organizations"
BSKY_PROFILE = "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile"
GITHUB_PNG = "https://github.com/{login}.png"

_GITHUB_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_BSKY_HANDLE = re.compile(r"^(?=.{4,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]([a-z0-9-]{0,61}[a-z0-9])?$", re.I)
_RANK = {"preferred": 0, "normal": 1}


class SourceError(RuntimeError):
    """A source could not answer (timeout, 5xx after retries, unparsable body). Not the same as "no data"."""


@dataclass
class MediaHttp:
    """One polite client per source group; every request of a group shares one throttle."""
    wikimedia: PoliteClient      # wikidata.org, commons.wikimedia.org, upload.wikimedia.org
    ror: PoliteClient
    github: PoliteClient
    bluesky: PoliteClient
    web: PoliteClient            # homepages, their icons, the Google favicon service
    queue: PoliteClient          # Supabase PostgREST: the client work queue
    resolve_dns: bool = True     # SSRF check of homepage hosts; tests with fake hosts turn it off

    @classmethod
    def create(cls, cfg: Config, *, transport: httpx.BaseTransport | None = None, sleep=None, resolve_dns: bool = True) -> "MediaHttp":
        kw: dict[str, Any] = {"transport": transport}
        if sleep is not None:
            kw["sleep"] = sleep
        iv, ua = cfg.media.min_interval_s, cfg.user_agent

        def client(interval: float, timeout: float = 30, retries: int = 2) -> PoliteClient:
            return PoliteClient(ua, interval, max_retries=retries, max_retry_after_s=30, timeout_s=timeout, **kw)

        return cls(wikimedia=client(iv), ror=client(iv), github=client(iv, 15, 1), bluesky=client(iv, 15, 1),
                   web=client(iv, 10, 0), queue=client(iv, 30, 1), resolve_dns=resolve_dns)

    def close(self) -> None:
        for c in (self.wikimedia, self.ror, self.github, self.bluesky, self.web, self.queue):
            c.close()

    @property
    def requests_made(self) -> int:
        return sum(c.requests_made for c in (self.wikimedia, self.ror, self.github, self.bluesky, self.web, self.queue))


def get_json(client: PoliteClient, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
             absent: tuple[int, ...] = (404,)) -> Any | None:
    """Parsed JSON, None when the status says "no such thing" (`absent`). SourceError when the source failed."""
    try:
        resp = client.get(url, params=params, headers=headers)
    except HttpError as exc:
        raise SourceError(str(exc)) from exc
    if resp.status_code in absent:
        return None
    if resp.status_code != 200:
        raise SourceError(f"GET {url} -> HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as exc:
        raise SourceError(f"GET {url}: not JSON") from exc


# ------------------------------------------------------------------ Wikidata


def find_qid(http: MediaHttp, prop: str, value: str) -> str | None:
    """Reverse lookup by an external id (P496 ORCID, P10283 OpenAlex). Exactly one item or nothing: ambiguity attaches nothing."""
    doc = get_json(http.wikimedia, WIKIDATA_API, {
        "action": "query", "list": "search", "srsearch": f"haswbstatement:{prop}={value}", "srlimit": 3, "format": "json"})
    hits = [h.get("title") for h in ((doc or {}).get("query") or {}).get("search") or []]
    qids = [h for h in hits if isinstance(h, str) and re.fullmatch(r"Q\d+", h)]
    return qids[0] if len(qids) == 1 else None


def wikidata_entity(http: MediaHttp, qid: str) -> dict[str, Any] | None:
    doc = get_json(http.wikimedia, WIKIDATA_API, {
        "action": "wbgetentities", "ids": qid, "props": "claims|labels", "languages": "en", "format": "json"})
    ent = ((doc or {}).get("entities") or {}).get(qid)
    return ent if isinstance(ent, dict) and "missing" not in ent else None


def _value(snak: dict[str, Any]) -> Any:
    if snak.get("snaktype") != "value":
        return None
    dv = snak.get("datavalue") or {}
    val = dv.get("value")
    if dv.get("type") == "wikibase-entityid" and isinstance(val, dict):
        return val.get("id")
    return val


def claim_values(entity: dict[str, Any], prop: str) -> list[Any]:
    """Non-deprecated values, preferred rank first; claims with an end time (P582) go last (a retired logo)."""
    claims = [c for c in (entity.get("claims") or {}).get(prop, []) if c.get("rank") != "deprecated"]
    claims.sort(key=lambda c: (1 if (c.get("qualifiers") or {}).get("P582") else 0, _RANK.get(c.get("rank", "normal"), 1)))
    return [v for v in (_value(c.get("mainsnak") or {}) for c in claims) if v is not None]


def person_facts(entity: dict[str, Any]) -> dict[str, Any]:
    first = lambda prop: next((v for v in claim_values(entity, prop) if isinstance(v, str)), None)  # noqa: E731
    return {
        "qid": entity.get("id"), "label": ((entity.get("labels") or {}).get("en") or {}).get("value"),
        "image": first("P18"), "github": first("P2037"), "bluesky": first("P12361"), "orcid": first("P496"),
        "openalex": [v for v in claim_values(entity, "P10283") if isinstance(v, str)],
    }


def org_facts(entity: dict[str, Any]) -> dict[str, Any]:
    strs = lambda prop: [v for v in claim_values(entity, prop) if isinstance(v, str)]  # noqa: E731
    return {
        "qid": entity.get("id"), "label": ((entity.get("labels") or {}).get("en") or {}).get("value"),
        "logos": strs("P154"), "website": (strs("P856") or [None])[0], "ror": (strs("P6782") or [None])[0],
    }


# ------------------------------------------------------------------ Commons


def filepath_url(file_name: str, width: int | None = None) -> str:
    """Special:FilePath redirects to the file; with ?width= it serves a scaled render (a PNG for SVG sources)."""
    url = FILE_PATH + quote(file_name.replace(" ", "_"))
    return f"{url}?width={width}" if width else url


def _strip_html(text: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def parse_imageinfo(payload: dict[str, Any]) -> dict[str, Any] | None:
    for page in ((payload.get("query") or {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        meta = info.get("extmetadata") or {}

        def mv(k: str) -> str | None:
            v = (meta.get(k) or {}).get("value")
            return v if isinstance(v, str) else None

        return {
            "license": mv("LicenseShortName"), "licenseUrl": mv("LicenseUrl"),
            "artist": _strip_html(mv("Artist")) or None, "credit": _strip_html(mv("Credit")) or None,
            "nonFree": (mv("NonFree") or "").lower() == "true", "descriptionUrl": info.get("descriptionurl"),
        }
    return None


def license_info(http: MediaHttp, file_name: str) -> dict[str, Any] | None:
    doc = get_json(http.wikimedia, COMMONS_API, {
        "action": "query", "titles": "File:" + file_name.replace("_", " "), "prop": "imageinfo", "iiprop": "extmetadata|url", "format": "json",
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|NonFree"})
    return parse_imageinfo(doc) if isinstance(doc, dict) else None


# Re-hosting is allowed for CC0, public domain (PD-*, incl. PD-textlogo), CC BY and CC BY-SA only (contract rule 2).
_REHOSTABLE = re.compile(r"^(?:cc0\b|cc[ -]by(?:[ -]sa)?\b|public domain|pd\b)", re.IGNORECASE)
_NC_ND = re.compile(r"\b(?:nc|nd)\b", re.IGNORECASE)
_NEEDS_NO_CREDIT = re.compile(r"^(?:cc0\b|public domain|pd\b)", re.IGNORECASE)


def license_verdict(info: dict[str, Any] | None) -> str:
    """free (may be hosted) | nonfree (known licence, not on the allow-list) | missing (no licence information)."""
    short = ((info or {}).get("license") or "").strip()
    if not short:
        return "missing"
    if (info or {}).get("nonFree") or _NC_ND.search(short) or not _REHOSTABLE.match(short):
        return "nonfree"
    return "free"


def credit_for(info: dict[str, Any]) -> str | None:
    """Artist (or credit line) for the attribution. PD and CC0 need none, but we still say where it came from."""
    who = info.get("artist") or info.get("credit")
    if who:
        return who[:512]
    return "Wikimedia Commons" if _NEEDS_NO_CREDIT.match((info.get("license") or "").strip()) else None


@dataclass
class ImageFetch:
    status: str                     # ok | not_modified | missing
    data: bytes = b""
    etag: str | None = None
    content_type: str = ""
    source_url: str = field(default="")


def fetch_image(client: PoliteClient, url: str, max_bytes: int, etag: str | None = None) -> ImageFetch:
    headers = {"If-None-Match": etag} if etag else None
    try:
        resp = client.get(url, headers=headers, max_bytes=max_bytes)
    except HttpError as exc:
        raise SourceError(str(exc)) from exc
    if resp.status_code == 304:
        return ImageFetch("not_modified", etag=etag, source_url=url)
    if resp.status_code in (404, 410):
        return ImageFetch("missing", source_url=url)
    if resp.status_code != 200:
        raise SourceError(f"GET {url} -> HTTP {resp.status_code}")
    return ImageFetch("ok", resp.content, resp.headers.get("etag"), resp.headers.get("content-type", ""), url)


# ------------------------------------------------------------------ ROR


def parse_ror(raw: dict[str, Any]) -> dict[str, Any]:
    names = raw.get("names") or []

    def pick(kind: str) -> str | None:
        return next((n.get("value") for n in names if kind in (n.get("types") or [])), None)

    wikidata = None
    for ext in raw.get("external_ids") or []:
        if ext.get("type") == "wikidata":
            wikidata = ext.get("preferred") or (ext.get("all") or [None])[0]
    return {
        "ror": (raw.get("id") or "").rsplit("/", 1)[-1],
        "name": pick("ror_display") or pick("label") or (names[0].get("value") if names else None),
        "website": next((lk.get("value") for lk in raw.get("links") or [] if lk.get("type") == "website"), None),
        "wikidata": wikidata if isinstance(wikidata, str) else None,
        "status": raw.get("status"),
    }


def ror_record(http: MediaHttp, ror: str) -> dict[str, Any] | None:
    doc = get_json(http.ror, f"{ROR_API}/{ror}")
    return parse_ror(doc) if isinstance(doc, dict) else None


# ------------------------------------------------------------------ GitHub, Bluesky (link-only avatars)


def github_avatar(http: MediaHttp, login: str, px: int = 256) -> str | None:
    """Avatar URL when the account exists (HEAD, redirect not followed). The bytes are never fetched or hosted."""
    if not _GITHUB_LOGIN.match(login or ""):
        return None
    url = GITHUB_PNG.format(login=login)
    try:
        resp = http.github.head(f"{url}?size={px}", follow_redirects=False)
    except HttpError as exc:
        raise SourceError(str(exc)) from exc
    return f"{url}?size={px}" if resp.status_code in (200, 301, 302, 303, 307, 308) else None


def looks_like_bluesky_handle(value: str) -> bool:
    return bool(_BSKY_HANDLE.match(value.strip().lstrip("@")))


def bluesky_avatar(http: MediaHttp, handle: str) -> dict[str, str] | None:
    """{'handle', 'avatar'} from the public profile endpoint (no auth), or None. The avatar must be on Bluesky's CDN."""
    handle = handle.strip().lstrip("@").lower()
    if not looks_like_bluesky_handle(handle):
        return None
    doc = get_json(http.bluesky, BSKY_PROFILE, {"actor": handle}, absent=(400, 404))  # unknown actor is a 400
    if not isinstance(doc, dict) or not doc.get("did") or not isinstance(doc.get("avatar"), str):
        return None
    if not doc["avatar"].startswith("https://cdn.bsky.app/"):
        return None
    return {"handle": doc.get("handle") or handle, "avatar": doc["avatar"]}
