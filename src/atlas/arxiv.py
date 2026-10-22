"""arXiv ingestion: OAI-PMH harvesting (metadata), export-API verification of v1 dates, category/ID filters.

Why two sources: in the OAI `arXiv` format `created` is the date of the LATEST version, so revised old
papers look new. The export API's `published` is the v1 timestamp, which is the only reliable "new" signal.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from .http import PoliteClient

log = logging.getLogger("atlas.arxiv")

NS_OAI = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
NS_ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

NEW_STYLE_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")
_WS = re.compile(r"\s+")


class OaiError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"OAI-PMH error {code}: {message}")
        self.code = code


@dataclass
class OaiRecord:
    arxiv_id: str
    datestamp: str  # YYYY-MM-DD, last modification (announcement day for new papers)
    set_specs: list[str] = field(default_factory=list)
    created: str | None = None
    updated: str | None = None
    title: str = ""
    abstract: str = ""
    authors: list[dict[str, Any]] = field(default_factory=list)  # {"name": str, "affiliation": str|None}
    categories: list[str] = field(default_factory=list)
    license: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comments: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "OaiRecord":
        return cls(**d)


@dataclass
class ApiEntry:
    arxiv_id: str
    published: str  # RFC 3339 Z (v1 submission time)
    updated: str
    version: int  # latest version number
    primary_category: str | None


def clean(text: str | None) -> str:
    return _WS.sub(" ", text or "").strip()


def _text(el: ET.Element | None, path: str) -> str | None:
    if el is None:
        return None
    child = el.find(path, NS_OAI)
    if child is None or child.text is None:
        return None
    v = clean(child.text)
    return v or None


def _author_name(author: ET.Element) -> tuple[str, str | None]:
    keyname = _text(author, "a:keyname") or ""
    forenames = _text(author, "a:forenames") or ""
    suffix = _text(author, "a:suffix") or ""
    name = clean(" ".join(p for p in (forenames, keyname, suffix) if p))
    affs = [clean(a.text) for a in author.findall("a:affiliation", NS_OAI) if a.text and clean(a.text)]
    return name, ("; ".join(affs) if affs else None)


def parse_list_records(xml_bytes: bytes) -> tuple[list[OaiRecord], str | None]:
    """Parse one ListRecords page. Returns (records, resumptionToken or None).
    `noRecordsMatch` yields ([], None); other OAI errors raise OaiError."""
    root = ET.fromstring(xml_bytes)
    err = root.find("o:error", NS_OAI)
    if err is not None:
        code = err.get("code", "unknown")
        if code == "noRecordsMatch":
            return [], None
        raise OaiError(code, clean(err.text))
    records: list[OaiRecord] = []
    for rec in root.iterfind(".//o:ListRecords/o:record", NS_OAI):
        header = rec.find("o:header", NS_OAI)
        if header is None or header.get("status") == "deleted":
            continue
        meta = rec.find("o:metadata/a:arXiv", NS_OAI)
        if meta is None:
            continue
        arxiv_id = _text(meta, "a:id")
        if not arxiv_id:
            continue
        authors = []
        for au in meta.iterfind("a:authors/a:author", NS_OAI):
            name, aff = _author_name(au)
            if name:
                authors.append({"name": name, "affiliation": aff})
        doi = _text(meta, "a:doi")
        records.append(
            OaiRecord(
                arxiv_id=arxiv_id,
                datestamp=_text(header, "o:datestamp") or "",
                set_specs=[clean(s.text) for s in header.findall("o:setSpec", NS_OAI) if s.text],
                created=_text(meta, "a:created"),
                updated=_text(meta, "a:updated"),
                title=_text(meta, "a:title") or "",
                abstract=_text(meta, "a:abstract") or "",
                authors=authors,
                categories=(_text(meta, "a:categories") or "").split(),
                license=_text(meta, "a:license"),
                doi=doi.split()[0] if doi else None,
                journal_ref=_text(meta, "a:journal-ref"),
                comments=_text(meta, "a:comments"),
            )
        )
    token_el = root.find(".//o:ListRecords/o:resumptionToken", NS_OAI)
    token = clean(token_el.text) if token_el is not None and token_el.text else None
    return records, token or None


def harvest(
    client: PoliteClient,
    oai_base: str,
    set_spec: str,
    from_date: str,
    *,
    until_date: str | None = None,
    stop: Callable[[], bool] | None = None,
) -> Iterator[OaiRecord]:
    """ListRecords for one set, honouring resumptionToken. 503/Retry-After is handled by the client."""
    params: dict[str, str] = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_spec, "from": from_date}
    if until_date:
        params["until"] = until_date
    page = 0
    while True:
        resp = client.get(oai_base, params=params)
        resp.raise_for_status()
        records, token = parse_list_records(resp.content)
        page += 1
        log.info("OAI set=%s page=%d records=%d token=%s", set_spec, page, len(records), "yes" if token else "no")
        yield from records
        if not token or (stop and stop()):
            return
        params = {"verb": "ListRecords", "resumptionToken": token}


# ---------------------------------------------------------------- filters


def is_recent_id(arxiv_id: str, datestamp: str) -> bool:
    """New-style id whose YYMM is the datestamp's month or the previous one (IDs follow announcement month)."""
    m = NEW_STYLE_ID.match(arxiv_id)
    if not m:
        return False
    try:
        ds = date.fromisoformat(datestamp)
    except ValueError:
        return False
    id_months = (2000 + int(m.group(1))) * 12 + int(m.group(2))
    ds_months = ds.year * 12 + ds.month
    return ds_months - 1 <= id_months <= ds_months


def matches_categories(categories: list[str], patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(c, p) for c in categories for p in patterns)


# ---------------------------------------------------------------- export API verification

_ENTRY_ID = re.compile(r"/abs/(?P<id>[^/]+?)v(?P<v>\d+)$")


def parse_api_feed(xml_bytes: bytes) -> dict[str, ApiEntry]:
    root = ET.fromstring(xml_bytes)
    out: dict[str, ApiEntry] = {}
    for entry in root.iterfind("atom:entry", NS_ATOM):
        id_el = entry.find("atom:id", NS_ATOM)
        m = _ENTRY_ID.search(id_el.text.strip()) if id_el is not None and id_el.text else None
        pub = entry.find("atom:published", NS_ATOM)
        upd = entry.find("atom:updated", NS_ATOM)
        if not m or pub is None or not pub.text:
            continue  # error entries (".../api/errors#...") and anything malformed
        prim = entry.find("arxiv:primary_category", NS_ATOM)
        out[m.group("id")] = ApiEntry(
            arxiv_id=m.group("id"),
            published=pub.text.strip(),
            updated=(upd.text.strip() if upd is not None and upd.text else pub.text.strip()),
            version=int(m.group("v")),
            primary_category=prim.get("term") if prim is not None else None,
        )
    return out


def verify_entries(client: PoliteClient, api_base: str, ids: list[str], batch: int = 100) -> dict[str, ApiEntry]:
    """Fetch v1 `published` timestamps for many ids (one throttled request per `batch` ids)."""
    result: dict[str, ApiEntry] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        resp = client.get(api_base, params={"id_list": ",".join(chunk), "max_results": str(len(chunk))})
        resp.raise_for_status()
        result.update(parse_api_feed(resp.content))
        log.info("API verify %d/%d ids", min(i + batch, len(ids)), len(ids))
    return result


def is_new_paper(entry: ApiEntry, datestamp: str, max_age_days: int) -> bool:
    """v1 was submitted within `max_age_days` before the OAI datestamp (announcement day)."""
    try:
        published = datetime.fromisoformat(entry.published.replace("Z", "+00:00")).astimezone(timezone.utc).date()
        ds = date.fromisoformat(datestamp)
    except ValueError:
        return False
    return -1 <= (ds - published).days <= max_age_days
