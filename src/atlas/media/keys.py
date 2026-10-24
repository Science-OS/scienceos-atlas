"""Filename-safe entity keys (contract v1). Everything that reaches the filesystem or the catalogue passes through here,
so a key from an untrusted queue row can never contain a path separator."""

from __future__ import annotations

import re

KINDS = ("author", "institution", "paper")
ORCID = r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]"

_AUTHOR = re.compile(rf"^(?:orcid-{ORCID}|wikidata-Q\d{{1,12}}|openalex-A\d{{1,14}})$")
_INSTITUTION = re.compile(r"^(?:ror-[0-9a-z]{6,12}|wikidata-Q\d{1,12})$")
_PAPER = re.compile(r"^arxiv-(?:\d{4}\.\d{4,5}|[a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?_\d{7})$")
_PATTERNS = {"author": _AUTHOR, "institution": _INSTITUTION, "paper": _PAPER}

_ORCID_IN = re.compile(rf"(?:^|[/:])({ORCID})$", re.IGNORECASE)
_QID_IN = re.compile(r"(?:^|[/:])(Q\d{1,12})$", re.IGNORECASE)
_OA_IN = re.compile(r"(?:^|[/:])(A\d{1,14})$", re.IGNORECASE)
_ROR_IN = re.compile(r"(?:^|[/:])([0-9a-z]{6,12})$", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$")

AUTHOR_ID_ORDER = ("orcid", "wikidata", "openalex")
INSTITUTION_ID_ORDER = ("ror", "wikidata")


def orcid_id(value: str | None) -> str | None:
    m = _ORCID_IN.search((value or "").strip().rstrip("/"))
    return m.group(1).upper() if m else None


def wikidata_id(value: str | None) -> str | None:
    m = _QID_IN.search((value or "").strip().rstrip("/"))
    return m.group(1).upper() if m else None


def openalex_id(value: str | None) -> str | None:
    m = _OA_IN.search((value or "").strip().rstrip("/"))
    return m.group(1).upper() if m else None


def ror_id(value: str | None) -> str | None:
    m = _ROR_IN.search((value or "").strip().rstrip("/"))
    return m.group(1).lower() if m else None


def author_keys(ids: dict[str, str | None]) -> list[str]:
    """Keys of one author, best first (orcid, wikidata, openalex). Unusable ids are dropped."""
    out = []
    if orcid_id(ids.get("orcid")):
        out.append(f"orcid-{orcid_id(ids['orcid'])}")
    if wikidata_id(ids.get("wikidata")):
        out.append(f"wikidata-{wikidata_id(ids['wikidata'])}")
    if openalex_id(ids.get("openalex")):
        out.append(f"openalex-{openalex_id(ids['openalex'])}")
    return out


def institution_keys(ids: dict[str, str | None]) -> list[str]:
    out = []
    if ror_id(ids.get("ror")):
        out.append(f"ror-{ror_id(ids['ror'])}")
    if wikidata_id(ids.get("wikidata")):
        out.append(f"wikidata-{wikidata_id(ids['wikidata'])}")
    return out


def entity_keys(kind: str, ids: dict[str, str | None]) -> list[str]:
    return author_keys(ids) if kind == "author" else institution_keys(ids)


def paper_key(arxiv_id: str) -> str:
    """2610.30108v2 -> arxiv-2610.30108; hep-th/9901001 -> arxiv-hep-th_9901001."""
    bare = re.sub(r"^arxiv:", "", arxiv_id.strip(), flags=re.IGNORECASE)
    return "arxiv-" + _ARXIV_VERSION.sub("", bare).replace("/", "_")


def valid_key(kind: str, key: str) -> bool:
    pat = _PATTERNS.get(kind)
    return bool(pat and isinstance(key, str) and pat.match(key))


def ids_from_keys(kind: str, keys: list[str]) -> dict[str, str]:
    """{'orcid': '0000-...', 'wikidata': 'Q1'} from contract keys (invalid ones are ignored)."""
    out: dict[str, str] = {}
    for key in keys:
        if not valid_key(kind, key):
            continue
        scheme, _, value = key.partition("-")
        out[scheme] = value
    return out
