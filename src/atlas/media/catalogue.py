"""Catalogue entries: ordered assets per entity, merged so that nothing already known is ever dropped."""

from __future__ import annotations

from typing import Any

VERSION = 1
ROLES = ("photo", "logo", "thumbnail")
SOURCES = ("wikidata_commons", "github", "bluesky", "homepage_icon", "favicon", "arxiv_pdf")
# lower = better; hosted assets always sort before link-only ones of the same role
SOURCE_RANK = {"wikidata_commons": 0, "arxiv_pdf": 0, "github": 1, "homepage_icon": 1, "bluesky": 2, "favicon": 2}
MAX_HOSTED_PER_ROLE = 3  # older hosted files of a role stay listed behind the newest ones
ID_ORDER = ("arxiv", "orcid", "wikidata", "openalex", "ror", "github", "bluesky")
FIELD_ORDER = ("role", "url", "hosted", "sha256", "mime", "width", "height", "source", "sourceUrl", "license", "licenseUrl",
               "attribution", "attributionUrl", "confidence", "fetchedAt")
VOLATILE = {"fetchedAt"}


def asset(role: str, url: str, hosted: bool, source: str, fetched_at: str, **fields: Any) -> dict[str, Any]:
    """One catalogue asset with the fields in contract order; unset (None) fields are omitted."""
    row: dict[str, Any] = {"role": role, "url": url, "hosted": hosted, "source": source, "fetchedAt": fetched_at, **fields}
    return {k: row[k] for k in FIELD_ORDER if row.get(k) is not None}


def sort_key(a: dict[str, Any]) -> tuple[int, int, int, float]:
    role = ROLES.index(a["role"]) if a.get("role") in ROLES else len(ROLES)
    return (role, 0 if a.get("hosted") else 1, SOURCE_RANK.get(a.get("source", ""), 9), -float(a.get("confidence") or 0))


def order_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(assets, key=sort_key)  # stable: equal rank keeps the incoming order


def _identity(a: dict[str, Any]) -> tuple[str, str]:
    return (a["role"], a["sha256"] if a.get("hosted") and a.get("sha256") else a["url"])


def merge_assets(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A fresh resolution replaces the link-only assets of the roles it produced and puts its hosted files first.
    Hosted files found earlier are never dropped (bytes are immutable and licences irrevocable), and roles the new
    resolution found nothing for keep what they had."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    new_roles = {a["role"] for a in new}
    for a in new:
        if _identity(a) not in seen:
            seen.add(_identity(a))
            merged.append(a)
    for a in old:
        if _identity(a) in seen:
            continue
        if a["role"] in new_roles and not a.get("hosted"):
            continue  # superseded link
        seen.add(_identity(a))
        merged.append(a)
    ordered = order_assets(merged)
    kept: list[dict[str, Any]] = []
    hosted_count: dict[str, int] = {}
    for a in ordered:
        if a.get("hosted"):
            hosted_count[a["role"]] = hosted_count.get(a["role"], 0) + 1
            if hosted_count[a["role"]] > MAX_HOSTED_PER_ROLE:
                continue
        kept.append(a)
    return kept


def merge_ids(old: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    merged = {**old, **{k: v for k, v in new.items() if v}}
    return {k: merged[k] for k in ID_ORDER if merged.get(k)} | {k: v for k, v in merged.items() if k not in ID_ORDER}


def entry(kind: str, key: str, ids: dict[str, str], assets: list[dict[str, Any]], updated_at: str) -> dict[str, Any]:
    return {"version": VERSION, "kind": kind, "key": key, "ids": ids, "updatedAt": updated_at, "assets": order_assets(assets)}


def stable_view(e: dict[str, Any]) -> dict[str, Any]:
    """The entry without timestamps that move on every refresh: equal views mean the file need not be rewritten."""
    return {
        "version": e.get("version"), "kind": e.get("kind"), "key": e.get("key"), "ids": e.get("ids"),
        "assets": [{k: v for k, v in a.items() if k not in VOLATILE} for a in e.get("assets", [])],
    }


def first_asset(e: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    """Clients use the first asset of the role they need (the list is ordered best first)."""
    return next((a for a in (e or {}).get("assets", []) if a.get("role") == role), None)


def as_photo(a: dict[str, Any]) -> dict[str, Any]:
    """Catalogue asset -> Link protocol `$defs.Photo` (url, source, license, attribution, attributionUrl, confidence, fetchedAt)."""
    return {k: a[k] for k in ("url", "source", "license", "attribution", "attributionUrl", "confidence", "fetchedAt") if a.get(k) is not None}
