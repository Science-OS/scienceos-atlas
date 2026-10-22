"""Validate every published JSON file against the copied Link v1 atlas schema, plus structural cross-checks
(manifest hashes/counts, author and thumbnail references, size budget)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from .config import REPO_ROOT

log = logging.getLogger("atlas.validate")

SCHEMA_DIR = REPO_ROOT / "schemas"
ATLAS_ID = "https://scienceos.app/link/v1/atlas.schema.json"
STATUS_ID = "https://scienceos.app/link/v1/atlas-status.schema.json"


@dataclass
class Report:
    checked: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    site_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, msg: str) -> None:
        if len(self.errors) < 200:
            self.errors.append(msg)


def make_validators(schema_dir: Path = SCHEMA_DIR) -> dict[str, Draft202012Validator]:
    resources = []
    for name in ("atlas.schema.json", "common.schema.json", "status.schema.json"):
        contents = json.loads((schema_dir / name).read_text())
        resources.append((contents["$id"], Resource.from_contents(contents, default_specification=DRAFT202012)))
    registry: Registry = Registry().with_resources(resources)

    def v(ref: str) -> Draft202012Validator:
        return Draft202012Validator({"$ref": ref}, registry=registry)

    return {
        "manifest": v(f"{ATLAS_ID}#/$defs/FeedManifest"),
        "shard": v(f"{ATLAS_ID}#/$defs/FeedShard"),
        "paper": v(f"{ATLAS_ID}#/$defs/AtlasPaperFile"),
        "author": v(f"{ATLAS_ID}#/$defs/AtlasAuthorFile"),
        "status": v(STATUS_ID),
    }


def _check(report: Report, validator: Draft202012Validator, doc: Any, label: str) -> bool:
    errs = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    for e in errs[:3]:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        report.err(f"{label}: {path}: {e.message[:200]}")
    return not errs


def validate_site(site: Path, max_site_mb: int = 450, schema_dir: Path = SCHEMA_DIR) -> Report:
    report = Report()
    vs = make_validators(schema_dir)

    def load(rel: str) -> Any | None:
        try:
            return json.loads((site / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report.err(f"{rel}: unreadable ({exc})")
            return None

    manifest = load("index.json")
    listed: set[str] = set()
    if manifest is not None and _check(report, vs["manifest"], manifest, "index.json"):
        report.checked["index"] = 1
        for ref in manifest["shards"]:
            listed.add(ref["path"])
            f = site / ref["path"]
            if not f.is_file():
                report.err(f"{ref['path']}: listed in index.json but missing")
                continue
            data = f.read_bytes()
            if hashlib.sha256(data).hexdigest() != ref["sha256"] or len(data) != ref["bytes"]:
                report.err(f"{ref['path']}: sha256/bytes differ from index.json")

    # every shard file (dated + latest aliases)
    shard_files = sorted(p for p in (site / "feeds").rglob("*.json")) if (site / "feeds").is_dir() else []
    author_ids_needed: set[str] = set()
    thumbs_needed: set[str] = set()
    for p in shard_files:
        rel = p.relative_to(site).as_posix()
        doc = load(rel)
        if doc is None or not _check(report, vs["shard"], doc, rel):
            continue
        report.checked["shards"] = report.checked.get("shards", 0) + 1
        if rel in listed and manifest:
            ref = next(r for r in manifest["shards"] if r["path"] == rel)
            if ref["count"] != len(doc["items"]) or ref["feed"] != doc["feed"] or ref["date"] != doc["date"]:
                report.err(f"{rel}: feed/date/count disagree with index.json")
        ranks = [i["rank"] for i in doc["items"]]
        if ranks != list(range(1, len(ranks) + 1)):
            report.err(f"{rel}: ranks are not 1..n")
        for item in doc["items"]:
            for a in item["paper"]["authors"]:
                if a.get("authorId"):
                    author_ids_needed.add(a["authorId"])
            for link in item["paper"]["links"]:
                if link.get("label") == "thumbnail":
                    thumbs_needed.add(link["url"].rsplit("/", 1)[-1])
    for rel_dir, key in (("papers", "paper"), ("authors", "author")):
        d = site / rel_dir
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            rel = p.relative_to(site).as_posix()
            doc = load(rel)
            if doc is not None and _check(report, vs[key], doc, rel):
                report.checked[key + "s"] = report.checked.get(key + "s", 0) + 1
    status = load("status.json") if (site / "status.json").exists() else None
    if status is not None and _check(report, vs["status"], status, "status.json"):
        report.checked["status"] = 1

    for aid in author_ids_needed:
        if not aid.startswith("oa:") or not (site / "authors" / f"{aid[3:]}.json").is_file():
            report.err(f"authorId {aid} referenced but authors/{aid[3:]}.json is missing")
    for t in thumbs_needed:
        if not (site / "thumbs" / t).is_file():
            report.err(f"thumbs/{t} referenced but missing")
    latest = site / "feeds" / "latest.json"
    if manifest and manifest["shards"] and not latest.is_file():
        report.err("feeds/latest.json is missing")

    report.site_bytes = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
    if report.site_bytes > max_site_mb * 1024 * 1024:
        report.err(f"site is {report.site_bytes / 1e6:.0f} MB, over the {max_site_mb} MB budget")
    return report
