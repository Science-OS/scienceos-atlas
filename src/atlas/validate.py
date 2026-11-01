"""Validate every published JSON file against the copied Link v1 atlas schema, plus structural cross-checks
(manifest hashes/counts, author and thumbnail references, size budget), the published media store (every catalogue
entry against schemas/media.schema.json, hosted files present and byte-exact, media size budget) and the audio brief
(every day manifest against schemas/audio.schema.json, clips present with the manifest's size and SHA-256, word timings
consistent with the text, audio size budget)."""

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

from .audio.storage import clip_id, manifest_clips, sha256_hex
from .audio.text import split_words, utf16_len
from .config import REPO_ROOT
from .media.keys import valid_key
from .media.store import MediaStore

log = logging.getLogger("atlas.validate")

SCHEMA_DIR = REPO_ROOT / "schemas"
ATLAS_ID = "https://scienceos.app/link/v1/atlas.schema.json"
STATUS_ID = "https://scienceos.app/link/v1/atlas-status.schema.json"
MEDIA_ID = "https://scienceos.app/link/v1/atlas-media.schema.json"
AUDIO_ID = "https://scienceos.app/link/v1/atlas-audio.schema.json"


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
    for name in ("atlas.schema.json", "common.schema.json", "status.schema.json", "media.schema.json", "audio.schema.json"):
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
        "media_entry": v(f"{MEDIA_ID}#/$defs/CatalogueEntry"),
        "media_manifest": v(f"{MEDIA_ID}#/$defs/Manifest"),
        "audio_day": v(f"{AUDIO_ID}#/$defs/Day"),
    }


def _check(report: Report, validator: Draft202012Validator, doc: Any, label: str) -> bool:
    errs = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    for e in errs[:3]:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        report.err(f"{label}: {path}: {e.message[:200]}")
    return not errs


def _media_path(url: str) -> str | None:
    """'media/b/ab/ab...cd.webp' for any URL that points into the media store, else None."""
    return "media/" + url.split("/media/", 1)[1] if "/media/b/" in url else None


def _check_media(site: Path, vs: dict[str, Draft202012Validator], report: Report, media_max_mb: int) -> None:
    root = site / "media"
    entries = 0
    hashed: set[str] = set()   # alias files share blobs: hash each file once
    for kind in ("author", "institution", "paper"):
        d = root / "c" / kind
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            rel = p.relative_to(site).as_posix()
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                report.err(f"{rel}: unreadable ({exc})")
                continue
            if not _check(report, vs["media_entry"], doc, rel):
                continue
            entries += 1
            if doc["kind"] != kind or doc["key"] != p.stem or not valid_key(kind, p.stem):
                report.err(f"{rel}: kind/key do not match the file path")
            for a in doc["assets"]:
                if not a["hosted"]:
                    continue
                want = f"media/b/{a['sha256'][:2]}/{a['sha256']}.{'webp' if a['mime'] == 'image/webp' else 'jpg'}"
                if _media_path(a["url"]) != want:
                    report.err(f"{rel}: hosted url does not match its sha256/mime")
                f = site / want
                if not f.is_file():
                    report.err(f"{rel}: hosted file {want} is missing")
                elif want not in hashed:
                    hashed.add(want)
                    if hashlib.sha256(f.read_bytes()).hexdigest() != a["sha256"]:
                        report.err(f"{want}: bytes do not match the sha256 in the file name")
    report.checked["media_entries"] = entries
    mf = root / "manifest.json"
    if not mf.is_file():
        report.err("media/manifest.json is missing")
    else:
        try:
            doc = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report.err(f"media/manifest.json: unreadable ({exc})")
            doc = None
        if doc is not None and _check(report, vs["media_manifest"], doc, "media/manifest.json"):
            real = MediaStore(root, "https://x.invalid").compute_manifest(doc["generatedAt"])
            if doc["counts"] != real["counts"] or doc["hostedBytes"] != real["hostedBytes"]:
                report.err("media/manifest.json: counts/hostedBytes do not match the files")
    media_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if media_bytes > media_max_mb * 1024 * 1024:
        report.err(f"media/ is {media_bytes / 1e6:.0f} MB, over the {media_max_mb} MB media budget")


def _timeline_errors(rows: list[list[int]], duration_ms: int, label: str) -> list[str]:
    errs = []
    ps = pe = 0
    for i, (s, e, *_rest) in enumerate(rows):
        if s > e or e > duration_ms:
            errs.append(f"{label}[{i}]: {s}-{e} ms is outside 0-{duration_ms} ms")
            break
        if s < ps or e < pe:
            errs.append(f"{label}[{i}]: not monotonic")
            break
        ps, pe = s, e
    return errs


def audio_day_errors(doc: Any, validator: Draft202012Validator, clips_dir: Path, *, label: str, expect_date: str | None = None,
                     hashed: dict[str, str | None] | None = None) -> list[str]:
    """Everything that must hold for one day manifest to be published: schema; every referenced clip file exists with the manifest's byte count and
    SHA-256; clip ids and urls are what the contract says; every word of the text has one monotonic timing and its exact UTF-16 span.
    `hashed` caches the verdict per clip id across days (a clip is checked once)."""
    errs = [f"{label}: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message[:160]}"
            for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))[:3]]
    if errs:
        return errs
    if expect_date and doc["date"] != expect_date:
        errs.append(f"{label}: date {doc['date']} does not match the file name")
    hashed = {} if hashed is None else hashed
    eng = doc["engine"]
    for clip in manifest_clips(doc):
        cid, tag = clip["id"], f"{label} clip {clip['id'][:8]}"
        if cid != clip_id(eng["version"], eng["voice"], clip["text"]):
            errs.append(f"{tag}: id is not sha256(version|voice|text)")
        if clip["url"] != f"audio/clips/{cid}.m4a":
            errs.append(f"{tag}: url does not match the id")
        if cid not in hashed:
            f = clips_dir / f"{cid}.m4a"
            if not f.is_file():
                hashed[cid] = "file is missing"
            else:
                data = f.read_bytes()
                hashed[cid] = None if len(data) == clip["bytes"] and sha256_hex(data) == clip["sha256"] else "bytes/sha256 differ from the manifest"
        if hashed[cid]:
            errs.append(f"{tag}: {hashed[cid]}")
        words = split_words(clip["text"])
        if len(clip["words"]) != len(words):
            errs.append(f"{tag}: {len(clip['words'])} timings for {len(words)} words")
        elif any(r[2] != w.u16_start or r[3] != w.u16_end for r, w in zip(clip["words"], words)):
            errs.append(f"{tag}: a word's UTF-16 span does not match the text")
        errs += [f"{tag}: {m}" for m in _timeline_errors(clip["words"], clip["durationMs"], "words")]
        errs += [f"{tag}: {m}" for m in _timeline_errors(clip["sentences"], clip["durationMs"], "sentences")]
        n16 = utf16_len(clip["text"])
        if any(r[2] > r[3] or r[3] > n16 for r in clip["sentences"]):
            errs.append(f"{tag}: a sentence span is outside the text")
    return errs


def _check_audio(site: Path, vs: dict[str, Draft202012Validator], report: Report, audio_max_mb: int) -> None:
    root = site / "audio"
    referenced: set[str] = set()
    hashed: dict[str, str | None] = {}
    days = 0
    for p in sorted(root.glob("*.json")):
        rel = p.relative_to(site).as_posix()
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report.err(f"{rel}: unreadable ({exc})")
            continue
        errs = audio_day_errors(doc, vs["audio_day"], root / "clips", label=rel, expect_date=p.stem, hashed=hashed)
        for e in errs[:5]:
            report.err(e)
        if not errs:
            days += 1
            referenced.update(c["id"] for c in manifest_clips(doc))
    for f in sorted((root / "clips").glob("*")) if (root / "clips").is_dir() else []:
        if f.stem not in referenced:
            report.err(f"audio/clips/{f.name}: not referenced by any day manifest")
    report.checked["audio_days"] = days
    report.checked["audio_clips"] = len(referenced)
    audio_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if audio_bytes > audio_max_mb * 1024 * 1024:
        report.err(f"audio/ is {audio_bytes / 1e6:.0f} MB, over the {audio_max_mb} MB audio budget")


def validate_site(site: Path, max_site_mb: int = 450, schema_dir: Path = SCHEMA_DIR, media_max_mb: int = 250, audio_max_mb: int = 120) -> Report:
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
    media_needed: set[str] = set()   # files under media/ that published JSON points at
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
                    if _media_path(link["url"]):
                        media_needed.add(_media_path(link["url"]))
                    else:
                        thumbs_needed.add(link["url"].rsplit("/", 1)[-1])
    for rel_dir, key in (("papers", "paper"), ("authors", "author")):
        d = site / rel_dir
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            rel = p.relative_to(site).as_posix()
            doc = load(rel)
            if doc is not None and _check(report, vs[key], doc, rel):
                report.checked[key + "s"] = report.checked.get(key + "s", 0) + 1
                if key == "author":
                    a = doc["author"]
                    urls = [(a.get("photo") or {}).get("url"), *(i.get("logoUrl") for i in a["affiliations"]),
                            *((i.get("logo") or {}).get("url") for i in doc["institutions"])]
                    media_needed.update(m for m in (_media_path(u) for u in urls if u) if m)
    status = load("status.json") if (site / "status.json").exists() else None
    if status is not None and _check(report, vs["status"], status, "status.json"):
        report.checked["status"] = 1

    for aid in author_ids_needed:
        if not aid.startswith("oa:") or not (site / "authors" / f"{aid[3:]}.json").is_file():
            report.err(f"authorId {aid} referenced but authors/{aid[3:]}.json is missing")
    for t in thumbs_needed:
        if not (site / "thumbs" / t).is_file():
            report.err(f"thumbs/{t} referenced but missing")
    for m in sorted(media_needed):
        if not (site / m).is_file():
            report.err(f"{m} is referenced by published JSON but missing")
    latest = site / "feeds" / "latest.json"
    if manifest and manifest["shards"] and not latest.is_file():
        report.err("feeds/latest.json is missing")

    if (site / "media").is_dir():
        _check_media(site, vs, report, media_max_mb)
    if (site / "audio").is_dir():
        _check_audio(site, vs, report, audio_max_mb)
    report.site_bytes = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
    if report.site_bytes > max_site_mb * 1024 * 1024:
        report.err(f"site is {report.site_bytes / 1e6:.0f} MB, over the {max_site_mb} MB budget")
    return report
