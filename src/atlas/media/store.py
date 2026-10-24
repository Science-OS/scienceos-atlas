"""The content-addressed store on disk: bytes under b/, catalogue under c/, manifest.json. Append-only by design:
bytes are never rewritten, nothing is ever deleted, catalogue files only change when their content does."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import catalogue as cat
from .keys import KINDS, entity_keys, valid_key

log = logging.getLogger("atlas.media.store")

BLOB_EXTS = ("webp", "jpg")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)   # mkstemp creates 0600
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def dump(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


class MediaStore:
    def __init__(self, root: Path | str, base_url: str):
        self.root = Path(root)
        self.base_url = base_url.rstrip("/")
        self.media_url = f"{self.base_url}/media"
        self._cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    # -- bytes ------------------------------------------------------------------
    def blob_path(self, sha256: str, ext: str) -> Path:
        return self.root / "b" / sha256[:2] / f"{sha256}.{ext}"

    def blob_url(self, sha256: str, ext: str) -> str:
        return f"{self.media_url}/b/{sha256[:2]}/{sha256}.{ext}"

    def put_bytes(self, data: bytes, ext: str) -> tuple[str, bool]:
        """Store bytes under their SHA-256. Returns (sha256, created). An existing file is never touched."""
        if ext not in BLOB_EXTS:
            raise ValueError(f"unsupported blob extension {ext!r}")
        sha = hashlib.sha256(data).hexdigest()
        path = self.blob_path(sha, ext)
        if path.exists():
            return sha, False
        _atomic_write(path, data)
        return sha, True

    def read_blob(self, sha256: str, ext: str) -> bytes | None:
        try:
            return self.blob_path(sha256, ext).read_bytes()
        except OSError:
            return None

    # -- catalogue ----------------------------------------------------------------
    def entry_path(self, kind: str, key: str) -> Path:
        if kind not in KINDS or not valid_key(kind, key):
            raise ValueError(f"invalid {kind} key {key!r}")
        return self.root / "c" / kind / f"{key}.json"

    def read_entry(self, kind: str, key: str) -> dict[str, Any] | None:
        if (kind, key) in self._cache:
            return self._cache[(kind, key)]
        try:
            doc = json.loads(self.entry_path(kind, key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        self._cache[(kind, key)] = doc if isinstance(doc, dict) else None
        return self._cache[(kind, key)]

    def lookup(self, kind: str, keys: list[str]) -> dict[str, Any] | None:
        """The first existing entry among an entity's keys (best key first)."""
        for key in keys:
            e = self.read_entry(kind, key)
            if e:
                return e
        return None

    def write_entry(self, kind: str, keys: list[str], ids: dict[str, str], assets: list[dict[str, Any]], now: str) -> dict[str, str]:
        """One file per key, identical content apart from `key`. Merges with whatever is already there.
        Returns {key: created | updated | unchanged}."""
        keys = [k for k in dict.fromkeys(keys) if valid_key(kind, k)]
        existing = {k: self.read_entry(kind, k) for k in keys}
        old_assets: list[dict[str, Any]] = []
        old_ids: dict[str, str] = {}
        for e in existing.values():
            if e:
                old_assets = cat.merge_assets(old_assets, e.get("assets", []))
                old_ids = cat.merge_ids(old_ids, e.get("ids", {}))
        merged_assets = cat.merge_assets(old_assets, assets)
        merged_ids = cat.merge_ids(old_ids, ids)
        out: dict[str, str] = {}
        for key in keys:
            prev = existing[key]
            new = cat.entry(kind, key, merged_ids, merged_assets, now)
            if prev and cat.stable_view(prev) == cat.stable_view(new):
                out[key] = "unchanged"
                continue
            _atomic_write(self.entry_path(kind, key), dump(new))
            self._cache[(kind, key)] = new
            out[key] = "updated" if prev else "created"
        return out

    def iter_entries(self, kind: str | None = None) -> Iterator[tuple[str, str, dict[str, Any]]]:
        for k in ([kind] if kind else KINDS):
            d = self.root / "c" / k
            for p in sorted(d.glob("*.json")) if d.is_dir() else []:
                try:
                    doc = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    log.warning("unreadable catalogue file %s", p)
                    continue
                if isinstance(doc, dict):
                    yield k, p.stem, doc

    # -- manifest -------------------------------------------------------------------
    def compute_manifest(self, now: str) -> dict[str, Any]:
        entities: dict[str, set[str]] = {k: set() for k in KINDS}
        link_only = 0
        counted: set[tuple[str, str]] = set()
        for kind, key, doc in self.iter_entries():
            canon = (entity_keys(kind, doc.get("ids", {})) or [key])[0] if kind != "paper" else key
            if (kind, canon) in counted:
                continue  # alias file of an entity already counted
            counted.add((kind, canon))
            entities[kind].add(canon)
            link_only += sum(1 for a in doc.get("assets", []) if not a.get("hosted"))
        files = [p for p in (self.root / "b").rglob("*") if p.is_file()] if (self.root / "b").is_dir() else []
        return {
            "version": 1, "generatedAt": now,
            "counts": {"author": len(entities["author"]), "institution": len(entities["institution"]), "paper": len(entities["paper"]),
                       "hostedFiles": len(files), "linkOnly": link_only},
            "hostedBytes": sum(p.stat().st_size for p in files),
        }

    def write_manifest(self, now: str) -> dict[str, Any]:
        """Rewrite manifest.json only when a count or the size moved (so a quiet run commits nothing)."""
        fresh = self.compute_manifest(now)
        path = self.root / "manifest.json"
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = None
        if current and current.get("counts") == fresh["counts"] and current.get("hostedBytes") == fresh["hostedBytes"]:
            return current
        _atomic_write(path, dump(fresh))
        return fresh
