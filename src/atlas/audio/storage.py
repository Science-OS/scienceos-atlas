"""The audio store on disk - the checkout of the `audio` branch in CI:

    clips/<id>.m4a      immutable AAC clips, named by content key (sha256(engine version | voice | text)[:32])
    days/<date>.json    one day manifest (scienceos.atlas.audio-day/1) per feed date, the last `retention_days` of them

Clips are only ever added or pruned (never rewritten): the same text and voice always map to the same name."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("atlas.audio.storage")

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID = re.compile(r"^[0-9a-f]{32}$")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clip_id(engine_version: str, voice: str, text: str) -> str:
    """The content key of a clip: sha256(engine version | voice | text), first 32 hex characters."""
    return hashlib.sha256(f"{engine_version}|{voice}|{text}".encode("utf-8")).hexdigest()[:32]


def dump_manifest(doc: Any) -> bytes:
    """Compact JSON (the word arrays are the bulk), UTF-8 as is, one trailing newline."""
    return (json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def manifest_clips(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every Clip object of a day manifest (intros, outro, papers)."""
    out = list((doc.get("intros") or {}).values())
    if doc.get("outro"):
        out.append(doc["outro"])
    out.extend((doc.get("papers") or {}).values())
    return out


class AudioStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.clips_dir = self.root / "clips"
        self.days_dir = self.root / "days"

    # -- days -------------------------------------------------------------------
    def days(self) -> list[str]:
        return sorted(p.stem for p in self.days_dir.glob("*.json") if _DAY.match(p.stem)) if self.days_dir.is_dir() else []

    def day_path(self, day: str) -> Path:
        return self.days_dir / f"{day}.json"

    def load_day(self, day: str) -> dict[str, Any] | None:
        try:
            return json.loads(self.day_path(day).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def write_day(self, day: str, doc: dict[str, Any]) -> bytes:
        data = dump_manifest(doc)
        atomic_write(self.day_path(day), data)
        return data

    # -- clips ------------------------------------------------------------------
    def clip_path(self, clip_id: str) -> Path:
        if not _ID.match(clip_id):
            raise ValueError(f"not a clip id: {clip_id!r}")
        return self.clips_dir / f"{clip_id}.m4a"

    def write_clip(self, clip_id: str, data: bytes) -> None:
        atomic_write(self.clip_path(clip_id), data)

    def clip_index(self) -> dict[str, dict[str, Any]]:
        """{clip id: Clip object} over every stored day (the newest manifest wins): what can be reused without synthesis."""
        index: dict[str, dict[str, Any]] = {}
        for day in self.days():   # ascending, so later days overwrite earlier ones
            doc = self.load_day(day)
            for clip in manifest_clips(doc) if doc else []:
                index[clip["id"]] = clip
        return index

    def intact(self, clip: dict[str, Any]) -> bool:
        """The file of a manifest clip exists and is byte-identical to what the manifest says."""
        try:
            data = self.clip_path(clip["id"]).read_bytes()
        except (OSError, ValueError):
            return False
        return len(data) == clip["bytes"] and sha256_hex(data) == clip["sha256"]

    # -- retention --------------------------------------------------------------
    def prune(self, latest: str, retention_days: int) -> dict[str, list[str]]:
        """Drop the manifests older than `retention_days` before `latest` (inclusive window), then every clip no remaining manifest references."""
        keep_from = (date.fromisoformat(latest) - timedelta(days=max(1, retention_days) - 1)).isoformat()
        removed_days = []
        for day in self.days():
            if day < keep_from:
                self.day_path(day).unlink(missing_ok=True)
                removed_days.append(day)
        referenced = {c["id"] for day in self.days() for c in manifest_clips(self.load_day(day) or {})}
        removed_clips = []
        for f in sorted(self.clips_dir.glob("*")) if self.clips_dir.is_dir() else []:
            if f.name.startswith(".tmp-") or (f.suffix == ".m4a" and f.stem not in referenced):
                f.unlink(missing_ok=True)
                removed_clips.append(f.name)
        return {"days": removed_days, "clips": removed_clips}

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file() and ".git" not in p.parts and p.name != ".git")
