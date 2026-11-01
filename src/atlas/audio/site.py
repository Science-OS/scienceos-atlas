"""Audio store -> site/audio for the Pages deploy.

    site/audio/<date>.json        the day manifests (url "audio/clips/<id>.m4a" is relative to the site root)
    site/audio/clips/<id>.m4a     the clips they reference, nothing else

Audio must never fail the feed, so a day is published only if it passes the same checks `validate_site` applies (schema, clip files present with
the right size and SHA-256, consistent timings) and fits the size budget; anything else is left out and reported in status.json."""

from __future__ import annotations

import logging
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ..config import AudioCfg
from ..validate import audio_day_errors
from .storage import AudioStore, manifest_clips

log = logging.getLogger("atlas.audio.site")


def publish_to_site(store: AudioStore, site_dir: Path, latest: str, cfg: AudioCfg, validator: Draft202012Validator) -> dict[str, Any]:
    dest = Path(site_dir) / "audio"
    shutil.rmtree(dest, ignore_errors=True)
    keep_from = (date.fromisoformat(latest) - timedelta(days=max(1, cfg.retention_days) - 1)).isoformat()
    days_out: list[str] = []
    skipped: dict[str, str] = {}
    copied: set[str] = set()
    hashed: dict[str, str | None] = {}
    total = 0
    budget = cfg.max_site_mb * 1024 * 1024
    for day in reversed(store.days()):   # newest first: if the budget bites, the oldest days go
        if day < keep_from or day > latest:
            skipped[day] = "outside the retention window"
            continue
        doc = store.load_day(day)
        errs = audio_day_errors(doc, validator, store.clips_dir, label=f"audio/{day}.json", expect_date=day, hashed=hashed) if doc else ["unreadable"]
        if errs:
            skipped[day] = errs[0]
            log.warning("audio day %s not published: %s", day, errs[0])
            continue
        manifest = store.day_path(day).read_bytes()
        new_ids = {c["id"]: c["bytes"] for c in manifest_clips(doc) if c["id"] not in copied}
        size = len(manifest) + sum(new_ids.values())
        if total + size > budget:
            skipped[day] = f"over the {cfg.max_site_mb} MB audio budget"
            log.warning("audio day %s not published: %s", day, skipped[day])
            continue
        (dest / "clips").mkdir(parents=True, exist_ok=True)
        (dest / f"{day}.json").write_bytes(manifest)
        for cid in new_ids:
            shutil.copyfile(store.clip_path(cid), dest / "clips" / f"{cid}.m4a")
        copied.update(new_ids)
        total += size
        days_out.append(day)
    return {"days": sorted(days_out), "clips": len(copied), "bytes": total, "skipped": skipped}
