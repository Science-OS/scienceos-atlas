"""`atlas audio render`: scripts and clips for an arbitrary list of papers, offline - no database, no feed, no network (unless asked: `--llm` for
spoken summaries, `--authors-from` for first-author institutions). Used to make the iPhone fixture and to try the voice on a few papers.

Output: `<out>/<date>.json` (the day manifest) and `<out>/clips/<id>.m4a`: exactly the layout of `site/audio/`."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..config import Config
from ..store import Store
from .align import Aligner
from .engine import TtsEngine
from .plan import PaperInput
from .select import select
from .stage import build_plan, run_synth
from .storage import AudioStore, manifest_clips
from .summaries import resolve_summary_provider

log = logging.getLogger("atlas.audio.render")


def load_papers(path: Path) -> list[tuple[dict[str, Any], int]]:
    """Paper objects with their rank, from a JSON list of Papers, {"papers": [...]}, or a feed shard ({"items": [{"paper": ..., "rank": ...}]})."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "items" in doc:
        return [(it["paper"], int(it.get("rank", i))) for i, it in enumerate(doc["items"], start=1)]
    rows = doc["papers"] if isinstance(doc, dict) else doc
    return [(p["paper"] if "paper" in p and "title" not in p else p, i) for i, p in enumerate(rows, start=1)]


def _affiliation_lookup(base: str):
    """First-author institution from the published author files (<base>/authors/<openalexId>.json), `base` a site URL or a local site directory."""
    cache: dict[str, str | None] = {}

    def fetch(author_id: str) -> str | None:
        oa = author_id.split(":", 1)[-1]
        if oa not in cache:
            try:
                if base.startswith(("http://", "https://")):
                    r = httpx.get(f"{base.rstrip('/')}/authors/{oa}.json", timeout=20, headers={"User-Agent": "ScienceOS-Atlas/1.0 (audio render)"})
                    r.raise_for_status()
                    doc = r.json()
                else:
                    doc = json.loads((Path(base) / "authors" / f"{oa}.json").read_text(encoding="utf-8"))
                affs = doc["author"].get("affiliations") or []
                cache[oa] = affs[0]["name"] if affs else None
            except Exception:
                cache[oa] = None
        return cache[oa]

    return fetch


def render(cfg: Config, papers_file: Path, day: str, out_dir: Path, *, limit: int | None = None, apply_selection: bool = False, llm: bool = False,
           authors_from: str | None = None, state_db: Path | None = None, engine: TtsEngine | None = None, aligner: Aligner | None | bool = True,
           now: datetime | None = None, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    rows = load_papers(papers_file)
    lookup = _affiliation_lookup(authors_from) if authors_from else None
    inputs: list[PaperInput] = []
    for paper, rank in rows:
        aff = None
        first = sorted(paper.get("authors") or [], key=lambda a: a.get("position", 0))[:1]
        if lookup and first and not first[0].get("affiliation") and first[0].get("authorId"):
            aff = lookup(first[0]["authorId"])
        inputs.append(PaperInput.from_paper_json(paper, rank, aff))
    if apply_selection:
        ac = cfg.audio
        by_id = {i.id: i for i in inputs}
        chosen = select([i.candidate() for i in inputs], top_overall=ac.top_overall, per_category=ac.per_category, categories=ac.categories, max_papers=ac.max_papers)
        inputs = []
        for c, why in chosen:
            by_id[c.key].why = why
            inputs.append(by_id[c.key])
    if limit:
        inputs = inputs[:limit]
    store = Store(state_db or ":memory:")
    t_scripts = time.monotonic()
    try:
        resolved = resolve_summary_provider(cfg, transport=transport) if llm else None
        plan, info = build_plan(cfg, day, inputs, store, now, resolved)
    finally:
        store.close()
    info["scripts_seconds"] = round(time.monotonic() - t_scripts, 2)
    with tempfile.TemporaryDirectory(prefix="atlas-audio-") as td:
        stats = run_synth(cfg, plan, Path(td), now, engine=engine, aligner=aligner)
        audio = AudioStore(td)
        doc = audio.load_day(day)
        out = Path(out_dir)
        (out / "clips").mkdir(parents=True, exist_ok=True)
        if doc is None:
            return {"ok": False, "synth": stats.as_dict(), **info}
        shutil.copyfile(audio.day_path(day), out / f"{day}.json")
        total = (out / f"{day}.json").stat().st_size
        for clip in manifest_clips(doc):
            shutil.copyfile(audio.clip_path(clip["id"]), out / "clips" / f"{clip['id']}.m4a")
            total += clip["bytes"]
    return {"ok": True, "date": day, "papers": len(doc["papers"]), "clips": len({c["id"] for c in manifest_clips(doc)}), "bytes": total,
            "manifest": str(out / f"{day}.json"), "synth": stats.as_dict(), **info}
