"""Hugging Face Daily Papers: upvotes, linked GitHub repo + stars, ai_summary. Undocumented API; every field optional."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

from .http import PoliteClient

log = logging.getLogger("atlas.hf")

ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")


@dataclass
class HfSignal:
    arxiv_id: str
    upvotes: int | None = None
    github_repo: str | None = None  # full URL as given by HF
    github_stars: int | None = None
    project_page: str | None = None
    ai_summary: str | None = None
    ai_keywords: list[str] | None = None
    daily_date: str | None = None  # submittedOnDailyAt, YYYY-MM-DD

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "HfSignal":
        return cls(**d)


def _int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_daily(payload: Any) -> list[HfSignal]:
    """Tolerant parser: skips anything that does not look like {paper: {id, ...}}."""
    out: list[HfSignal] = []
    if not isinstance(payload, list):
        return out
    for item in payload:
        paper = item.get("paper") if isinstance(item, dict) else None
        if not isinstance(paper, dict):
            continue
        pid = paper.get("id")
        if not isinstance(pid, str) or not ARXIV_ID.match(pid):
            continue
        daily = item.get("submittedOnDailyAt") or paper.get("submittedOnDailyAt")
        summary = paper.get("ai_summary")
        kws = paper.get("ai_keywords")
        out.append(
            HfSignal(
                arxiv_id=pid,
                upvotes=_int(paper.get("upvotes")),
                github_repo=paper.get("githubRepo") if isinstance(paper.get("githubRepo"), str) else None,
                github_stars=_int(paper.get("githubStars")),
                project_page=paper.get("projectPage") if isinstance(paper.get("projectPage"), str) else None,
                ai_summary=summary.strip() if isinstance(summary, str) and summary.strip() else None,
                ai_keywords=[k for k in kws if isinstance(k, str)] if isinstance(kws, list) else None,
                daily_date=daily[:10] if isinstance(daily, str) and len(daily) >= 10 else None,
            )
        )
    return out


def fetch_day(client: PoliteClient, api_base: str, day: date, limit: int = 100, max_pages: int = 3) -> list[HfSignal]:
    signals: list[HfSignal] = []
    for page in range(max_pages):
        params: dict[str, str] = {"date": day.isoformat(), "limit": str(limit)}
        if page:
            params["p"] = str(page)
        resp = client.get(api_base, params=params)
        if resp.status_code != 200:
            log.warning("HF daily_papers %s page %d -> HTTP %d", day, page, resp.status_code)
            break
        items = resp.json()
        signals.extend(parse_daily(items))
        if not isinstance(items, list) or len(items) < limit:
            break
    return signals


def fetch_recent(client: PoliteClient, api_base: str, today: date, days: int, limit: int, max_pages: int) -> list[HfSignal]:
    """Latest signal wins when a paper shows up on several daily pages."""
    by_id: dict[str, HfSignal] = {}
    for offset in range(days):
        day = today - timedelta(days=offset)
        try:
            for sig in fetch_day(client, api_base, day, limit, max_pages):
                prev = by_id.get(sig.arxiv_id)
                if prev is None or (sig.upvotes or 0) >= (prev.upvotes or 0):
                    by_id[sig.arxiv_id] = sig
        except Exception as exc:  # one bad day must not sink the run
            log.warning("HF daily_papers %s failed: %s", day, exc)
    return list(by_id.values())
