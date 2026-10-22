"""OpenAlex enrichment (CC0 data). Singleton GETs are free; filter/list calls cost 1 credit.

No API key is required. Without one the daily budget is shared per IP, so we read X-RateLimit-Remaining
after every response and STOP (never fail) when it drops below `min_remaining_credits` or on HTTP 429.
An optional OPENALEX_API_KEY env var raises the budget; it is sent as a query parameter and never logged.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import OpenAlexCfg
from .http import HttpError, PoliteClient

log = logging.getLogger("atlas.openalex")

WORK_SELECT = "id,doi,cited_by_count,publication_date,authorships,primary_topic"
AUTHOR_SELECT = (
    "id,display_name,display_name_alternatives,orcid,works_count,cited_by_count,summary_stats,"
    "last_known_institutions,topics,updated_date"
)

_OA_ID = re.compile(r"(?:^|/)([WAIST]\d+)$")


def short_id(value: str | None) -> str | None:
    """https://openalex.org/A123 -> A123."""
    if not value:
        return None
    m = _OA_ID.search(value.strip())
    return m.group(1) if m else None


def short_orcid(value: str | None) -> str | None:
    if not value:
        return None
    m = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])$", value.strip())
    return m.group(1) if m else None


def short_ror(value: str | None) -> str | None:
    if not value:
        return None
    tail = value.strip().rstrip("/").rsplit("/", 1)[-1].lower()
    return tail if re.fullmatch(r"[0-9a-z]+", tail) else None


def compact_institution(inst: dict[str, Any]) -> dict[str, Any] | None:
    ror = short_ror(inst.get("ror"))
    name = inst.get("display_name")
    if not ror or not name:
        return None
    country = inst.get("country_code")
    return {
        "ror": ror,
        "openalex": short_id(inst.get("id")),
        "name": name,
        "country": country if isinstance(country, str) and re.fullmatch(r"[A-Z]{2}", country) else None,
        "type": inst.get("type"),
    }


def compact_work(work: dict[str, Any]) -> dict[str, Any]:
    authorships = []
    for au in work.get("authorships") or []:
        author = au.get("author") or {}
        insts = [ci for ci in (compact_institution(i) for i in au.get("institutions") or []) if ci]
        authorships.append(
            {
                "id": short_id(author.get("id")),
                "name": author.get("display_name"),
                "orcid": short_orcid(author.get("orcid")),
                "institutions": insts,
            }
        )
    topic = work.get("primary_topic") or {}
    return {
        "work_id": short_id(work.get("id")),
        "cited_by_count": work.get("cited_by_count"),
        "publication_date": work.get("publication_date"),
        "primary_topic": topic.get("display_name"),
        "authorships": authorships,
    }


def compact_author(a: dict[str, Any]) -> dict[str, Any]:
    stats = a.get("summary_stats") or {}
    topics = sorted(a.get("topics") or [], key=lambda t: -(t.get("count") or 0))
    insts = [ci for ci in (compact_institution(i) for i in a.get("last_known_institutions") or []) if ci]
    return {
        "id": short_id(a.get("id")),
        "name": a.get("display_name"),
        "alternatives": [x for x in (a.get("display_name_alternatives") or []) if isinstance(x, str)],
        "orcid": short_orcid(a.get("orcid")),
        "works_count": a.get("works_count"),
        "cited_by_count": a.get("cited_by_count"),
        "h_index": stats.get("h_index"),
        "i10_index": stats.get("i10_index"),
        "institutions": insts,
        "topics": [t["display_name"] for t in topics[:5] if t.get("display_name")],
        "updated_date": a.get("updated_date"),
    }


@dataclass
class WorkResult:
    status: str  # found | missing | stopped | error
    work: dict[str, Any] | None = None


class OpenAlexClient:
    def __init__(self, cfg: OpenAlexCfg, user_agent: str, *, transport: httpx.BaseTransport | None = None,
                 api_key: str | None = None, sleep=None):
        kwargs: dict[str, Any] = {}
        if sleep is not None:
            kwargs["sleep"] = sleep
        self.cfg = cfg
        self._http = PoliteClient(user_agent, cfg.min_interval_s, max_retries=1, max_retry_after_s=5,
                                  transport=transport, **kwargs)
        self._api_key = api_key if api_key is not None else os.environ.get("OPENALEX_API_KEY")
        self.remaining: int | None = None
        self.limit: int | None = None
        self.credits_used = 0
        self.first_remaining: int | None = None
        self.stopped_reason: str | None = None
        self.requests = 0

    def close(self) -> None:
        self._http.close()

    # -- budget -----------------------------------------------------------------
    def _observe(self, resp: httpx.Response) -> None:
        def _i(name: str) -> int | None:
            try:
                return int(resp.headers[name])
            except (KeyError, ValueError):
                return None

        rem, lim = _i("x-ratelimit-remaining"), _i("x-ratelimit-limit")
        if rem is not None:
            if self.first_remaining is None:
                self.first_remaining = rem
            self.remaining = rem
        if lim is not None:
            self.limit = lim
        used = _i("x-ratelimit-credits-used")
        if used:
            self.credits_used += used
        if self.remaining is not None and self.remaining < self.cfg.min_remaining_credits and not self.stopped_reason:
            self.stopped_reason = f"budget_low (remaining={self.remaining})"
            log.warning("OpenAlex budget low (%s credits left); stopping enrichment", self.remaining)

    @property
    def active(self) -> bool:
        return self.stopped_reason is None

    def _params(self, extra: dict[str, str]) -> dict[str, str]:
        params = dict(extra)
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def _get(self, path: str, params: dict[str, str]) -> httpx.Response | None:
        self.requests += 1
        try:
            resp = self._http.get(f"{self.cfg.api_base}/{path}", params=self._params(params))
        except HttpError as exc:
            if exc.status == 429:
                self.stopped_reason = "rate_limited (HTTP 429)"
                log.warning("OpenAlex 429; stopping enrichment for this run")
            else:
                log.warning("OpenAlex request failed: %s", exc)
            return None
        self._observe(resp)
        return resp

    # -- calls ------------------------------------------------------------------
    def get_work(self, arxiv_id: str) -> WorkResult:
        """Free singleton GET works/doi:10.48550/arXiv.<id>."""
        if not self.active:
            return WorkResult("stopped")
        resp = self._get(f"works/doi:10.48550/arXiv.{arxiv_id}", {"select": WORK_SELECT})
        if resp is None:
            return WorkResult("stopped" if not self.active else "error")
        if resp.status_code == 404:
            return WorkResult("missing")
        if resp.status_code != 200:
            log.warning("OpenAlex work %s -> HTTP %d", arxiv_id, resp.status_code)
            return WorkResult("error")
        try:
            return WorkResult("found", compact_work(resp.json()))
        except (ValueError, AttributeError, TypeError) as exc:
            log.warning("OpenAlex work %s unparsable: %s", arxiv_id, exc)
            return WorkResult("error")

    def get_authors(self, oa_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch lookup, 1 credit per call of up to 100 ids."""
        found: dict[str, dict[str, Any]] = {}
        for i in range(0, len(oa_ids), 100):
            if not self.active:
                break
            chunk = oa_ids[i : i + 100]
            resp = self._get(
                "authors",
                {"filter": "ids.openalex:" + "|".join(chunk), "per-page": "100", "select": AUTHOR_SELECT},
            )
            if resp is None or resp.status_code != 200:
                if resp is not None:
                    log.warning("OpenAlex authors batch -> HTTP %d", resp.status_code)
                continue
            try:
                for raw in resp.json().get("results", []):
                    ca = compact_author(raw)
                    if ca["id"]:
                        found[ca["id"]] = ca
            except (ValueError, AttributeError, TypeError) as exc:
                log.warning("OpenAlex authors batch unparsable: %s", exc)
        return found
