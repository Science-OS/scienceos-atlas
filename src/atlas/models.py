"""Typed builders for the atlas.schema.json objects we emit. The JSON Schema stays the contract;
these models just make construction safe, and `dump()` omits unset fields (schema convention)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def ts(dt: datetime | None = None) -> str:
    """RFC 3339 UTC with literal Z, no fractional digits."""
    dt = (dt or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


class MetricValue(_M):
    value: int | float
    source: str
    fetchedAt: str


class LinkOut(_M):
    kind: str
    url: str
    label: str | None = None


class PaperAuthor(_M):
    name: str
    authorId: str | None = None
    position: int
    affiliation: str | None = None


class Brief(_M):
    text: str
    source: str
    model: str | None = None
    generatedAt: str


class PaperSignals(_M):
    hfUpvotes: MetricValue | None = None
    githubStars: MetricValue | None = None
    githubRepo: str | None = None


class Paper(_M):
    id: str
    arxivId: str | None = None
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[PaperAuthor]
    categories: list[str]
    primaryCategory: str | None = None
    publishedAt: str
    updatedAt: str | None = None
    version: int | None = None
    license: str | None = None
    venue: str | None = None
    links: list[LinkOut]
    citations: MetricValue | None = None
    signals: PaperSignals | None = None
    brief: Brief | None = None


class InstitutionRef(_M):
    id: str
    name: str
    country: str | None = None
    type: str | None = None
    logoUrl: str | None = None


class InstitutionExternalIds(_M):
    ror: str | None = None
    wikidata: str | None = None
    openalex: str | None = None


class Institution(_M):
    id: str
    name: str
    acronym: str | None = None
    country: str | None = None
    city: str | None = None
    type: str | None = None
    website: str | None = None
    ids: InstitutionExternalIds
    updatedAt: str | None = None


class AuthorExternalIds(_M):
    orcid: str | None = None
    openalex: str | None = None
    wikidata: str | None = None
    semanticScholar: str | None = None
    dblp: str | None = None
    googleScholar: str | None = None
    openreview: str | None = None
    github: str | None = None
    bluesky: str | None = None
    x: str | None = None


class AuthorMetrics(_M):
    citations: list[MetricValue]
    hIndex: list[MetricValue]
    i10Index: list[MetricValue]
    worksCount: list[MetricValue] | None = None


class Enrichment(_M):
    stage: Literal["A", "B", "C"]
    complete: bool
    pending: list[str]
    updatedAt: str


class Author(_M):
    id: str
    name: str
    aliases: list[str]
    ids: AuthorExternalIds
    bio: str | None = None
    affiliations: list[InstitutionRef]
    experience: list[dict[str, Any]]
    links: list[LinkOut]
    metrics: AuthorMetrics
    citationsByYear: list[dict[str, Any]]
    topics: list[str]
    coauthors: list[dict[str, Any]]
    claimed: bool = False
    enrichment: Enrichment
    updatedAt: str
