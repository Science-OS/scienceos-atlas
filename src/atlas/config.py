"""Single source of truth for every tunable. Loaded from config.yaml (override: ATLAS_CONFIG)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArxivCfg(_Cfg):
    oai_base: str = "https://oaipmh.arxiv.org/oai"
    api_base: str = "https://export.arxiv.org/api/query"
    pdf_base: str = "https://export.arxiv.org/pdf"
    sets: list[str] = ["cs", "stat", "eess", "q-bio", "econ"]
    categories: list[str] = ["cs.*", "stat.*", "eess.*", "q-bio.*", "econ.*"]
    min_interval_s: float = 3.0
    max_retries: int = 5
    max_retry_after_s: float = 300
    new_paper_max_age_days: int = 6
    verify_batch: int = 100
    cold_start_backfill_days: int = 2
    max_catchup_days: int = 7


class FeedsCfg(_Cfg):
    retention_days: int = 30
    today_max_items: int = 200
    trending_max_items: int = 50
    trending_window_days: int = 7
    category_max_items: int = 300
    paper_files_top_n: int = 300
    categories: list[str] = []


class ScoringCfg(_Cfg):
    w_hf: float = 0.50
    w_github: float = 0.15
    w_cross: float = 0.10
    w_recency: float = 0.15
    w_code: float = 0.10
    hf_upvotes_saturation: float = 100
    github_stars_saturation: float = 5000
    cross_listing_saturation: float = 3
    recency_half_life_days: float = 3.0


class HfCfg(_Cfg):
    enabled: bool = True
    api_base: str = "https://huggingface.co/api/daily_papers"
    days: int = 7
    limit: int = 100
    max_pages: int = 3
    min_interval_s: float = 0.5


class OpenAlexCfg(_Cfg):
    enabled: bool = True
    api_base: str = "https://api.openalex.org"
    enrich_top_n: int = 100
    retry_days: int = 7
    retry_after_hours: float = 18
    author_ttl_days: int = 7
    max_authors_per_paper: int = 25
    min_remaining_credits: int = 100
    min_interval_s: float = 0.2


class ThumbsCfg(_Cfg):
    enabled: bool = True
    top_n: int = 100
    max_per_run: int = 40
    width_px: int = 600
    jpeg_quality: int = 72
    max_pdf_mb: int = 30
    retry_failed_after_days: int = 7
    allowed_license_prefixes: list[str] = [
        "http://creativecommons.org/licenses/by/",
        "http://creativecommons.org/licenses/by-sa/",
        "http://creativecommons.org/publicdomain/zero/",
    ]


class ProviderCfg(_Cfg):
    env: str
    model: str
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    extra: dict[str, Any] = Field(default_factory=dict)  # merged into the request body


class BriefsCfg(_Cfg):
    enabled: Literal["auto", "true", "false"] | bool = "auto"
    provider: Literal["openai", "anthropic"] = "openai"
    top_n: int = 100
    max_cost_usd_per_run: float = 0.25
    max_output_tokens: int = 160
    providers: dict[str, ProviderCfg] = Field(default_factory=dict)


class OutputCfg(_Cfg):
    max_site_mb: int = 450


class Config(_Cfg):
    site_base_url: str = "https://science-os.github.io/scienceos-atlas"
    user_agent: str = "ScienceOS-Atlas/1.0 (+https://github.com/Science-OS/scienceos-atlas)"
    arxiv: ArxivCfg = ArxivCfg()
    feeds: FeedsCfg = FeedsCfg()
    scoring: ScoringCfg = ScoringCfg()
    hf: HfCfg = HfCfg()
    openalex: OpenAlexCfg = OpenAlexCfg()
    thumbs: ThumbsCfg = ThumbsCfg()
    briefs: BriefsCfg = BriefsCfg()
    output: OutputCfg = OutputCfg()


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path or os.environ.get("ATLAS_CONFIG") or REPO_ROOT / "config.yaml")
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    return Config.model_validate(data or {})
