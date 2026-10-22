"""Trending score. Pure function, documented in README, unit-tested.

score = w_hf * hf + w_github * gh + w_cross * cross + w_recency * recency + w_code * code

  hf       = min(1, ln(1 + upvotes) / ln(1 + hf_upvotes_saturation))
  gh       = min(1, ln(1 + stars)   / ln(1 + github_stars_saturation))
  cross    = min(1, (n_categories - 1) / cross_listing_saturation)
  recency  = 0.5 ** (age_days / recency_half_life_days)      (age relative to the shard date)
  code     = 1 if a GitHub repo is linked on Hugging Face else 0

The weights sum to 1, so score is in [0, 1]. Rounded to 4 decimals.
"""

from __future__ import annotations

import math

from .config import ScoringCfg


def score_paper(
    cfg: ScoringCfg,
    *,
    hf_upvotes: int | float | None,
    github_stars: int | float | None,
    n_categories: int,
    age_days: float,
    has_code: bool,
) -> float:
    hf = min(1.0, math.log1p(max(0.0, hf_upvotes or 0)) / math.log1p(cfg.hf_upvotes_saturation))
    gh = min(1.0, math.log1p(max(0.0, github_stars or 0)) / math.log1p(cfg.github_stars_saturation))
    cross = min(1.0, max(0, n_categories - 1) / cfg.cross_listing_saturation)
    recency = 0.5 ** (max(0.0, age_days) / cfg.recency_half_life_days)
    code = 1.0 if has_code else 0.0
    raw = cfg.w_hf * hf + cfg.w_github * gh + cfg.w_cross * cross + cfg.w_recency * recency + cfg.w_code * code
    return round(raw, 4)
