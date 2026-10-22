from __future__ import annotations

import math

import pytest

from atlas.config import ScoringCfg
from atlas.scoring import score_paper

CFG = ScoringCfg()
BASE = dict(hf_upvotes=None, github_stars=None, n_categories=1, age_days=0, has_code=False)


def s(**kw):
    return score_paper(CFG, **{**BASE, **kw})


def test_weights_sum_to_one():
    assert math.isclose(CFG.w_hf + CFG.w_github + CFG.w_cross + CFG.w_recency + CFG.w_code, 1.0)


def test_fresh_paper_without_signals_scores_only_recency():
    assert s() == 0.15


def test_maximum_is_one():
    assert s(hf_upvotes=10_000, github_stars=10**6, n_categories=9, has_code=True) == 1.0


def test_documented_formula_value():
    # hf: ln(1+25)/ln(101) ; gh: ln(1+500)/ln(5001) ; cross: 2/3 ; recency: 0.5 ; code: 1
    hf = math.log1p(25) / math.log1p(100)
    gh = math.log1p(500) / math.log1p(5000)
    expected = 0.5 * hf + 0.15 * gh + 0.10 * (2 / 3) + 0.15 * 0.5 ** (3 / 3.0) + 0.10
    assert s(hf_upvotes=25, github_stars=500, n_categories=3, age_days=3, has_code=True) == pytest.approx(expected, abs=1e-4)


def test_monotonic_in_each_signal():
    assert s(hf_upvotes=10) > s(hf_upvotes=1) > s()
    assert s(github_stars=100) > s(github_stars=1) > s()
    assert s(n_categories=3) > s(n_categories=2) > s()
    assert s(age_days=1) > s(age_days=5)
    assert s(has_code=True) > s()


def test_recency_half_life():
    assert s(age_days=3) == pytest.approx(0.075, abs=1e-4)


def test_negative_and_missing_inputs_are_clamped():
    assert s(hf_upvotes=-5, github_stars=-1, age_days=-2, n_categories=0) == s()


def test_hf_dominates_ranking_between_otherwise_equal_papers():
    assert s(hf_upvotes=30) > s(github_stars=5000, n_categories=2)
