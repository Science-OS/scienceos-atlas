from __future__ import annotations

from atlas.audio.select import Candidate, select


def feed(spec: list[tuple[str, str]]) -> list[Candidate]:
    """[(key, category)] best first -> candidates with ranks 1..n."""
    return [Candidate(k, c, i) for i, (k, c) in enumerate(spec, start=1)]


def keys(chosen):
    return [c.key for c, _ in chosen]


def test_top_n_by_rank_then_top_k_of_the_largest_categories():
    ranked = feed([(f"p{i}", "cs.LG" if i % 3 else "cs.CV") for i in range(1, 31)] + [("q1", "cs.RO"), ("q2", "cs.RO"), ("q3", "cs.RO"), ("q4", "cs.RO")])
    chosen = select(ranked, top_overall=5, per_category=2, categories=2, max_papers=100)
    assert keys(chosen)[:5] == ["p1", "p2", "p3", "p4", "p5"]
    assert [w for _, w in chosen[:5]] == ["top"] * 5
    # the two biggest categories are cs.LG (20) and cs.CV (10); cs.RO (4) is too small. Their best two papers each are p1, p2 and p3, p6:
    # p1 p2 p3 are in the top 5 already, so p6 is the only addition
    assert [(c.key, w) for c, w in chosen[5:]] == [("p6", "category:cs.CV")]


def test_deduplication_and_exact_membership():
    ranked = feed([("a", "X"), ("b", "X"), ("c", "Y"), ("d", "Y"), ("e", "Z")])
    chosen = select(ranked, top_overall=2, per_category=1, categories=3, max_papers=40)
    # top: a, b. categories by size: X(2), Y(2) [tie -> alphabetical], Z(1). picks: X->a (dup), Y->c, Z->e
    assert keys(chosen) == ["a", "b", "c", "e"]
    assert [w for _, w in chosen] == ["top", "top", "category:Y", "category:Z"]
    assert len(set(keys(chosen))) == len(chosen)


def test_cap_drops_third_picks_before_whole_categories():
    spec = [(f"{c}{i}", c) for i in range(1, 4) for c in ("A", "B", "C", "D")]    # 12 papers, 4 categories of 3, best first interleaved
    ranked = feed(spec)
    chosen = select(ranked, top_overall=0, per_category=3, categories=4, max_papers=6)
    assert len(chosen) == 6
    cats = [w for _, w in chosen]
    # depth-first: every category's best, then the second best of the first two categories
    assert cats[:4] == ["category:A", "category:B", "category:C", "category:D"]
    assert {w for w in cats} == {"category:A", "category:B", "category:C", "category:D"}


def test_cap_applies_after_dedup_and_keeps_rank_order_for_the_top():
    ranked = feed([(f"p{i}", "A") for i in range(1, 50)])
    chosen = select(ranked, top_overall=20, per_category=3, categories=12, max_papers=40)
    assert keys(chosen) == [f"p{i}" for i in range(1, 21)]        # one category: the picks are all inside the top 20 already
    ranked = feed([(f"p{i}", f"C{i % 15:02d}") for i in range(1, 200)])
    chosen = select(ranked, top_overall=20, per_category=3, categories=12, max_papers=40)
    assert 20 < len(chosen) <= 40 and keys(chosen)[:20] == [f"p{i}" for i in range(1, 21)] and len(set(keys(chosen))) == len(chosen)
    capped = select(ranked, top_overall=20, per_category=3, categories=12, max_papers=25)
    assert len(capped) == 25 and keys(capped) == keys(chosen)[:25]       # the cap only truncates, never reorders


def test_empty_and_small_feeds_and_missing_categories():
    assert select([], top_overall=20, per_category=3, categories=12, max_papers=40) == []
    ranked = feed([("a", None), ("b", None)])
    assert keys(select(ranked, top_overall=1, per_category=3, categories=12, max_papers=40)) == ["a"]    # papers without a category only count via the top N
    assert keys(select(feed([("a", "X")]), top_overall=20, per_category=3, categories=12, max_papers=40)) == ["a"]
    assert select(feed([("a", "X")]), top_overall=0, per_category=0, categories=0, max_papers=0) == []


def test_ties_between_categories_break_alphabetically():
    ranked = feed([("a", "B"), ("b", "A"), ("c", "B"), ("d", "A")])
    chosen = select(ranked, top_overall=0, per_category=1, categories=1, max_papers=10)
    assert keys(chosen) == ["b"]       # A and B both have 2 papers; A wins the tie; its best paper is b
