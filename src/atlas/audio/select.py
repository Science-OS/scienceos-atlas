"""Which papers of a day get a clip: the top N of the day's feed by rank, plus the top K of each of the largest primary categories, de-duplicated and
capped. Operates on the same ranked list the `today` shard is built from, so the phone's digest (which picks from that shard) finds its papers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    key: str                  # whatever the caller identifies a paper by (arXiv id)
    category: str | None      # primary category
    rank: int                 # 1-based rank in the shard


def select(ranked: Sequence[Candidate], *, top_overall: int, per_category: int, categories: int, max_papers: int) -> list[tuple[Candidate, str]]:
    """`ranked` is best first. Returns [(candidate, why)] in generation priority order, at most `max_papers`:

    1. the `top_overall` best papers (why = "top"), in rank order;
    2. then the category picks, de-duplicated against everything already chosen: the `per_category` best papers of each of the `categories` most
       common primary categories of the feed (ties: alphabetical), ordered by pick depth first (every category's best, then every category's
       second best ...) so that hitting the cap drops the third picks, not whole categories. why = "category:<cat>"."""
    chosen: list[tuple[Candidate, str]] = []
    seen: set[str] = set()
    for c in list(ranked)[: max(0, top_overall)]:
        if c.key not in seen:
            seen.add(c.key)
            chosen.append((c, "top"))
    counts = Counter(c.category for c in ranked if c.category)
    biggest = [cat for cat, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max(0, categories)]]
    picks_by_cat = {cat: [c for c in ranked if c.category == cat][: max(0, per_category)] for cat in biggest}
    for depth in range(max(0, per_category)):
        for cat in biggest:
            picks = picks_by_cat[cat]
            if depth < len(picks) and picks[depth].key not in seen:
                seen.add(picks[depth].key)
                chosen.append((picks[depth], f"category:{cat}"))
    return chosen[: max(0, max_papers)]
