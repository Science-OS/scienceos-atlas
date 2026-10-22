"""Paper-level author join: arXiv gives name strings only, OpenAlex gives ids. Match by position + name
similarity; never attach an id on a name-only guess below the thresholds."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_name(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _PUNCT.sub(" ", s)
    return s.split()


def name_similarity(a: str, b: str) -> float:
    ta, tb = normalize_name(a), normalize_name(b)
    if not ta or not tb:
        return 0.0
    if sorted(ta) == sorted(tb):
        return 1.0
    # same surname (last token) and same first initial: "J. Smith" ~ "John Smith"
    if ta[-1] == tb[-1] and ta[0][0] == tb[0][0]:
        return 0.9
    # surname and given name swapped with an initial: "Wei L" ~ "Li Wei"
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


POSITIONAL_MIN = 0.75
ANYWHERE_MIN = 0.9


def match_authors(oai_names: list[str], authorships: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Returns {position in oai_names -> OpenAlex authorship}, only for authorships that carry an author id."""
    matched: dict[int, dict[str, Any]] = {}
    used: set[int] = set()
    for i, name in enumerate(oai_names):  # pass 1: same position
        if i < len(authorships):
            au = authorships[i]
            if au.get("id") and name_similarity(name, au.get("name") or "") >= POSITIONAL_MIN:
                matched[i] = au
                used.add(i)
    for i, name in enumerate(oai_names):  # pass 2: strong match elsewhere
        if i in matched:
            continue
        best, best_j = 0.0, -1
        for j, au in enumerate(authorships):
            if j in used or not au.get("id"):
                continue
            sim = name_similarity(name, au.get("name") or "")
            if sim > best:
                best, best_j = sim, j
        if best >= ANYWHERE_MIN:
            matched[i] = authorships[best_j]
            used.add(best_j)
    return matched
