"""The script of a day: which papers, and the exact words of every clip (papers, intros, outro). The `audio_scripts` stage writes this as JSON
(`state/audio/plan-<date>.json`); the `audio` stage reads it and speaks it. The split keeps the LLM and the database away from the heavy,
isolated synthesis step, and lets `atlas audio render` build a plan from arbitrary papers without a database."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .select import Candidate
from .summaries import fallback_summary
from .text import INTRO_COUNTS, OUTRO_TEXT, intro_text, paper_clip_text


@dataclass
class PaperInput:
    """What a script needs to know about a paper (the same whether it came from the SQLite state or from a Paper JSON object)."""
    id: str                       # Paper.id exactly as in the feed: "arxiv:2610.30108"
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    affiliation: str | None = None      # of the first author, if known
    brief: str | None = None
    abstract: str | None = None
    category: str | None = None
    rank: int = 0
    why: str = "top"              # "top" | "category:<cat>" (why the selection took it)

    @classmethod
    def from_paper_json(cls, p: dict[str, Any], rank: int = 0, affiliation: str | None = None) -> "PaperInput":
        authors = sorted(p.get("authors") or [], key=lambda a: a.get("position", 0))
        pid = p["id"]
        return cls(
            id=pid, arxiv_id=p.get("arxivId") or pid.split(":", 1)[-1], title=p.get("title") or "",
            authors=[a.get("name", "") for a in authors], affiliation=affiliation or (authors[0].get("affiliation") if authors else None),
            brief=(p.get("brief") or {}).get("text"), abstract=p.get("abstract"),
            category=p.get("primaryCategory") or ((p.get("categories") or [None])[0]), rank=rank,
        )

    def candidate(self) -> Candidate:
        return Candidate(self.id, self.category, self.rank)


@dataclass
class PlanPaper:
    id: str
    arxiv_id: str
    title: str
    text: str
    summary_source: str     # "llm" | "fallback"
    why: str                # "top" | "category:cs.CV"
    rank: int


@dataclass
class DayPlan:
    date: str
    papers: list[PlanPaper]
    intros: dict[str, str]
    outro: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DayPlan":
        return cls(d["date"], [PlanPaper(**p) for p in d["papers"]], dict(d["intros"]), d["outro"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DayPlan":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def script_for(inp: PaperInput, llm_summary: str | None, *, max_words: int = 85) -> tuple[str, str] | None:
    """(clip text, summary source) for a paper, or None when there is nothing to say beyond the title."""
    summary, source = (llm_summary, "llm") if llm_summary else (fallback_summary(inp.brief, inp.abstract, max_words=max_words), "fallback")
    if not summary or not inp.title.strip():
        return None
    return paper_clip_text(inp.title, inp.authors[0] if inp.authors else "", len(inp.authors), inp.affiliation, summary), source


def day_texts(day: date) -> tuple[dict[str, str], str]:
    """The intros for 1..8 briefs and the outro of a day."""
    return {str(n): intro_text(n, day) for n in INTRO_COUNTS}, OUTRO_TEXT
