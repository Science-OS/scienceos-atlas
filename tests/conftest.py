from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

DATA = Path(__file__).parent / "data"
FIXTURES = Path(__file__).parent / "fixtures" / "atlas.static"


def data_bytes(name: str) -> bytes:
    return (DATA / name).read_bytes()


def data_json(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class FakeClock:
    """Deterministic clock + sleep so throttle/backoff tests run instantly."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def atom_feed(entries: list[dict]) -> bytes:
    """Minimal export-API Atom feed: entries = {id, v, published, updated?, primary}."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">']
    for e in entries:
        parts.append(
            f'<entry><id>http://arxiv.org/abs/{e["id"]}v{e.get("v", 1)}</id>'
            f'<updated>{e.get("updated", e["published"])}</updated><published>{e["published"]}</published>'
            f'<title>t</title><arxiv:primary_category term="{e.get("primary", "econ.GN")}"/></entry>'
        )
    parts.append("</feed>")
    return "\n".join(parts).encode()
