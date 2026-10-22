"""Optional LLM briefs. OFF unless the provider's API key env var exists (briefs.enabled: auto).

Provider interface + cost guard + cache-by-paper-id (the cache lives in the SQLite store). The apps never
need this: without it a brief is HF `ai_summary` when present, else absent. Secrets are read from the
environment only and never logged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import BriefsCfg, ProviderCfg

log = logging.getLogger("atlas.briefs")

SYSTEM_PROMPT = (
    "You write one-to-two sentence plain-language briefs of scientific papers for a researcher's feed. "
    "Say what the paper does and the main result or why it matters, in at most 45 words. "
    "No hype, no hedging boilerplate, no markdown, no quotes around the text. Use only the title and abstract."
)


@dataclass
class BriefResult:
    text: str
    input_tokens: int
    output_tokens: int


class BriefProvider(Protocol):
    name: str
    model: str

    def generate(self, title: str, abstract: str, max_output_tokens: int) -> BriefResult: ...


def _user_prompt(title: str, abstract: str) -> str:
    return f"Title: {title}\n\nAbstract: {abstract[:3500]}"


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, pcfg: ProviderCfg, transport: httpx.BaseTransport | None = None):
        self.model = pcfg.model
        self._extra = pcfg.extra
        self._http = httpx.Client(
            base_url="https://api.openai.com/v1", timeout=60, transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def generate(self, title: str, abstract: str, max_output_tokens: int) -> BriefResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _user_prompt(title, abstract)}],
            "max_completion_tokens": max_output_tokens,
            **self._extra,
        }
        resp = self._http.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        return BriefResult(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, pcfg: ProviderCfg, transport: httpx.BaseTransport | None = None):
        self.model = pcfg.model
        self._extra = pcfg.extra
        self._http = httpx.Client(
            base_url="https://api.anthropic.com/v1", timeout=60, transport=transport,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )

    def generate(self, title: str, abstract: str, max_output_tokens: int) -> BriefResult:
        body = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": _user_prompt(title, abstract)}],
            **self._extra,
        }
        resp = self._http.post("/messages", json=body)
        resp.raise_for_status()
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        usage = data.get("usage") or {}
        return BriefResult(text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))


def resolve_provider(cfg: BriefsCfg, env: Mapping[str, str] | None = None,
                     transport: httpx.BaseTransport | None = None) -> tuple[BriefProvider, ProviderCfg] | None:
    """None when briefs are off (no secret, or enabled: false)."""
    env = os.environ if env is None else env
    enabled = str(cfg.enabled).lower()
    if enabled == "false":
        return None
    pcfg = cfg.providers.get(cfg.provider)
    if pcfg is None:
        log.warning("briefs.provider %r has no entry in briefs.providers", cfg.provider)
        return None
    key = env.get(pcfg.env)
    if not key:
        if enabled == "true":
            log.warning("briefs.enabled=true but %s is not set; briefs stay off", pcfg.env)
        return None
    cls = OpenAIProvider if cfg.provider == "openai" else AnthropicProvider
    return cls(key, pcfg, transport), pcfg


def estimate_cost_usd(pcfg: ProviderCfg, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * pcfg.usd_per_mtok_in + output_tokens * pcfg.usd_per_mtok_out) / 1_000_000


@dataclass
class BriefStats:
    generated: int = 0
    skipped_cached: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    stopped_reason: str | None = None


def generate_briefs(
    provider: BriefProvider,
    pcfg: ProviderCfg,
    papers: Iterable[tuple[str, str, str]],  # (arxiv_id, title, abstract), best first
    cached_ids: set[str],
    *,
    max_cost_usd: float,
    max_output_tokens: int,
    save: Callable[[str, BriefResult, float], None],
) -> BriefStats:
    """Cost guard: before each call project the worst case (input estimate + max output); stop at the cap."""
    stats = BriefStats()
    for arxiv_id, title, abstract in papers:
        if arxiv_id in cached_ids:
            stats.skipped_cached += 1
            continue
        projected_in = (len(title) + len(abstract[:3500]) + len(SYSTEM_PROMPT)) // 3 + 20
        worst = estimate_cost_usd(pcfg, projected_in, max_output_tokens)
        if stats.cost_usd + worst > max_cost_usd:
            stats.stopped_reason = f"cost_guard (spent ${stats.cost_usd:.4f} of ${max_cost_usd:.2f})"
            log.warning("Brief cost guard hit: %s", stats.stopped_reason)
            break
        try:
            result = provider.generate(title, abstract, max_output_tokens)
        except Exception as exc:  # never crash the run for one paper
            stats.failed += 1
            log.warning("Brief for %s failed: %s", arxiv_id, type(exc).__name__)
            continue
        if not result.text:
            stats.failed += 1
            continue
        cost = estimate_cost_usd(pcfg, result.input_tokens, result.output_tokens)
        stats.cost_usd += cost
        stats.generated += 1
        save(arxiv_id, result, cost)
    return stats
