"""Polite HTTP: per-host throttle, single connection, Retry-After handling, bounded retries."""

from __future__ import annotations

import email.utils
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("atlas.http")

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class _TooLarge(Exception):
    pass


class HttpError(RuntimeError):
    """Raised when a request still fails after all retries."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Throttle:
    """Enforces a minimum interval between the starts of consecutive requests."""

    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delta = self.min_interval_s - (now - self._last)
            if delta > 0:
                self._sleep(delta)
                now = self._clock()
        self._last = now


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP date. Returns seconds or None."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds())


class PoliteClient:
    """One connection, one throttle. Every request (including retries) waits its turn."""

    def __init__(
        self,
        user_agent: str,
        min_interval_s: float,
        *,
        max_retries: int = 5,
        max_retry_after_s: float = 300,
        timeout_s: float = 60,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        throttle: Throttle | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self._sleep = sleep
        self.throttle = throttle or Throttle(min_interval_s, clock=clock, sleep=sleep)
        self.max_retries = max_retries
        self.max_retry_after_s = max_retry_after_s
        headers = {"User-Agent": user_agent, **(default_headers or {})}
        self._client = httpx.Client(
            headers=headers,
            timeout=timeout_s,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            transport=transport,
        )
        self.requests_made = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _fetch(self, method: str, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None,
               max_bytes: int | None, follow_redirects: bool | None) -> httpx.Response:
        extra: dict[str, Any] = {} if follow_redirects is None else {"follow_redirects": follow_redirects}
        if max_bytes is None:
            return self._client.request(method, url, params=params, headers=headers, **extra)
        with self._client.stream(method, url, params=params, headers=headers, **extra) as resp:
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise _TooLarge()
            buf = bytearray()
            for chunk in resp.iter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise _TooLarge()
            kept = [(k, v) for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "content-length")]
            return httpx.Response(resp.status_code, headers=kept, content=bytes(buf), request=resp.request)

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Response:
        """GET with throttle + retries. With `max_bytes` the body is streamed and the call fails
        (HttpError, status None) as soon as it grows past the cap. `follow_redirects` overrides the
        client default for this call (None keeps it)."""
        return self.request("GET", url, params=params, headers=headers, max_bytes=max_bytes, follow_redirects=follow_redirects)

    def head(self, url: str, headers: dict[str, str] | None = None, follow_redirects: bool | None = None) -> httpx.Response:
        return self.request("HEAD", url, headers=headers, follow_redirects=follow_redirects)

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Response:
        last: str = "no attempt"
        for attempt in range(self.max_retries + 1):
            self.throttle.wait()
            self.requests_made += 1
            try:
                resp = self._fetch(method, url, params, headers, max_bytes, follow_redirects)
            except _TooLarge as exc:
                raise HttpError(f"{method} {url} exceeds {max_bytes} bytes") from exc
            except httpx.TransportError as exc:  # timeouts, resets, DNS
                last = f"{type(exc).__name__}: {exc}"
                delay = min(2.0 * (2**attempt), self.max_retry_after_s)
                log.warning("%s %s failed (%s); retry %d/%d in %.0fs", method, url, last, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
                continue
            if resp.status_code not in RETRY_STATUSES:
                return resp
            last = f"HTTP {resp.status_code}"
            retry_after = parse_retry_after(resp.headers.get("Retry-After"))
            delay = retry_after if retry_after is not None else 2.0 * (2**attempt)
            delay = min(delay, self.max_retry_after_s)
            if attempt == self.max_retries:
                break
            log.warning("%s %s -> %s; Retry-After=%s; sleeping %.0fs (retry %d/%d)", method, url, resp.status_code,
                        resp.headers.get("Retry-After"), delay, attempt + 1, self.max_retries)
            self._sleep(delay)
        raise HttpError(f"{method} {url} failed after {self.max_retries} retries ({last})",
                        status=int(last.split()[-1]) if last.startswith("HTTP") else None)
