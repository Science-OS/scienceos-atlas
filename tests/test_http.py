from __future__ import annotations

import httpx
import pytest

from atlas.http import HttpError, PoliteClient, Throttle, parse_retry_after
from conftest import mock_transport


def test_throttle_enforces_min_interval(clock):
    th = Throttle(3.0, clock=clock.now, sleep=clock.sleep)
    th.wait()  # first call never sleeps
    th.wait()
    th.wait()
    assert clock.sleeps == [3.0, 3.0]


def test_throttle_does_not_sleep_when_enough_time_passed(clock):
    th = Throttle(3.0, clock=clock.now, sleep=clock.sleep)
    th.wait()
    clock.t += 10
    th.wait()
    assert clock.sleeps == []


def test_parse_retry_after_seconds_and_date():
    assert parse_retry_after("30") == 30
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    from datetime import datetime, timezone
    now = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Thu, 22 Oct 2026 12:01:00 GMT", now=now) == 60


def test_503_retry_after_is_honoured_then_succeeds(clock):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(503, headers={"Retry-After": "42"})
        return httpx.Response(200, text="ok")

    c = PoliteClient("ua", 3.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    assert c.get("https://x.test/oai").text == "ok"
    assert 42 in clock.sleeps  # slept exactly Retry-After
    assert calls[0].headers["user-agent"] == "ua"


def test_retry_after_is_capped(clock):
    def handler(req):
        return httpx.Response(503, headers={"Retry-After": "99999"})

    c = PoliteClient("ua", 0.0, max_retries=1, max_retry_after_s=60, transport=mock_transport(handler),
                     sleep=clock.sleep, clock=clock.now)
    with pytest.raises(HttpError) as ei:
        c.get("https://x.test/")
    assert ei.value.status == 503
    assert max(clock.sleeps) == 60


def test_gives_up_after_max_retries_on_transport_errors(clock):
    def handler(req):
        raise httpx.ConnectError("boom")

    c = PoliteClient("ua", 0.0, max_retries=2, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    with pytest.raises(HttpError):
        c.get("https://x.test/")
    assert c.requests_made == 3


def test_404_is_returned_not_retried(clock):
    c = PoliteClient("ua", 0.0, transport=mock_transport(lambda r: httpx.Response(404)), sleep=clock.sleep, clock=clock.now)
    assert c.get("https://x.test/").status_code == 404
    assert c.requests_made == 1


def test_max_bytes_cap(clock):
    c = PoliteClient("ua", 0.0, max_retries=0, transport=mock_transport(lambda r: httpx.Response(200, content=b"x" * 5000)),
                     sleep=clock.sleep, clock=clock.now)
    with pytest.raises(HttpError):
        c.get("https://x.test/", max_bytes=1000)
    assert c.get("https://x.test/", max_bytes=10_000).content == b"x" * 5000


def test_every_request_goes_through_the_shared_throttle(clock):
    c = PoliteClient("ua", 3.0, transport=mock_transport(lambda r: httpx.Response(200, text="a")), sleep=clock.sleep, clock=clock.now)
    for _ in range(4):
        c.get("https://x.test/")
    assert clock.sleeps == [3.0, 3.0, 3.0]


def test_head_goes_through_the_same_throttle_and_retries(clock):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.method)
        return httpx.Response(503, headers={"Retry-After": "5"}) if len(seen) == 1 else httpx.Response(200)

    c = PoliteClient("ua", 3.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    assert c.head("https://x.test/a").status_code == 200
    assert seen == ["HEAD", "HEAD"] and 5 in clock.sleeps and c.requests_made == 2


def test_follow_redirects_can_be_switched_off_per_call(clock):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://y.test/b"}) if req.url.host == "x.test" else httpx.Response(200, text="end")

    c = PoliteClient("ua", 0.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    assert c.get("https://x.test/a").text == "end"                                       # the client default follows
    r = c.get("https://x.test/a", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "https://y.test/b"
    streamed = c.get("https://x.test/a", max_bytes=100, follow_redirects=False)
    assert streamed.status_code == 302
