"""Institution homepage icons (>= 120 px) and the Google faviconV2 fallback. Both are link-only: found, size-checked, never re-hosted.
The homepage URL comes from third-party data (ROR), so every fetch and every redirect hop is SSRF-checked."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlparse

import httpx

from ..http import HttpError
from .images import longest_side
from .sources import MediaHttp

log = logging.getLogger("atlas.media.web")

FAVICON = "https://t3.gstatic.com/faviconV2"
MAX_HTML_BYTES = 1_000_000
MAX_ICON_BYTES = 400_000


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified)


def is_public_http_url(url: str, *, resolve: bool = False) -> bool:
    """Only http(s) URLs to public hosts: never localhost, RFC 1918, link-local, .local/.internal. With `resolve`, every DNS answer must be public."""
    try:
        u = urlparse(url)
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    host = u.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith((".local", ".localhost", ".internal", ".lan", ".home.arpa")):
        return False
    try:
        ipaddress.ip_address(host)
        return _is_public_ip(host)
    except ValueError:
        pass
    if "." not in host:
        return False
    if resolve:
        try:
            infos = socket.getaddrinfo(host, u.port or (443 if u.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        except OSError:
            return False
        return bool(infos) and all(_is_public_ip(i[4][0]) for i in infos)
    return True


class _IconParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.icons: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "link":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        rel = " ".join(a.get("rel", "").lower().split())
        if "icon" in rel and a.get("href"):
            self.icons.append({"href": a["href"], "rel": rel, "sizes": a.get("sizes", "")})


def _declared_size(sizes: str, rel: str) -> int:
    best = 0
    for tok in sizes.lower().split():
        m = re.fullmatch(r"(\d+)x(\d+)", tok)
        if m:
            best = max(best, int(m.group(1)), int(m.group(2)))
        elif tok == "any":
            best = max(best, 512)
    return best or (180 if "apple-touch-icon" in rel else 0)


def parse_icons(base_url: str, page: str) -> list[dict]:
    """`<link rel=icon|apple-touch-icon ...>` candidates, largest declared size first."""
    p = _IconParser()
    try:
        p.feed(page)
    except Exception:  # noqa: BLE001 - HTMLParser can choke on hostile markup; whatever was collected so far is still usable
        pass
    icons = [{"href": urljoin(base_url, i["href"]), "size": _declared_size(i["sizes"], i["rel"]), "rel": i["rel"]} for i in p.icons]
    return sorted(icons, key=lambda i: -i["size"])


def fetch_public(http: MediaHttp, url: str, max_bytes: int, hops: int = 4) -> httpx.Response | None:
    """GET a public URL, following redirects by hand so each hop is checked. None when unsafe, unreachable or not 200."""
    cur = url
    for _ in range(hops):
        if not is_public_http_url(cur, resolve=http.resolve_dns):
            return None
        try:
            resp = http.web.get(cur, max_bytes=max_bytes, follow_redirects=False)
        except HttpError:
            return None
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
            cur = urljoin(cur, resp.headers["location"])
            continue
        return resp if resp.status_code == 200 else None
    return None


def _is_svg(data: bytes) -> bool:
    head = data[:2048].lstrip().lower()
    return head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head)


def homepage_icon(http: MediaHttp, site: str, min_px: int = 120, tries: int = 3) -> tuple[str, int | None] | None:
    """(icon url, measured px) of the largest declared icon that really is >= min_px. SVG counts as scalable (px None)."""
    page = fetch_public(http, site, MAX_HTML_BYTES)
    if page is None or "html" not in page.headers.get("content-type", "html").lower():
        return None
    candidates = [i for i in parse_icons(str(page.url), page.text) if i["size"] >= min_px][:tries]
    for icon in candidates:
        resp = fetch_public(http, icon["href"], MAX_ICON_BYTES)
        if resp is None:
            continue
        if _is_svg(resp.content):
            return icon["href"], None
        px = longest_side(resp.content)
        if px and px >= min_px:
            return icon["href"], px
    return None


def favicon(http: MediaHttp, site: str, min_px: int = 64, size: int = 128) -> tuple[str, int] | None:
    """Google faviconV2 at `size`; accepted only when the image that comes back is >= min_px (the service answers tiny placeholders otherwise)."""
    if not is_public_http_url(site):
        return None
    url = f"{FAVICON}?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url={quote(site, safe='')}&size={size}"
    try:
        resp = http.web.get(url, max_bytes=MAX_ICON_BYTES)
    except HttpError:
        return None
    px = longest_side(resp.content) if resp.status_code == 200 else None
    return (url, px) if px and px >= min_px else None
