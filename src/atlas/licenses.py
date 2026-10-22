"""arXiv licence classification. Only CC BY, CC BY-SA and CC0 papers may have derived images re-served."""

from __future__ import annotations

DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "http://creativecommons.org/licenses/by/",
    "http://creativecommons.org/licenses/by-sa/",
    "http://creativecommons.org/publicdomain/zero/",
)


def _canon(url: str) -> str:
    url = url.strip()
    if url.startswith("https://"):
        url = "http://" + url[len("https://"):]
    if not url.endswith("/"):
        url += "/"
    return url


def allows_redistribution(license_url: str | None, allowed_prefixes: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_PREFIXES) -> bool:
    """True only for an explicit allow-list match. Missing, arXiv's non-exclusive licence, and any
    NC/ND variant (by-nc, by-nc-sa, by-nc-nd, by-nd) are rejected."""
    if not license_url:
        return False
    canon = _canon(license_url)
    return any(canon.startswith(p) for p in allowed_prefixes)
