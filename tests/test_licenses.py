from __future__ import annotations

import pytest

from atlas.licenses import allows_redistribution


@pytest.mark.parametrize("url", [
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "https://creativecommons.org/licenses/by/4.0",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
])
def test_allowed(url):
    assert allows_redistribution(url)


@pytest.mark.parametrize("url", [
    None,
    "",
    "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
    "http://creativecommons.org/licenses/by-nc/4.0/",
    "http://creativecommons.org/licenses/by-nc-sa/4.0/",
    "http://creativecommons.org/licenses/by-nc-nd/4.0/",
    "http://creativecommons.org/licenses/by-nd/4.0/",
    "http://example.com/licenses/by/4.0/",
    "http://creativecommons.org/licenses/byx/4.0/",
])
def test_rejected(url):
    assert not allows_redistribution(url)
