"""A fake internet for the media tests: Wikidata, Commons, ROR, GitHub, Bluesky, homepages, the favicon service and the Supabase
work queue, answering from the recorded responses in tests/data/media (captured 2026-10-24) plus generated images."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote

import httpx
from PIL import Image

from atlas.config import Config, load_config
from atlas.media.cascade import Cascade
from atlas.media.sources import MediaHttp
from atlas.media.store import MediaStore

FIX = Path(__file__).parent / "data" / "media"
BASE = "https://science-os.github.io/scienceos-atlas"
NOW = "2026-10-24T06:00:00Z"

PAGE = {"orcid": "0000-0002-7101-9767", "openalex": "A5068055706", "qid": "Q7356570", "name": "Roderic D. M. Page"}
LECUN = {"orcid": "0000-0002-1992-2684", "qid": "Q3571662", "name": "Yann LeCun"}
ETH = {"ror": "05a28rw58", "qid": "Q11942"}


def fx(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def make_image(w: int = 400, h: int = 300, fmt: str = "PNG", mode: str = "RGB") -> bytes:
    img = Image.new(mode, (w, h), (0, 0, 0, 0) if mode == "RGBA" else (200, 40, 40))   # RGBA: a transparent background like a real logo
    if mode == "RGBA":
        img.paste((200, 40, 40, 255), (w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def imageinfo_payload(title: str, license_short: str | None, *, artist: str | None = "A. Photographer", credit: str | None = None,
                      license_url: str | None = None, non_free: bool = False) -> dict:
    """Same shape as the recorded Commons responses; used to vary the licence (non-free, missing, no artist)."""
    meta: dict = {}
    for k, v in (("LicenseShortName", license_short), ("LicenseUrl", license_url), ("Artist", artist), ("Credit", credit),
                 ("NonFree", "true" if non_free else None)):
        if v is not None:
            meta[k] = {"value": v, "source": "commons-desc-page"}
    return {"batchcomplete": "", "query": {"pages": {"1": {"pageid": 1, "title": "File:" + title, "imageinfo": [
        {"descriptionurl": "https://commons.wikimedia.org/wiki/File:" + title.replace(" ", "_"), "extmetadata": meta}]}}}}


class FakeNet:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.search: dict[tuple[str, str], dict] = {
            ("P496", PAGE["orcid"]): fx("wikidata_search_orcid_page.json"),
            ("P10283", PAGE["openalex"]): fx("wikidata_search_openalex_page.json"),
            ("P496", LECUN["orcid"]): fx("wikidata_search_orcid_lecun.json"),
        }
        self.entities = {"Q7356570": fx("wikidata_entity_page.json")["entities"]["Q7356570"],
                         "Q3571662": fx("wikidata_entity_lecun.json")["entities"]["Q3571662"],
                         "Q11942": fx("wikidata_entity_eth.json")["entities"]["Q11942"]}
        self.imageinfo = {"Roderic D M Page.jpg": fx("commons_info_page.json"),
                          "Yann LeCun - 2025 (cropped).jpg": fx("commons_info_lecun.json"),
                          "ETH Zürich Logo black.svg": fx("commons_info_eth.json")}
        self.images: dict[str, tuple[bytes, str]] = {   # Special:FilePath name (underscores) -> (bytes, content-type)
            "Roderic_D_M_Page.jpg": (make_image(500, 700, "JPEG"), "image/jpeg"),
            "Yann_LeCun_-_2025_(cropped).jpg": (make_image(300, 330, "JPEG"), "image/jpeg"),
            "ETH_Zürich_Logo_black.svg": (make_image(330, 56, "PNG", "RGBA"), "image/png"),   # Commons renders SVG to PNG
        }
        self.etag = 'W/"v1"'
        self.ror = {"05a28rw58": fx("ror_eth.json")}
        self.github: dict[str, int] = {"rdmpage": 302}
        self.bluesky = {"rdmpage.bsky.social": fx("bsky_profile_page.json")}
        self.redirects: dict[str, str] = {}           # url -> Location (homepages)
        self.pages: dict[str, str] = {"https://ethz.ch/": (FIX / "homepage_eth_head.html").read_text(encoding="utf-8")}
        self.icons: dict[str, bytes] = {"https://ethz.ch/etc/designs/ethz/img/icons/ETH-APP-Icons-Theme-white/192-xxxhpdi.png": make_image(192, 192)}
        self.favicon: bytes | None = None
        self.down: dict[str, int] = {}                # host -> forced status
        self.queue_rows: list[dict] = []
        self.queue_raw: list | None = None            # returned verbatim (malformed rows)
        self.queue_status = 200

    # -- helpers
    def hits(self, host: str | None = None, contains: str | None = None, method: str | None = None) -> list[httpx.Request]:
        return [r for r in self.requests if (host is None or r.url.host == host) and (contains is None or contains in str(r.url))
                and (method is None or r.method == method)]

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        host, path, q = req.url.host, req.url.path, {k: v[0] for k, v in parse_qs(req.url.query.decode()).items()}
        if host in self.down:
            return httpx.Response(self.down[host])
        if host == "www.wikidata.org":
            return self._wikidata(q)
        if host == "commons.wikimedia.org" and path == "/w/api.php":
            title = unquote(q["titles"]).removeprefix("File:")
            return httpx.Response(200, json=self.imageinfo.get(title, {"query": {"pages": {"-1": {"title": "File:" + title, "missing": ""}}}}))
        if host == "commons.wikimedia.org" and path.startswith("/wiki/Special:FilePath/"):
            return httpx.Response(302, headers={"Location": f"https://upload.wikimedia.org/render/{quote(path.rsplit('/', 1)[1])}"})
        if host == "upload.wikimedia.org":
            name = unquote(path.rsplit("/", 1)[1])
            if name not in self.images:
                return httpx.Response(404)
            if req.headers.get("if-none-match") == self.etag:
                return httpx.Response(304, headers={"ETag": self.etag})
            data, ctype = self.images[name]
            return httpx.Response(200, content=data, headers={"Content-Type": ctype, "ETag": self.etag})
        if host == "api.ror.org":
            rec = self.ror.get(path.rsplit("/", 1)[1])
            return httpx.Response(200, json=rec) if rec else httpx.Response(404)
        if host == "public.api.bsky.app":
            rec = self.bluesky.get(q.get("actor", ""))
            return httpx.Response(200, json=rec) if rec else httpx.Response(400, json={"error": "InvalidRequest"})
        if host == "github.com" and path.endswith(".png"):
            status = self.github.get(path[1:-4], 404)
            return httpx.Response(status, headers={"Location": "https://avatars.githubusercontent.com/u/1"} if status == 302 else {})
        if host == "t3.gstatic.com":
            return httpx.Response(200, content=self.favicon, headers={"Content-Type": "image/png"}) if self.favicon else httpx.Response(404)
        if path.startswith("/rest/v1/"):
            return self._queue(req, q)
        url = str(req.url.copy_with(query=None)).rstrip("/")      # 'https://x.org' and 'https://x.org/' are the same page
        norm = lambda d: {k.rstrip("/"): v for k, v in d.items()}  # noqa: E731
        if url in norm(self.redirects):
            return httpx.Response(302, headers={"Location": norm(self.redirects)[url]})
        if url in norm(self.pages):
            return httpx.Response(200, text=norm(self.pages)[url], headers={"Content-Type": "text/html; charset=utf-8"})
        if url in norm(self.icons):
            return httpx.Response(200, content=norm(self.icons)[url], headers={"Content-Type": "image/png"})
        return httpx.Response(404)

    def _wikidata(self, q: dict) -> httpx.Response:
        if q.get("action") == "query":
            prop, _, value = q["srsearch"].removeprefix("haswbstatement:").partition("=")
            return httpx.Response(200, json=self.search.get((prop, value.upper()), fx("wikidata_search_none.json")))
        ent = self.entities.get(q["ids"], {"id": q["ids"], "missing": ""})
        return httpx.Response(200, json={"entities": {q["ids"]: ent}})

    def _queue(self, req: httpx.Request, q: dict) -> httpx.Response:
        if self.queue_status != 200:
            return httpx.Response(self.queue_status, json={"message": "relation does not exist"})
        if self.queue_raw is not None:
            return httpx.Response(200, json=self.queue_raw)
        after = int(q["id"].removeprefix("gt."))
        rows = [r for r in sorted(self.queue_rows, key=lambda r: r["id"]) if r["id"] > after][: int(q["limit"])]
        return httpx.Response(200, json=rows)


def make_cfg(**media) -> Config:
    cfg = load_config()
    cfg.media.min_interval_s = 0
    for k, v in media.items():
        setattr(cfg.media, k, v)
    return cfg


def make_http(cfg: Config, net: FakeNet) -> MediaHttp:
    return MediaHttp.create(cfg, transport=httpx.MockTransport(net.handler), sleep=lambda s: None, resolve_dns=False)


def cascade_world(tmp_path: Path, **media_cfg):
    cfg, net = make_cfg(**media_cfg), FakeNet()
    store = MediaStore(tmp_path / "media-store", BASE)
    http = make_http(cfg, net)
    return cfg, net, http, store, Cascade(cfg, http, store, NOW, {})
