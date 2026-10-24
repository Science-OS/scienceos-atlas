"""Paper first-page thumbnails: every one ever rendered is moved into the store (bytes under b/, a catalogue entry per paper)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import Config
from ..licenses import allows_redistribution
from ..store import Store
from ..thumbs import thumb_path
from . import keys as K
from .catalogue import asset, first_asset
from .images import ImageError, image_dimensions
from .store import MediaStore

log = logging.getLogger("atlas.media.papers")

_BY = re.compile(r"/licenses/(by|by-sa)/(\d\.\d)", re.IGNORECASE)
_ZERO = re.compile(r"/publicdomain/zero/(\d\.\d)", re.IGNORECASE)


def license_label(url: str | None) -> tuple[str | None, str | None]:
    """arXiv licence URL -> ('CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0')."""
    if not url:
        return None, None
    canon = "https://" + re.sub(r"^https?://", "", url.strip()).rstrip("/")
    if m := _BY.search(canon):
        return f"CC {m.group(1).upper()} {m.group(2)}", canon
    if m := _ZERO.search(canon):
        return f"CC0 {m.group(1)}", canon
    return None, canon


def thumbnail_url(media: MediaStore, arxiv_id: str) -> str | None:
    """Hosted thumbnail URL of a paper, if the catalogue has one."""
    a = first_asset(media.read_entry("paper", K.paper_key(arxiv_id)), "thumbnail")
    return a["url"] if a and a.get("hosted") else None


def restore_thumbnail(media: MediaStore, arxiv_id: str, dest: Path) -> bool:
    """Copy a hosted thumbnail back into state/thumbs (the Actions cache can lose that directory). True when dest now exists."""
    if dest.exists():
        return True
    a = first_asset(media.read_entry("paper", K.paper_key(arxiv_id)), "thumbnail")
    data = media.read_blob(a["sha256"], "jpg") if a and a.get("hosted") and a.get("sha256") else None
    if data is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def ingest_thumbnails(state: Store, media: MediaStore, thumbs_dir: Path, cfg: Config, now: str) -> dict[str, Any]:
    """Catalogue and store the thumbnails the thumbs stage rendered. Licence is re-checked against the paper; the bytes are kept as they are."""
    stats = {"ingested": 0, "already": 0, "not_licensed": 0, "no_paper": 0, "bad_image": 0}
    for aid, row in sorted(state.thumb_rows().items()):
        path = thumb_path(thumbs_dir, aid)
        if row["status"] != "ok" or not path.exists():
            continue
        key = K.paper_key(aid)
        if first_asset(media.read_entry("paper", key), "thumbnail"):
            stats["already"] += 1
            continue
        sp = state.get_paper(aid)
        if sp is None:
            stats["no_paper"] += 1
            continue
        if not allows_redistribution(sp.record.license, cfg.thumbs.allowed_license_prefixes):
            stats["not_licensed"] += 1
            continue
        data = path.read_bytes()
        try:
            width, height = image_dimensions(data)
        except ImageError:
            stats["bad_image"] += 1
            continue
        sha, _ = media.put_bytes(data, "jpg")
        label, lic_url = license_label(sp.record.license)
        names = [a["name"] for a in sp.record.authors if a.get("name")]
        credit = (f"{names[0]} et al." if len(names) > 1 else names[0] if names else "The authors") + f", arXiv:{aid}"
        abs_url = f"https://arxiv.org/abs/{aid}"
        media.write_entry("paper", [key], {"arxiv": aid}, [asset(
            "thumbnail", media.blob_url(sha, "jpg"), True, "arxiv_pdf", now, sha256=sha, mime="image/jpeg", width=width, height=height,
            sourceUrl=f"https://arxiv.org/pdf/{aid}", license=label, licenseUrl=lic_url, attribution=credit[:512], attributionUrl=abs_url,
            confidence=1.0)], now)
        stats["ingested"] += 1
    return stats
