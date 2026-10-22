"""First-page JPEG thumbnails, ONLY for papers whose arXiv licence allows redistribution (CC BY, CC BY-SA, CC0).
Uses pypdfium2 (optional extra `thumbs`); everything degrades to "no thumbnail" when it is missing."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from .http import PoliteClient

log = logging.getLogger("atlas.thumbs")


class ThumbsUnavailable(RuntimeError):
    pass


def render_first_page_jpeg(pdf_bytes: bytes, width_px: int = 600, quality: int = 78) -> bytes:
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ThumbsUnavailable("pypdfium2/Pillow not installed (uv sync --extra thumbs)") from exc
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        page = pdf[0]
        scale = width_px / max(1.0, page.get_width())
        image = page.render(scale=scale).to_pil().convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        return buf.getvalue()
    finally:
        pdf.close()


def thumb_path(thumbs_dir: Path, arxiv_id: str) -> Path:
    return thumbs_dir / f"{arxiv_id.replace('/', '_')}.jpg"


def make_thumbnail(client: PoliteClient, pdf_base: str, arxiv_id: str, out: Path, *, width_px: int, quality: int,
                   max_pdf_bytes: int) -> int:
    """Download the PDF politely (shared arXiv throttle), render page 1, write the JPEG. Returns bytes written."""
    resp = client.get(f"{pdf_base}/{arxiv_id}", max_bytes=max_pdf_bytes)
    if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
        raise RuntimeError(f"PDF unavailable (HTTP {resp.status_code})")
    jpeg = render_first_page_jpeg(resp.content, width_px, quality)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(jpeg)
    return len(jpeg)
