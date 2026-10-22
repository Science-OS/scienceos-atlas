from __future__ import annotations

import io

import httpx
import pytest

from atlas.http import HttpError, PoliteClient
from atlas.thumbs import make_thumbnail, render_first_page_jpeg, thumb_path
from conftest import mock_transport

pdfium = pytest.importorskip("pypdfium2")
PIL = pytest.importorskip("PIL")


def tiny_pdf(width=300, height=400) -> bytes:
    doc = pdfium.PdfDocument.new()
    doc.new_page(width, height)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_render_first_page_is_600px_jpeg():
    from PIL import Image

    jpeg = render_first_page_jpeg(tiny_pdf(), 600, 78)
    assert jpeg[:3] == b"\xff\xd8\xff"
    img = Image.open(io.BytesIO(jpeg))
    assert img.width == 600 and img.height == 800  # aspect preserved


def test_make_thumbnail_downloads_through_client_and_writes_file(tmp_path, clock):
    pdf = tiny_pdf()
    urls = []

    def handler(req: httpx.Request) -> httpx.Response:
        urls.append(str(req.url))
        return httpx.Response(200, content=pdf)

    c = PoliteClient("ua", 3.0, transport=mock_transport(handler), sleep=clock.sleep, clock=clock.now)
    out = thumb_path(tmp_path, "2610.27576")
    n = make_thumbnail(c, "https://export.arxiv.org/pdf", "2610.27576", out, width_px=600, quality=78, max_pdf_bytes=10**7)
    assert out.exists() and out.stat().st_size == n > 0
    assert urls == ["https://export.arxiv.org/pdf/2610.27576"]


def test_non_pdf_or_error_raises(tmp_path, clock):
    c = PoliteClient("ua", 0.0, max_retries=0, transport=mock_transport(lambda r: httpx.Response(200, content=b"<html>")),
                     sleep=clock.sleep, clock=clock.now)
    with pytest.raises(RuntimeError):
        make_thumbnail(c, "https://x", "2610.00001", tmp_path / "a.jpg", width_px=600, quality=70, max_pdf_bytes=10**6)


def test_oversized_pdf_is_refused(tmp_path, clock):
    c = PoliteClient("ua", 0.0, max_retries=0, transport=mock_transport(lambda r: httpx.Response(200, content=b"%PDF" + b"0" * 5000)),
                     sleep=clock.sleep, clock=clock.now)
    with pytest.raises(HttpError):
        make_thumbnail(c, "https://x", "2610.00001", tmp_path / "a.jpg", width_px=600, quality=70, max_pdf_bytes=1000)
