from __future__ import annotations

import io

import pytest
from PIL import Image

from atlas.media.images import ImageError, longest_side, normalise_image
from media_fakes import make_image


def decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def test_large_landscape_fits_256_and_keeps_aspect():
    n = normalise_image(make_image(1200, 600, "JPEG"))
    assert (n.width, n.height) == (256, 128) and n.mime == "image/webp" and n.ext == "webp"
    img = decode(n.data)
    assert img.format == "WEBP" and img.size == (256, 128)


def test_portrait_is_limited_by_height_not_cropped():
    n = normalise_image(make_image(500, 700, "JPEG"))
    assert (n.width, n.height) == (183, 256)


def test_small_images_are_never_upscaled():
    n = normalise_image(make_image(64, 40))
    assert (n.width, n.height) == (64, 40)


def test_alpha_is_kept():
    img = Image.new("RGBA", (300, 300), (255, 0, 0, 0))
    img.putpixel((10, 10), (0, 0, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = decode(normalise_image(buf.getvalue()).data)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 0                      # transparent stays transparent


def test_palette_png_with_transparency_counts_as_alpha():
    img = Image.new("P", (100, 100), 0)
    img.info["transparency"] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG", transparency=0)
    assert decode(normalise_image(buf.getvalue()).data).mode == "RGBA"


def test_opaque_images_do_not_grow_an_alpha_channel():
    assert decode(normalise_image(make_image(300, 300, "JPEG")).data).mode == "RGB"


def test_exif_rotation_is_applied():
    img = Image.new("RGB", (400, 200), (10, 200, 10))
    exif = Image.Exif()
    exif[0x0112] = 6                                          # rotate 90 cw when displayed
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    n = normalise_image(buf.getvalue())
    assert n.height > n.width


def test_same_input_gives_same_bytes_and_hash():
    src = make_image(800, 800, "JPEG")
    a, b = normalise_image(src), normalise_image(src)
    assert a.data == b.data and a.sha256 == b.sha256 and len(a.sha256) == 64


@pytest.mark.parametrize("junk", [b"", b"<html><body>Not Found</body></html>", b'<svg xmlns="http://www.w3.org/2000/svg"/>', b"\x89PNG\r\n\x1a\ntruncated"])
def test_non_images_raise(junk):
    with pytest.raises(ImageError):
        normalise_image(junk)


def test_pixel_bomb_is_refused_before_decoding():
    bomb = io.BytesIO()
    Image.new("1", (7000, 7000)).save(bomb, format="PNG")      # 49 Mpx, a few KB on the wire
    assert len(bomb.getvalue()) < 100_000
    with pytest.raises(ImageError):
        normalise_image(bomb.getvalue())


def test_longest_side():
    assert longest_side(make_image(90, 40)) == 90 and longest_side(b"nope") is None
