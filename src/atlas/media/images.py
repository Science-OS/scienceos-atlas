"""Normalise any fetched image to a small WebP (longest side <= max_px, alpha kept, never cropped, never upscaled)."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 40_000_000  # decompression-bomb guard, applied before the pixels are decoded


class ImageError(ValueError):
    """The bytes are not an image Pillow can decode (an SVG source, HTML error page, truncated file...)."""


@dataclass(frozen=True)
class Normalised:
    data: bytes
    mime: str
    ext: str
    width: int
    height: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def _open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise ImageError("image has too many pixels")
        img.load()
        return img
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageError(f"not a decodable image: {exc}") from exc


def has_alpha(img: Image.Image) -> bool:
    return "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info)


def normalise_image(data: bytes, max_px: int = 256, quality: int = 80) -> Normalised:
    """Decode, fix EXIF rotation, shrink to fit max_px x max_px, encode WebP. Deterministic for the same input bytes."""
    img = _open(data)
    try:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA" if has_alpha(img) else "RGB")
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality, method=6)
    except (OSError, ValueError) as exc:
        raise ImageError(f"cannot re-encode image: {exc}") from exc
    return Normalised(buf.getvalue(), "image/webp", "webp", img.width, img.height)


def image_dimensions(data: bytes) -> tuple[int, int]:
    """(width, height) of encoded bytes, without keeping the pixels."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageError(f"not a decodable image: {exc}") from exc


def longest_side(data: bytes) -> int | None:
    """Longest side in px, or None when the bytes are not a raster image."""
    try:
        w, h = image_dimensions(data)
    except ImageError:
        return None
    return max(w, h)
