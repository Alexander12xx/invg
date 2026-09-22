"""smart-engine/image_tools.py — Image → PDF and image optimisation."""

from __future__ import annotations
import io
from typing import List

from PIL import Image, ImageOps

PAGE_SIZES = {
    "A4":     (595, 842),
    "A3":     (842, 1191),
    "Letter": (612, 792),
    "Legal":  (612, 1008),
}


def _fit(img: Image.Image, box_w: int, box_h: int, margin: int) -> Image.Image:
    img = ImageOps.exif_transpose(img)  # auto-rotate
    avail_w = box_w - margin * 2
    avail_h = box_h - margin * 2
    img.thumbnail((avail_w, avail_h), Image.LANCZOS)

    canvas = Image.new("RGB", (box_w, box_h), "white")
    x = (box_w - img.width) // 2
    y = (box_h - img.height) // 2
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        canvas.paste(img, (x, y), img)
    else:
        canvas.paste(img.convert("RGB"), (x, y))
    return canvas


def images_to_pdf(
    blobs: List[bytes],
    page_size: str = "A4",
    orientation: str = "portrait",
    margin: int = 20,
) -> bytes:
    w, h = PAGE_SIZES.get(page_size, PAGE_SIZES["A4"])
    if orientation == "landscape":
        w, h = h, w

    pages: list[Image.Image] = []
    for b in blobs:
        img = Image.open(io.BytesIO(b))
        pages.append(_fit(img, w, h, margin))

    if not pages:
        raise ValueError("No valid images")

    out = io.BytesIO()
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=150)
    return out.getvalue()


def optimize_image(data: bytes, max_dim: int = 2000, quality: int = 82) -> bytes:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    out = io.BytesIO()
    if img.mode in ("RGBA", "P"):
        img.save(out, "PNG", optimize=True)
    else:
        img.convert("RGB").save(out, "JPEG", quality=quality, optimize=True)
    return out.getvalue()
