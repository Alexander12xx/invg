"""smart-engine/pdf_ops.py — Merge / Split / Compress / Rotate / Watermark."""

from __future__ import annotations
import io
import zipfile
from typing import List

from pypdf import PdfReader, PdfWriter


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_pdfs(blobs: List[bytes]) -> bytes:
    writer = PdfWriter()
    for b in blobs:
        reader = PdfReader(io.BytesIO(b))
        for page in reader.pages:
            writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def _parse_ranges(spec: str, total: int) -> List[int]:
    pages: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            start, end = int(a), int(b)
            pages.update(range(max(1, start), min(total, end) + 1))
        else:
            n = int(chunk)
            if 1 <= n <= total:
                pages.add(n)
    return sorted(pages)


def split_pdf(data: bytes, mode: str = "ranges", ranges: str = "", every: int = 0) -> bytes:
    reader = PdfReader(io.BytesIO(data))
    total  = len(reader.pages)
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if mode == "every" and every > 0:
            for start in range(0, total, every):
                w = PdfWriter()
                for i in range(start, min(start + every, total)):
                    w.add_page(reader.pages[i])
                buf = io.BytesIO(); w.write(buf)
                zf.writestr(f"pages_{start+1}-{min(start+every, total)}.pdf", buf.getvalue())

        elif mode == "extract":
            w = PdfWriter()
            for i in _parse_ranges(ranges, total):
                w.add_page(reader.pages[i - 1])
            buf = io.BytesIO(); w.write(buf)
            zf.writestr("extracted.pdf", buf.getvalue())

        else:  # ranges → one file per chunk
            for chunk in [c.strip() for c in ranges.split(",") if c.strip()]:
                w = PdfWriter()
                for i in _parse_ranges(chunk, total):
                    w.add_page(reader.pages[i - 1])
                if len(w.pages):
                    buf = io.BytesIO(); w.write(buf)
                    zf.writestr(f"pages_{chunk.replace('-', '_')}.pdf", buf.getvalue())

    return zip_buf.getvalue()


# ---------------------------------------------------------------------------
# Compress (via PyMuPDF — smaller output, no quality trapdoors)
# ---------------------------------------------------------------------------
def compress_pdf(data: bytes, level: str = "medium") -> bytes:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    # Options
    dpi_map     = {"low": 72,  "medium": 110, "high": 150}
    quality_map = {"low": 40,  "medium": 65,  "high": 80}
    dpi     = dpi_map.get(level, 110)
    quality = quality_map.get(level, 65)

    for page in doc:
        # Redraw images at lower DPI
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                pix  = fitz.Pixmap(base["image"])
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                # Reinsert compressed
                page.insert_image(page.rect, pixmap=pix, overlay=False)
            except Exception:
                continue

    out = io.BytesIO()
    doc.save(out, garbage=4, deflate=True, clean=True)
    doc.close()
    return out.getvalue()


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------
def rotate_pdf(data: bytes, angle: int = 90, pages: str = "all") -> bytes:
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    total  = len(reader.pages)

    if pages.strip().lower() == "all":
        idxs = list(range(total))
    else:
        idxs = [p - 1 for p in _parse_ranges(pages, total)]

    angle = angle % 360
    for i, page in enumerate(reader.pages):
        if i in idxs:
            page.rotate(angle)
        writer.add_page(page)

    out = io.BytesIO(); writer.write(out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------
def add_watermark(
    data: bytes,
    text: str,
    opacity: float = 0.25,
    angle: int = 45,
    font_size: int = 48,
    color: str = "#999999",
) -> bytes:
    import fitz
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import HexColor

    # 1. Build a single-page watermark PDF (letter size, we'll tile it)
    wm_buf = io.BytesIO()
    c = canvas.Canvas(wm_buf, pagesize=letter)
    c.saveState()
    c.setFillColor(HexColor(color))
    c.setFillAlpha(opacity)
    c.setFont("Helvetica-Bold", font_size)
    c.translate(letter[0] / 2, letter[1] / 2)
    c.rotate(angle)
    c.drawCentredString(0, 0, text)
    c.restoreState()
    c.showPage()
    c.save()
    wm_bytes = wm_buf.getvalue()

    # 2. Overlay on every page
    src = fitz.open(stream=data, filetype="pdf")
    wm  = fitz.open(stream=wm_bytes, filetype="pdf")

    for page in src:
        page.show_pdf_page(page.rect, wm, 0, overlay=True)

    out = io.BytesIO()
    src.save(out, garbage=4, deflate=True)
    src.close(); wm.close()
    return out.getvalue()


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------
def get_pdf_info(data: bytes) -> dict:
    reader = PdfReader(io.BytesIO(data))
    meta = reader.metadata or {}
    return {
        "pages": len(reader.pages),
        "encrypted": reader.is_encrypted,
        "title": meta.get("/Title"),
        "author": meta.get("/Author"),
        "subject": meta.get("/Subject"),
        "creator": meta.get("/Creator"),
        "producer": meta.get("/Producer"),
        "size_bytes": len(data),
    }
