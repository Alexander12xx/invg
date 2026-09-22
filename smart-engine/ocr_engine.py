"""smart-engine/ocr_engine.py — OCR for scanned PDFs & images (Tesseract)."""

from __future__ import annotations
import io
import os
import tempfile


def ocr_image(data: bytes, lang: str = "eng") -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(img, lang=lang)


def ocr_pdf(data: bytes, lang: str = "eng", output: str = "txt"):
    """
    OCR a scanned PDF.
    output: 'txt'                → returns bytes (text)
            'searchable-pdf'     → returns bytes (PDF with invisible text layer)
    """
    import fitz
    import pytesseract
    from PIL import Image

    src = fitz.open(stream=data, filetype="pdf")

    if output == "txt":
        chunks: list[str] = []
        for i, page in enumerate(src):
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=lang)
            chunks.append(f"--- Page {i+1} ---\n{text}")
        src.close()
        return "\n\n".join(chunks).encode("utf-8")

    # Searchable PDF: rasterize + overlay invisible text
    out_doc = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension="pdf", lang=lang)
        out_doc.insert_pdf(fitz.open(stream=pdf_bytes, filetype="pdf"))

    out = io.BytesIO()
    out_doc.save(out, garbage=4, deflate=True)
    out_doc.close(); src.close()
    return out.getvalue()
