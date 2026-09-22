"""smart-engine/pdf_to_word.py — PDF → editable Word (.docx) via pdf2docx."""

from __future__ import annotations
import io
import os
import tempfile


def convert_pdf_to_word(pdf_bytes: bytes) -> bytes:
    """
    Convert a text-based PDF into .docx using pdf2docx (open source).
    Scanned PDFs are NOT supported here — use OCR endpoint instead.
    """
    from pdf2docx import Converter  # pip install pdf2docx

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "in.pdf")
        docx_path = os.path.join(tmp, "out.docx")

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        cv = Converter(pdf_path)
        try:
            cv.convert(docx_path, start=0, end=None)
        finally:
            cv.close()

        with open(docx_path, "rb") as f:
            return f.read()
