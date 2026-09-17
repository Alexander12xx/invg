"""
Universal file-type detector.
Decides whether a file should use the native engine or the Unstructured helper.
"""
import io
import logging
from typing import Tuple

log = logging.getLogger("altech-detector")


def detect_engine(file_ext: str, content: bytes, filename: str) -> Tuple[str, bool]:
    """
    Returns (engine_choice, use_unstructured).

    engine_choice is one of: excel, csv, docx, pdf, image, text
    use_unstructured is True only when the file benefits from layout-aware
    parsing that native engines can't provide.
    """
    ext = (file_ext or "").lower()

    if ext in ("xlsx", "xlsm", "xls"):
        return "excel", _is_complex_excel(content)

    if ext == "csv":
        return "csv", False

    if ext == "docx":
        return "docx", _is_complex_docx(content)

    if ext == "pdf":
        return "pdf", _is_complex_or_scanned_pdf(content)

    if ext in ("png", "jpg", "jpeg", "tiff", "bmp", "webp", "gif"):
        return "image", True

    return "text", False


def _is_complex_excel(content: bytes) -> bool:
    """Excel with merged cells or multiple tables on one sheet."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
        for ws in wb.worksheets:
            if ws.merged_cells.ranges:
                return True
            rows = list(ws.iter_rows(values_only=True))
            if len(rows) > 10:
                blanks = sum(1 for r in rows if all(c is None for c in r))
                if blanks > 2:
                    return True
        return False
    except Exception as e:
        log.warning(f"_is_complex_excel failed: {e}")
        return False


def _is_complex_docx(content: bytes) -> bool:
    """DOCX with multiple tables."""
    try:
        import docx
        d = docx.Document(io.BytesIO(content))
        return len(d.tables) > 2
    except Exception:
        return False


def _is_complex_or_scanned_pdf(content: bytes) -> bool:
    """
    True if the PDF is scanned (no text layer) or has a dense layout
    that a plain text extraction would misparse.
    """
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        if len(doc) == 0:
            return False
        text_len = sum(len(p.get_text("text") or "") for p in doc)
        # Scanned: very little text but pages exist
        if text_len < 200:
            return True
        # Dense layout: many blocks suggests columns/tables
        for page in list(doc)[:3]:
            if len(page.get_text("blocks") or []) > 50:
                return True
        return False
    except Exception as e:
        log.warning(f"_is_complex_or_scanned_pdf failed: {e}")
        return True
