"""
Universal file router.
Determines which engine to use based on file type and complexity.
"""
import os
from typing import Tuple, Dict, Any

def detect_engine(file_ext: str, content: bytes, filename: str) -> Tuple[str, bool]:
    """
    Returns (engine_choice, use_unstructured).
    
    engine_choice: "excel", "csv", "docx", "pdf", "image", "text"
    use_unstructured: True if Unstructured should be used for this file.
    
    Strategy:
      - Excel/CSV: Try native first, fall back to Unstructured for complex workbooks.
      - DOCX: Native python-docx is usually sufficient.
      - PDF: Use Unstructured if complex layout or scanned; else native PyMuPDF.
      - Images: Always use Unstructured for OCR + layout.
      - Text: Native always.
    """
    ext = file_ext.lower()
    
    if ext in ("xlsx", "xlsm", "xls"):
        # Try native openpyxl first. If the workbook has complex merged cells,
        # multiple tables per sheet, or looks like a form, use Unstructured.
        if _is_complex_excel(content):
            return "excel", True
        return "excel", False
    
    elif ext == "csv":
        # CSV is always simple; native pandas is faster and sufficient.
        return "csv", False
    
    elif ext == "docx":
        # Native python-docx handles most Word docs. Use Unstructured for
        # docs with complex tables/layouts.
        if _is_complex_docx(content):
            return "docx", True
        return "docx", False
    
    elif ext == "pdf":
        # Check if PDF has extractable text.
        try:
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            text_len = sum(len(p.get_text("text")) for p in doc)
            # If very little text, it's scanned → use Unstructured.
            if text_len < 200 and len(doc) > 0:
                return "pdf", True
            # If text is present but layout looks complex (tables, columns),
            # use Unstructured for better structure.
            if _is_complex_pdf(doc):
                return "pdf", True
            return "pdf", False
        except Exception:
            return "pdf", True  # Fallback to Unstructured for weird PDFs
    
    elif ext in ("png", "jpg", "jpeg", "tiff", "bmp", "webp"):
        # Always use Unstructured for images (OCR + layout detection).
        return "image", True
    
    else:
        # Plain text, JSON, etc.
        return "text", False


def _is_complex_excel(content: bytes) -> bool:
    """Check for complex workbook features that native openpyxl struggles with."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
        for ws in wb.worksheets:
            # Check for merged cells (common in form-style invoices)
            if ws.merged_cells.ranges:
                return True
            # Check for multiple potential tables (non-contiguous data)
            # This is a heuristic; could be refined.
            rows = list(ws.iter_rows(values_only=True))
            if len(rows) > 10:
                # Check for blank row gaps suggesting multiple tables
                blank_rows = sum(1 for r in rows if all(c is None for c in r))
                if blank_rows > 2:
                    return True
        return False
    except Exception:
        return False


def _is_complex_docx(content: bytes) -> bool:
    """Check for complex DOCX features."""
    try:
        import docx
        d = docx.Document(io.BytesIO(content))
        # Multiple tables suggest a structured form
        if len(d.tables) > 2:
            return True
        return False
    except Exception:
        return False


def _is_complex_pdf(doc) -> bool:
    """Heuristic for complex PDF layouts."""
    try:
        # Check for multiple columns or tables
        for page in doc[:3]:  # Check first 3 pages
            blocks = page.get_text("blocks")
            if len(blocks) > 50:  # Dense text suggests tables/columns
                return True
        return False
    except Exception:
        return False
