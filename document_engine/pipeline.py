"""
Universal document pipeline.

- Uses native app.py helpers for all simple files.
- Uses Unstructured only for complex files; canonicalization and role
  classification still come from app.py so behavior stays identical.
- Always falls back to native if Unstructured fails or is missing.
"""
from __future__ import annotations

import io
import time
import logging
from typing import Any, Dict, List

from .detector import detect_engine
from .unstructured_helper import partition_file, is_available

log = logging.getLogger("altech-pipeline")


def analyze_document(content: bytes, filename: str, ext: str,
                     company_id: int = 0) -> Dict[str, Any]:
    # Import shared helpers from app.py (single source of truth)
    try:
        from app import (
            extract_excel, extract_pdf, extract_docx, ocr_image,
            text_to_structure, canonicalize, build_columns,
            infer_document_type, parse_number, ENGINE_VERSION,
        )
    except Exception as e:
        log.exception(f"Unable to import helpers from app.py: {e}")
        return {
            "success": False,
            "engine_version": "unknown",
            "error": "engine helpers unavailable",
            "columns": [], "items": [], "raw_items": [],
        }

    started = time.perf_counter()
    ext = (ext or "").lower()
    engine_choice, use_unstructured = detect_engine(ext, content, filename)

    structures: List[Dict[str, Any]] = []
    raw_text = ""
    method = engine_choice
    unstructured_used = False

    # ---------------------------------------------------------------
    # EXCEL / CSV
    # ---------------------------------------------------------------
    if engine_choice in ("excel", "csv"):
        if use_unstructured and is_available():
            elements = partition_file(content, filename, strategy="fast")
            if elements:
                structures = _unstructured_to_structures(elements, build_columns)
                method = f"{engine_choice}_unstructured"
                unstructured_used = True
        if not structures:
            structures = extract_excel(content, ext)
            method = f"{engine_choice}_native"

    # ---------------------------------------------------------------
    # DOCX
    # ---------------------------------------------------------------
    elif engine_choice == "docx":
        if use_unstructured and is_available():
            elements = partition_file(content, filename, strategy="fast")
            if elements:
                structures = _unstructured_to_structures(elements, build_columns)
                method = "docx_unstructured"
                unstructured_used = True
        if not structures:
            raw_text = extract_docx(content)
            structures = [text_to_structure(raw_text)] if raw_text else []
            method = "docx_native"

    # ---------------------------------------------------------------
    # PDF
    # ---------------------------------------------------------------
    elif engine_choice == "pdf":
        if use_unstructured and is_available():
            elements = partition_file(content, filename, strategy="hi_res")
            if elements:
                structures = _unstructured_to_structures(elements, build_columns)
                method = "pdf_unstructured_hi_res"
                unstructured_used = True
        if not structures:
            raw_text, native_method = extract_pdf(content)
            structures = [text_to_structure(raw_text)] if raw_text else []
            method = f"{native_method}_native"

    # ---------------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------------
    elif engine_choice == "image":
        if is_available():
            elements = partition_file(content, filename, strategy="hi_res")
            if elements:
                structures = _unstructured_to_structures(elements, build_columns)
                method = "image_unstructured"
                unstructured_used = True
        if not structures:
            try:
                from PIL import Image
                raw_text = ocr_image(Image.open(io.BytesIO(content)))
                structures = [text_to_structure(raw_text)] if raw_text else []
                method = "image_tesseract"
            except Exception as e:
                log.warning(f"Image OCR fallback failed: {e}")
                method = "image_failed"

    # ---------------------------------------------------------------
    # TEXT / JSON / unknown
    # ---------------------------------------------------------------
    else:
        raw_text = content.decode("utf-8", errors="ignore")
        structures = [text_to_structure(raw_text)] if raw_text else []
        method = "text_native"

    # ---------------------------------------------------------------
    # No structures → report failure
    # ---------------------------------------------------------------
    if not structures:
        return {
            "success": False,
            "engine_version": ENGINE_VERSION,
            "error": "No readable structure was detected.",
            "columns": [], "items": [], "raw_items": [],
            "raw_text": raw_text[:20000],
            "diagnostics": {
                "stage": "ingestion", "file": filename,
                "method": method, "trust_score": 0,
                "confidence_band": "review",
                "unstructured_used": unstructured_used,
            },
        }

    # ---------------------------------------------------------------
    # Canonicalize exactly like app.py did
    # ---------------------------------------------------------------
    primary_columns = structures[0].get("columns", [])
    merged_items = []
    sheet_names = []
    for s in structures:
        sheet_names.append(s.get("sheet") or "")
        for row in s.get("items", []):
            r = dict(row)
            r["_sheet"] = s.get("sheet") or None
            merged_items.append(r)

    items = canonicalize({"columns": primary_columns, "items": merged_items})

    total_qty = sum(parse_number(r.get("quantity")) or 0 for r in items)
    total_amount = sum(parse_number(r.get("total")) or 0 for r in items)

    trust = (sum(r["_meta"]["confidence"] for r in items) / len(items)
             if items else 0.25)
    anomalies = [
        {"row": r["_meta"]["row_index"], "code": w}
        for r in items for w in r["_meta"]["warnings"]
    ]
    trust = round(max(0, min(1, trust - 0.03 * len(anomalies))), 3)

    return {
        "success": True,
        "engine_version": ENGINE_VERSION,
        "filename": filename,
        "file_type": ext,
        "extraction_method": method,
        "document_type": infer_document_type(raw_text, items, filename),
        "columns": primary_columns,
        "items": items,
        "raw_items": items,
        "item_count": len(items),
        "total_quantity": int(total_qty) if total_qty else 0,
        "total_amount": round(total_amount, 2),
        "raw_text": raw_text[:20000],
        "diagnostics": {
            "sheets": sheet_names,
            "trust_score": trust,
            "confidence_band": ("high" if trust >= 0.85
                                else "medium" if trust >= 0.65
                                else "review"),
            "anomalies": anomalies,
            "engine_selected": engine_choice,
            "unstructured_used": unstructured_used,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "company_id": company_id,
    }


def _unstructured_to_structures(elements: List[Dict[str, Any]],
                                build_columns) -> List[Dict[str, Any]]:
    """
    Convert Unstructured elements into the same structure format app.py uses.
    Tables → real columns/items.
    Text lines → single raw_text column.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None

    structures: List[Dict[str, Any]] = []
    text_lines: List[str] = []

    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            continue

        if (el.get("role") == "table"
                and el.get("table_html")
                and BeautifulSoup is not None):
            try:
                soup = BeautifulSoup(el["table_html"], "html.parser")
                headers: List[str] = []
                rows: List[List[str]] = []
                for tr in soup.find_all("tr"):
                    cells = [td.get_text(strip=True)
                             for td in tr.find_all(["td", "th"])]
                    if cells and not headers:
                        headers = cells
                    elif cells:
                        rows.append(cells)
                if not headers:
                    continue
                columns = build_columns(headers)
                items = []
                for row in rows:
                    record: Dict[str, Any] = {}
                    for j, col in enumerate(columns):
                        record[col["key"]] = row[j] if j < len(row) else None
                    items.append(record)
                structures.append({
                    "columns": columns,
                    "items": items,
                    "sheet": f"Table_p{el.get('page_number', '?')}",
                })
            except Exception as e:
                log.warning(f"Table parse failed: {e}")
            continue

        text_lines.append(text)

    if text_lines and not structures:
        from app import text_to_structure
        structures.append(text_to_structure("\n".join(text_lines)))

    return structures
