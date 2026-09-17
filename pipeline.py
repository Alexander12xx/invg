"""
Universal document analysis pipeline.
Routes ALL file types through the appropriate engine.
"""
import io
import time
from typing import Dict, Any, List

from .detector import detect_engine
from .unstructured_helper import partition_file
from .canonical import CanonicalDocument, CanonicalItem


def analyze_document(content: bytes, filename: str, ext: str, 
                     company_id: int = 0) -> Dict[str, Any]:
    """
    Universal analysis entry point.
    Handles Excel, CSV, DOCX, PDF, Images, and Text.
    """
    started = time.perf_counter()
    
    engine_choice, use_unstructured = detect_engine(ext, content, filename)
    
    structures = []
    raw_text = ""
    method = engine_choice
    unstructured_elements = []

    # ================================================================
    # EXCEL / CSV
    # ================================================================
    if engine_choice in ("excel", "csv") and not use_unstructured:
        # Fast path: your proven native logic
        structures = extract_excel_native(content, ext)
        method = f"{engine_choice}_native"
        
    elif engine_choice in ("excel", "csv") and use_unstructured:
        # Advanced path: Unstructured handles complex workbooks
        unstructured_elements = partition_file(content, filename, strategy="hi_res")
        if unstructured_elements:
            structures = _unstructured_to_structures(unstructured_elements)
            method = f"{engine_choice}_unstructured"
        else:
            # Fallback to native
            structures = extract_excel_native(content, ext)
            method = f"{engine_choice}_native_fallback"
    
    # ================================================================
    # DOCX
    # ================================================================
    elif engine_choice == "docx" and not use_unstructured:
        raw_text = extract_docx_native(content)
        method = "docx_native"
        
    elif engine_choice == "docx" and use_unstructured:
        unstructured_elements = partition_file(content, filename, strategy="hi_res")
        if unstructured_elements:
            structures = _unstructured_to_structures(unstructured_elements)
            method = "docx_unstructured"
        else:
            raw_text = extract_docx_native(content)
            method = "docx_native_fallback"
    
    # ================================================================
    # PDF
    # ================================================================
    elif engine_choice == "pdf" and not use_unstructured:
        # Fast path: native PyMuPDF
        raw_text, method = extract_pdf_native(content)
        structures = [text_to_structure(raw_text)]
        
    elif engine_choice == "pdf" and use_unstructured:
        # Advanced path: layout-aware extraction
        unstructured_elements = partition_file(content, filename, strategy="hi_res")
        if unstructured_elements:
            structures = _unstructured_to_structures(unstructured_elements)
            method = "pdf_unstructured_hi_res"
        else:
            raw_text, method = extract_pdf_native(content)
            method = "pdf_native_fallback"
    
    # ================================================================
    # IMAGES (always Unstructured for OCR + layout)
    # ================================================================
    elif engine_choice == "image":
        unstructured_elements = partition_file(content, filename, strategy="hi_res")
        if unstructured_elements:
            structures = _unstructured_to_structures(unstructured_elements)
            method = "image_unstructured_ocr"
        else:
            # Fallback to basic Tesseract
            raw_text = ocr_image_native(content)
            method = "image_tesseract_fallback"
    
    # ================================================================
    # TEXT / JSON
    # ================================================================
    else:
        raw_text = content.decode("utf-8", errors="ignore")
        structures = [text_to_structure(raw_text)]
        method = "text_native"
    
    # Normalize everything into Canonical Items
    canonical_items = _normalize_to_canonical(
        structures, unstructured_elements, engine_choice
    )
    
    return {
        "success": True,
        "engine_version": "26.0.0",
        "filename": filename,
        "file_type": ext,
        "extraction_method": method,
        "document_type": "line_item_document",
        "items": [item.dict() for item in canonical_items],
        "raw_text": raw_text[:5000] if raw_text else "",
        "diagnostics": {
            "engine_selected": engine_choice,
            "unstructured_used": len(unstructured_elements) > 0,
            "elapsed": round(time.perf_counter() - started, 3)
        }
    }


def _unstructured_to_structures(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Unstructured elements into your existing structure format.
    Handles tables (via HTML parsing), text lines, and key-value pairs.
    """
    import re
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None
    
    structures = []
    current_items = []
    
    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            continue
        
        # TABLE: parse HTML into rows
        if el.get("role") == "table" and el.get("table_html") and BeautifulSoup:
            soup = BeautifulSoup(el["table_html"], "html.parser")
            table_rows = []
            headers = []
            
            # Extract header row
            for tr in soup.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells and not headers:
                    headers = cells
                elif cells:
                    table_rows.append(cells)
            
            if headers:
                # Build column definitions
                columns = [{"key": f"col_{i}", "label": h, "role": _classify_basic(h)} 
                          for i, h in enumerate(headers)]
                # Build items
                items = []
                for row in table_rows:
                    record = {}
                    for j, col in enumerate(columns):
                        record[col["key"]] = row[j] if j < len(row) else None
                    items.append(record)
                structures.append({"columns": columns, "items": items, 
                                  "sheet": f"Table_p{el.get('page_number', '?')}"})
            continue
        
        # KEY-VALUE PAIR: "Product: Hydrangea"
        if ":" in text and len(text) < 100:
            key, val = text.split(":", 1)
            current_items.append({
                key.strip().lower().replace(" ", "_"): val.strip(),
                "_type": "key_value"
            })
            continue
        
        # TEXT LINE: accumulate for later table detection
        if el.get("role") in ("text", "narrative", "list_item"):
            current_items.append({
                "raw_text": text,
                "_page": el.get("page_number"),
                "_coords": el.get("coordinates"),
            })
    
    # If we have accumulated text lines, try to structure them
    if current_items and not structures:
        structures.append(_lines_to_structure(current_items))
    
    return structures


def _lines_to_structure(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a list of text lines into a structured table.
    Uses coordinate alignment if available.
    """
    # Simple heuristic: find header line, then parse subsequent lines
    # This is where you'd implement your advanced table detection
    items = []
    for line in lines:
        text = line.get("raw_text", "")
        if text:
            items.append({"raw_text": text, "product_name": text[:50]})
    return {
        "columns": [{"key": "raw_text", "label": "Text", "role": None}],
        "items": items,
        "sheet": "Unstructured_Lines"
    }


def _normalize_to_canonical(structures, unstructured_elements, engine_choice) -> List[CanonicalItem]:
    """
    Map all extracted data into the standard CanonicalItem format.
    This is what chat_engine.py will consume.
    """
    # ... (Implementation: take items from structures, map fields,
    #      add _meta with provenance, create CanonicalItem objects)
    # This is the same as before but now handles ALL file types.
    pass
