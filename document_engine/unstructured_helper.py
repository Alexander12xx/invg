"""
Unstructured helper — fully optional.
If the `unstructured` package is not installed, returns [] so the pipeline
falls back to native engines. Never raises.
"""
import io
import logging
from typing import Any, Dict, List

log = logging.getLogger("altech-unstructured")

try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import (
        Table, Title, ListItem, NarrativeText
    )
    UNSTRUCTURED_AVAILABLE = True
except Exception as _e:
    UNSTRUCTURED_AVAILABLE = False
    partition = None
    Table = Title = ListItem = NarrativeText = None
    log.warning(f"Unstructured not available (will use native engines): {_e}")


def is_available() -> bool:
    return UNSTRUCTURED_AVAILABLE


def partition_file(content: bytes, filename: str,
                   strategy: str = "auto") -> List[Dict[str, Any]]:
    """
    Returns a list of element dicts. Empty list on any failure.
    strategy: "auto" | "fast" | "hi_res" | "ocr_only"
    """
    if not UNSTRUCTURED_AVAILABLE or partition is None:
        return []

    try:
        elements = partition(
            file=io.BytesIO(content),
            metadata_filename=filename,
            strategy=strategy,
            include_page_breaks=True,
            languages=["eng"],
            pdf_infer_table_structure=True,
        )
    except Exception as e:
        log.error(f"Unstructured partition failed for {filename}: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for el in elements:
        el_type = type(el).__name__
        role = "text"
        if Table is not None and isinstance(el, Table):
            role = "table"
        elif Title is not None and isinstance(el, Title):
            role = "header"
        elif ListItem is not None and isinstance(el, ListItem):
            role = "list_item"
        elif NarrativeText is not None and isinstance(el, NarrativeText):
            role = "narrative"

        coords = None
        try:
            if getattr(el.metadata, "coordinates", None):
                coords = el.metadata.coordinates.points
        except Exception:
            coords = None

        table_html = None
        try:
            if role == "table" and hasattr(el.metadata, "text_as_html"):
                table_html = el.metadata.text_as_html
        except Exception:
            table_html = None

        page_num = None
        try:
            page_num = getattr(el.metadata, "page_number", None)
        except Exception:
            pass

        out.append({
            "text": getattr(el, "text", "") or "",
            "type": el_type,
            "role": role,
            "page_number": page_num,
            "coordinates": coords,
            "table_html": table_html,
            "source": "unstructured",
            "strategy": strategy,
        })
    return out
