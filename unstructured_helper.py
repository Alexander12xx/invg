"""
Unstructured universal helper.
Routes ALL supported file types through Unstructured's partition function.
"""
import io
import logging
from typing import List, Dict, Any, Optional

try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import (
        Table, Title, NarrativeText, ListItem, Text, Image
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning("Unstructured not available. Using native engines only.")

log = logging.getLogger("altech-unstructured")


def partition_file(content: bytes, filename: str, 
                   strategy: str = "auto") -> List[Dict[str, Any]]:
    """
    Universal partitioner using Unstructured.
    
    Handles: Excel, CSV, DOCX, PDF, Images, HTML, and more.
    
    strategy: "auto", "fast", "hi_res", "ocr_only"
    - "auto": dynamically routes each page (best for mixed documents) [citation:2]
    - "fast": rule-based, 100x faster, good for text-only [citation:4]
    - "hi_res": model-based layout detection, best for tables/complex layouts [citation:9]
    - "ocr_only": for scanned images/PDFs
    """
    if not UNSTRUCTURED_AVAILABLE:
        return []
    
    try:
        # Unstructured auto-detects file type from content/filename
        elements = partition(
            file=io.BytesIO(content),
            metadata_filename=filename,
            strategy=strategy,
            include_page_breaks=True,
            languages=["eng"],
            # Enable table structure inference for HTML output
            pdf_infer_table_structure=True,
        )
    except Exception as e:
        log.error(f"Unstructured partition failed for {filename}: {e}")
        return []

    # Convert Unstructured Elements to Canonical Raw format
    canonical_raw = []
    
    for el in elements:
        el_type = type(el).__name__
        
        # Determine semantic role
        role = "text"
        if isinstance(el, Table):
            role = "table"
        elif isinstance(el, Title):
            role = "header"
        elif isinstance(el, ListItem):
            role = "list_item"
        elif isinstance(el, Image):
            role = "image"
        elif isinstance(el, NarrativeText):
            role = "narrative"
            
        # Extract coordinates if available
        coords = None
        if hasattr(el.metadata, 'coordinates') and el.metadata.coordinates:
            coords = el.metadata.coordinates.points
        
        # Extract table HTML if available (requires hi_res + pdf_infer_table_structure)
        table_html = None
        if isinstance(el, Table) and hasattr(el.metadata, 'text_as_html'):
            table_html = el.metadata.text_as_html
        
        # Extract page number
        page_num = getattr(el.metadata, 'page_number', None)
        
        canonical_raw.append({
            "text": el.text,
            "type": el_type,
            "role": role,
            "page_number": page_num,
            "coordinates": coords,
            "table_html": table_html,
            "source": "unstructured",
            "strategy": strategy,
        })
        
    return canonical_raw
