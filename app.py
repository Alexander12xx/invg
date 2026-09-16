"""
ALTECH SOFTWARE DEVELOPERS
SMART DOCUMENT INTELLIGENCE ENGINE v21
--------------------------------------
Universal document ingestion + semantic structure understanding.

Supported formats:
  • XLSX / XLS / XLSM (multi-sheet, formulas, currency strings)
  • CSV (comma, semicolon, tab auto-detected)
  • PDF (text layer first, then OCR)
  • DOCX (paragraphs + tables)
  • Plain text, JSON
  • Images (PNG, JPG, TIFF, BMP, WEBP) via OCR

Design principles:
  1. Never assume a fixed invoice template.
  2. Never silently drop data — anything unrecognized is preserved
     under its original column or in raw_text.
  3. Semantic roles are inferred, not hard-coded; a column called
     "QTY Trial only - stems" is recognized as quantity.
  4. Structure is exposed as {columns, items} so the chat engine can
     operate on it regardless of source format.
  5. Every response carries diagnostics: sheets, trust score, anomalies.
"""

from __future__ import annotations

import io
import os
import re
import json
import math
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import openpyxl
import PyPDF2
import docx
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from rapidfuzz import fuzz

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from chat_engine import process_message as chat_process_message
    _CHAT_AVAILABLE = True
except Exception as _e:
    logging.getLogger("altech-engine").warning(f"chat_engine not available: {_e}")
    chat_process_message = None
    _CHAT_AVAILABLE = False


# ===========================================================================
#  LOGGING + CONFIG
# ===========================================================================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("altech-smart-document")

ENGINE_VERSION = "21.0.0"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "40"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "20000"))
MAX_SHEETS = int(os.getenv("MAX_SHEETS", "50"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "30"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD


# ===========================================================================
#  FASTAPI
# ===========================================================================
app = FastAPI(title="Altech Smart Document Intelligence Engine",
              version=ENGINE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
#  SEMANTIC ROLE MODEL
#  A column is classified into one of these roles. The chat engine can
#  address columns by role ("set unit price") or by label ("set Price").
# ===========================================================================
ROLE_ALIASES = {
    "product_name": [
        "product", "product name", "item", "item name", "flower",
        "flower name", "description", "item description",
        "particulars", "goods", "article", "articles",
        "flower variety", "variety", "cultivar", "product/service",
        "service", "commodity",
    ],
    "variety": ["variety", "cultivar", "cultivar name", "flower variety name"],
    "quantity": [
        "quantity", "qty", "qnty", "stems", "total stems", "pieces", "pcs",
        "units", "count", "total quantity", "qty trial only - stems",
        "qty trial only-stems", "qty trial only stems", "trial qty",
    ],
    "boxes": ["boxes", "box", "bx", "cartons", "carton", "ctn",
              "cases", "case", "bundles", "bundle", "packages", "pkg"],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box", "quantity per box",
        "stems per carton", "qty/carton",
    ],
    "unit_price": [
        "price", "unit price", "unit_price", "rate", "cost", "unit cost",
        "price per stem", "price/stem", "cost per stem", "selling price",
        "unit selling price", "price per unit",
    ],
    "total": [
        "total", "amount", "line total", "line amount", "total amount",
        "extended price", "revenue", "line value",
    ],
    "length_cm": [
        "length", "length cm", "length (cm)", "length(cm)",
        "stem length", "size", "size cm", "height",
    ],
    "head_size_cm": ["head size", "head size cm", "head size (cm)"],
    "color": ["color", "colour", "shade"],
    "farm_code": ["farm code", "farm", "farmcode", "farm ref",
                  "grower code", "supplier code"],
    "invoice_number": [
        "invoice number", "invoice no", "invoice #", "invoice no.",
        "inv no", "inv #", "reference", "ref no", "document number",
    ],
    "date": [
        "date", "invoice date", "shipment date", "date of shipment",
        "issue date", "document date",
    ],
    "due_date": ["due date", "payment due", "valid until", "expiry"],
    "currency": ["currency", "currency code", "ccy"],
    "vat_rate": ["vat", "vat rate", "vat %", "tax", "tax rate", "tax %"],
    "discount": ["discount", "disc.", "rebate"],
    "consignee": ["consignee", "bill to", "ship to", "buyer", "customer"],
    "seller": ["seller", "vendor", "supplier", "exporter"],
    "awb": ["awb", "air waybill", "waybill", "tracking number"],
    "net_weight": ["net weight", "net kg", "net weight (kgs)"],
    "gross_weight": ["gross weight", "gross kg"],
}

NUMERIC_ROLES = {
    "quantity", "boxes", "pack_rate", "length_cm", "head_size_cm",
    "unit_price", "total", "vat_rate", "discount", "net_weight",
    "gross_weight",
}

TEXT_ROLES = {
    "product_name", "variety", "description", "color", "farm_code",
    "invoice_number", "date", "due_date", "currency", "consignee",
    "seller", "awb", "notes",
}


def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s.strip())


def clean_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")


# ===========================================================================
#  NUMBER PARSING  (the "unknown how many errors" fix lives here)
# ===========================================================================
def parse_number(v: Any) -> Optional[float]:
    """
    Robust number parser. Handles:
      $1.30, USD 1,350.00, KES 1,234.56, 1.234,56 (EU), "1 200", "=PRODUCT(...)"
      Returns None if the input is not numeric-looking.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None

    s = str(v).strip()
    if not s:
        return None

    # Formula cell like "=PRODUCT(G3:H3)" — not parseable, return None
    if s.startswith("="):
        return None

    # Strip currency words and symbols
    s = re.sub(
        r"(?i)\b(?:usd|us\$|kes|ksh|kshs|eur|gbp|aed|sar|qar|dollars?|shillings?)\b",
        "", s)
    s = s.replace("$", "").replace("€", "").replace("£", "").strip()

    # Handle thousands/decimal separators — the ambiguous case
    if "," in s and "." in s:
        # Which separator is the decimal? The one that comes last.
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # Single comma: if it looks like European decimal (2 digits after), treat as decimal
        if re.search(r",\d{1,2}$", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    # Non-breaking spaces sometimes appear between thousand groups
    s = s.replace("\u00a0", "").replace("\u202f", "")
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


# ===========================================================================
#  ROLE CLASSIFICATION
# ===========================================================================
def classify(label: Any) -> Optional[str]:
    l = re.sub(r"[^a-z0-9 ]+", " ", norm(label)).strip()
    if not l:
        return None
    best_role, best_score = None, 0
    for role, names in ROLE_ALIASES.items():
        for n in names:
            a = re.sub(r"[^a-z0-9 ]+", " ", norm(n)).strip()
            if l == a:
                return role
            s = max(fuzz.WRatio(l, a), fuzz.token_set_ratio(l, a))
            if s > best_score:
                best_score, best_role = s, role
    return best_role if best_score >= 78 else None


def slugify(label: str, taken: set) -> str:
    base = clean_key(label) or "column"
    key = base
    i = 2
    while key in taken:
        key = f"{base}_{i}"
        i += 1
    taken.add(key)
    return key


def build_columns(headers: List[Any]) -> List[Dict[str, Any]]:
    taken: set = set()
    cols = []
    for h in headers:
        label = str(h or "").strip() or "Column"
        key = slugify(label, taken)
        cols.append({
            "key": key,
            "label": label,
            "role": classify(label),
            "source": "original",
        })
    return cols


# ===========================================================================
#  CELL / ROW CLEANING
# ===========================================================================
def clean_cell(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (int, float)):
        x = float(v)
        return int(x) if x.is_integer() else x
    return str(v).strip()


def row_is_summary(row: Dict[str, Any]) -> bool:
    text = " ".join(norm(v) for v in row.values() if v not in (None, ""))
    return bool(re.search(
        r"\b(?:subtotal|grand total|invoice total|balance due|"
        r"thank you|total amount)\b", text))


def row_is_empty(row: Dict[str, Any]) -> bool:
    return all(v in (None, "") for v in row.values())


# ===========================================================================
#  HEADER DETECTION
# ===========================================================================
def find_header_row(df: pd.DataFrame) -> int:
    """
    The header row is the one whose cells most often resolve to known roles.
    Bias towards the first few rows so we don't accidentally pick a summary row.
    """
    best = (0, 0)  # (score, index)
    scan = min(len(df), 40)
    for i in range(scan):
        try:
            vals = [str(x).strip() for x in df.iloc[i].tolist()
                    if pd.notna(x) and str(x).strip()]
        except Exception:
            continue
        if len(vals) < 2:
            continue
        recognized = sum(1 for x in vals if classify(x))
        textish = sum(1 for x in vals if re.search(r"[A-Za-z]", x))
        # Require some semantic hits to consider this a header
        if recognized == 0:
            continue
        score = recognized * 10 + min(textish, 10) - i  # prefer earlier rows
        if score > best[0]:
            best = (score, i)
    return best[1]


# ===========================================================================
#  DATAFRAME → STRUCTURE
# ===========================================================================
def dataframe_to_structure(df: pd.DataFrame, sheet_name: str = "") -> Dict[str, Any]:
    if df is None or df.empty:
        return {"columns": [], "items": [], "sheet": sheet_name, "header_row": None}

    df = df.iloc[:MAX_ROWS, :]
    header_idx = find_header_row(df)

    try:
        raw_headers = [clean_cell(x) for x in df.iloc[header_idx].tolist()]
    except Exception:
        return {"columns": [], "items": [], "sheet": sheet_name, "header_row": None}

    # Drop trailing empty headers only (interior blanks become "Column")
    while raw_headers and raw_headers[-1] in (None, ""):
        raw_headers.pop()

    if not raw_headers:
        return {"columns": [], "items": [], "sheet": sheet_name, "header_row": None}

    columns = build_columns(raw_headers)

    items: List[Dict[str, Any]] = []
    for ridx in range(header_idx + 1, len(df)):
        try:
            row_vals = list(df.iloc[ridx].tolist())
        except Exception:
            continue
        row_vals = row_vals[:len(columns)]
        record: Dict[str, Any] = {}
        for j, col in enumerate(columns):
            raw = row_vals[j] if j < len(row_vals) else None
            cell = clean_cell(raw)
            if col.get("role") in NUMERIC_ROLES:
                num = parse_number(cell)
                record[col["key"]] = num if num is not None else cell
            else:
                record[col["key"]] = cell
        if row_is_empty(record):
            continue
        if row_is_summary(record):
            continue
        # Semantic aliases (so downstream chat code can use role names)
        for col in columns:
            role = col.get("role")
            if role and role not in record:
                record[role] = record.get(col["key"])
            elif role and record.get(role) in (None, ""):
                record[role] = record.get(col["key"])
        items.append(record)

    return {
        "columns": columns,
        "items": items,
        "sheet": sheet_name,
        "header_row": header_idx + 1,
    }


# ===========================================================================
#  EXCEL / CSV
# ===========================================================================
def extract_excel(content: bytes, ext: str) -> List[Dict[str, Any]]:
    structures: List[Dict[str, Any]] = []

    if ext == "csv":
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(
                    io.BytesIO(content), header=None, sep=sep,
                    engine="python", dtype=object)
                if df.shape[1] > 1 or sep == ",":
                    return [dataframe_to_structure(df, "CSV")]
            except Exception:
                continue
        return []

    try:
        wb = openpyxl.load_workbook(
            io.BytesIO(content), read_only=True, data_only=True)
    except Exception as e:
        log.warning(f"openpyxl failed: {e}")
        return []

    for ws in list(wb.worksheets)[:MAX_SHEETS]:
        try:
            rows = list(ws.values)
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        structure = dataframe_to_structure(df, ws.title)
        if structure.get("items") or structure.get("columns"):
            structures.append(structure)
    return structures


# ===========================================================================
#  PDF / DOCX / IMAGE / TEXT
# ===========================================================================
def ocr_image(img: Image.Image) -> str:
    img = ImageOps.exif_transpose(img).convert("L")
    w, h = img.size
    if max(w, h) < 2200:
        scale = 2200 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    img = ImageEnhance.Contrast(img).enhance(1.7)
    best = ""
    for psm in (6, 4, 11):
        try:
            t = pytesseract.image_to_string(
                img, lang="eng", config=f"--oem 3 --psm {psm}", timeout=25)
            if len(re.findall(r"[A-Za-z0-9]", t)) > \
               len(re.findall(r"[A-Za-z0-9]", best)):
                best = t
        except Exception:
            continue
    return best


def extract_pdf(content: bytes) -> Tuple[str, str]:
    """Return (text, method)."""
    # Native text first
    if fitz is not None:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            chunks = []
            for p in doc:
                t = p.get_text("text", sort=True) or ""
                if t.strip():
                    chunks.append(t)
            doc.close()
            text = "\n".join(chunks)
            if len(re.sub(r"\s+", "", text)) >= 30:
                return text, "pdf_text"
        except Exception as e:
            log.warning(f"fitz failed: {e}")

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        if len(re.sub(r"\s+", "", text)) >= 30:
            return text, "pdf_text_pypdf2"
    except Exception:
        pass

    if fitz is None:
        return "", "pdf_no_reader"

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        parts = []
        for i, p in enumerate(doc):
            if i >= MAX_OCR_PAGES:
                break
            pix = p.get_pixmap(dpi=OCR_DPI, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(ocr_image(img))
        return "\n".join(parts), "pdf_ocr"
    except Exception:
        return "", "pdf_ocr_failed"


def extract_docx(content: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(content))
        parts = []
        for p in d.paragraphs:
            if p.text.strip():
                parts.append(p.text)
        for table in d.tables:
            for row in table.rows:
                vals = [c.text.strip() for c in row.cells]
                if any(vals):
                    parts.append(" | ".join(vals))
        return "\n".join(parts)
    except Exception:
        return ""


def text_to_structure(text: str) -> Dict[str, Any]:
    """Parse a plain-text or OCR'd table into a structure."""
    lines = [x.strip() for x in str(text or "").splitlines() if x.strip()]
    if not lines:
        return {"columns": [], "items": [], "sections": []}

    # Locate the most header-like line
    best_score, best_idx, best_cells = 0, None, None
    for i, line in enumerate(lines[:80]):
        cells = [x.strip() for x in re.split(r"\s*\|\s*|\t+|\s{2,}", line)
                 if x.strip()]
        if len(cells) < 2:
            continue
        score = sum(1 for c in cells if classify(c))
        if score > best_score:
            best_score, best_idx, best_cells = score, i, cells

    if best_idx is None or best_score == 0:
        return {
            "columns": [{"key": "raw_text", "label": "Text",
                         "role": None, "source": "original"}],
            "items": [{"raw_text": x, "product_name": x} for x in lines],
            "sections": [],
        }

    columns = build_columns(best_cells)

    items: List[Dict[str, Any]] = []
    for line in lines[best_idx + 1:]:
        low = line.lower()
        if re.search(r"\b(?:subtotal|grand total|balance due|"
                     r"thank you|invoice total)\b", low):
            continue
        cells = [x.strip() for x in re.split(r"\s*\|\s*|\t+|\s{2,}", line)
                 if x.strip()]
        if len(cells) < 2:
            continue
        record: Dict[str, Any] = {}
        for j, col in enumerate(columns):
            raw = cells[j] if j < len(cells) else None
            if col.get("role") in NUMERIC_ROLES:
                num = parse_number(raw)
                record[col["key"]] = num if num is not None else raw
            else:
                record[col["key"]] = raw
            if col.get("role"):
                record[col["role"]] = record[col["key"]]
        if row_is_empty(record):
            continue
        items.append(record)

    return {"columns": columns, "items": items, "sections": []}


# ===========================================================================
#  CANONICALIZATION + VALIDATION
# ===========================================================================
def canonicalize(structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    cols = structure.get("columns", [])
    items = structure.get("items", [])

    def role_col(role):
        for c in cols:
            if c.get("role") == role:
                return c

    out = []
    for i, row in enumerate(items):
        x = dict(row)
        warnings: List[str] = []
        confidence = 0.55

        q_col = role_col("quantity")
        p_col = role_col("unit_price")
        t_col = role_col("total")
        b_col = role_col("boxes")
        pr_col = role_col("pack_rate")

        qv = parse_number(x.get(q_col["key"])) if q_col else None
        pv = parse_number(x.get(p_col["key"])) if p_col else None
        tv = parse_number(x.get(t_col["key"])) if t_col else None
        bv = parse_number(x.get(b_col["key"])) if b_col else None
        prv = parse_number(x.get(pr_col["key"])) if pr_col else None

        if qv is not None:
            confidence += 0.08
        if pv is not None:
            confidence += 0.08
        if tv is not None:
            confidence += 0.08

        # Arithmetic reconciliation
        if qv is not None and pv is not None:
            expected = qv * pv
            if t_col is None:
                x.setdefault("total", round(expected, 4))
                x["total"] = round(expected, 4)
                confidence += 0.06
            elif tv is None:
                x[t_col["key"]] = round(expected, 4)
                confidence += 0.06
            elif abs(expected - tv) > max(0.05, abs(tv) * 0.02):
                warnings.append("quantity_x_unit_price_mismatch")

        if bv is not None and prv is not None and qv is not None:
            if abs(bv * prv - qv) > max(0.5, abs(qv) * 0.02):
                warnings.append("boxes_x_packrate_mismatch")

        x["_meta"] = {
            "row_index": i + 1,
            "confidence": round(min(confidence, 1), 3),
            "warnings": warnings,
        }
        out.append(x)
    return out


def infer_document_type(text: str, items: List[Dict[str, Any]], filename: str) -> str:
    blob = norm(" ".join([Path(filename).stem, (text or "")[:5000]]))
    for kw, dt in [
        ("proforma", "proforma_invoice"),
        ("purchase order", "purchase_order"),
        ("packing list", "packing_list"),
        ("delivery note", "delivery_note"),
        ("quotation", "quotation"),
        ("quote", "quotation"),
        ("receipt", "receipt"),
        ("invoice", "invoice"),
    ]:
        if kw in blob:
            return dt
    if items:
        return "line_item_document"
    return "business_document"


# ===========================================================================
#  MAIN ANALYSIS ENTRY POINT
# ===========================================================================
def analyze_bytes(content: bytes, fname: str, ext: str,
                  company_id: int = 0, prompt: str = "") -> Dict[str, Any]:
    started = time.perf_counter()
    ext = (ext or Path(fname).suffix.lstrip(".")).lower()

    structures: List[Dict[str, Any]] = []
    raw_text = ""
    method = "unknown"

    if ext in ("xlsx", "xlsm", "xls"):
        structures = extract_excel(content, ext)
        method = "excel"
    elif ext == "csv":
        structures = extract_excel(content, "csv")
        method = "csv"
    elif ext == "pdf":
        raw_text, method = extract_pdf(content)
        if raw_text:
            structures = [text_to_structure(raw_text)]
    elif ext in ("docx", "doc"):
        raw_text = extract_docx(content)
        method = "docx"
        if raw_text:
            structures = [text_to_structure(raw_text)]
    elif ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"):
        try:
            raw_text = ocr_image(Image.open(io.BytesIO(content)))
            method = "image_ocr"
        except Exception:
            raw_text = ""
        if raw_text:
            structures = [text_to_structure(raw_text)]
    elif ext == "json":
        try:
            obj = json.loads(content.decode("utf-8", errors="ignore"))
            raw_text = json.dumps(obj, ensure_ascii=False, indent=2)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                headers = list(obj[0].keys())
                structures = [{
                    "columns": build_columns(headers),
                    "items": obj,
                    "sheet": "JSON",
                }]
            else:
                structures = [text_to_structure(raw_text)]
            method = "json"
        except Exception:
            raw_text = content.decode("utf-8", errors="ignore")
            structures = [text_to_structure(raw_text)]
            method = "text"
    else:
        raw_text = content.decode("utf-8", errors="ignore")
        structures = [text_to_structure(raw_text)]
        method = "text"

    if not structures:
        return {
            "success": False,
            "engine_version": ENGINE_VERSION,
            "error": "No readable structure was detected.",
            "columns": [],
            "items": [],
            "raw_items": [],
            "raw_text": raw_text[:20000],
            "diagnostics": {
                "stage": "ingestion",
                "file": fname,
                "method": method,
                "trust_score": 0,
                "confidence_band": "review",
            },
        }

    # Merge sheets — first sheet supplies the primary column schema.
    primary_columns = structures[0].get("columns", [])
    merged_items: List[Dict[str, Any]] = []
    sheet_names: List[str] = []
    for s in structures:
        sheet_names.append(s.get("sheet") or "")
        for row in s.get("items", []):
            r = dict(row)
            r["_sheet"] = s.get("sheet") or None
            merged_items.append(r)

    items = canonicalize({"columns": primary_columns, "items": merged_items})

    total_qty = sum(
        (parse_number(r.get("quantity")) or 0) for r in items)
    total_amount = sum(
        (parse_number(r.get("total")) or 0) for r in items)

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
        "filename": fname,
        "file_type": ext,
        "extraction_method": method,
        "document_type": infer_document_type(raw_text, items, fname),
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
            "confidence_band": (
                "high" if trust >= 0.85
                else "medium" if trust >= 0.65
                else "review"),
            "anomalies": anomalies,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "company_id": company_id,
        "prompt_received": bool(prompt),
    }


# ===========================================================================
#  PYDANTIC MODELS
# ===========================================================================
class ChatRequest(BaseModel):
    items: List[Dict[str, Any]] = []
    columns: List[Dict[str, Any]] = []
    message: str = ""


class PromptRequest(BaseModel):
    items: List[Dict[str, Any]] = []
    columns: List[Dict[str, Any]] = []
    prompt: str = ""


class MatchRequest(BaseModel):
    items: List[Dict[str, Any]] = []
    company_products: List[Dict[str, Any]] = []


# ===========================================================================
#  ENDPOINTS
# ===========================================================================
@app.get("/")
async def root():
    return {
        "service": "Altech Smart Document Intelligence Engine",
        "version": ENGINE_VERSION,
        "status": "operational",
        "chat_engine_available": _CHAT_AVAILABLE,
        "capabilities": [
            "xlsx", "xls", "xlsm", "csv", "pdf", "docx", "image", "text",
            "json", "semantic_roles", "ocr",
            "natural_language_commands",
            "calculation_verification", "multi_sheet",
        ],
    }


@app.get("/api/ping")
def ping():
    return {"ok": True, "version": ENGINE_VERSION,
            "chat_engine_available": _CHAT_AVAILABLE}


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "version": ENGINE_VERSION,
        "ocr_available": True,
        "pdf_available": fitz is not None,
        "chat_engine_available": _CHAT_AVAILABLE,
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    company_id: int = Form(0),
    file_type: Optional[str] = Form(None),
    prompt: str = Form(""),
):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")

    fname = file.filename or "upload"
    ext = (file_type or Path(fname).suffix.lstrip(".")).lower()

    try:
        return analyze_bytes(content, fname, ext, company_id, prompt)
    except Exception as e:
        log.exception("Analysis failed")
        return {
            "success": False,
            "engine_version": ENGINE_VERSION,
            "error": "Analysis failed safely.",
            "columns": [],
            "items": [],
            "raw_items": [],
            "diagnostics": {"exception": str(e), "file": fname},
        }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not _CHAT_AVAILABLE or chat_process_message is None:
        return {
            "success": False,
            "status": "error",
            "items": req.items,
            "columns": req.columns,
            "explanation": "Command engine is not loaded on the server.",
            "needs_clarification": False,
        }
    try:
        return chat_process_message(req.items, req.columns, req.message)
    except Exception as e:
        log.exception("chat failure")
        return {
            "success": False,
            "status": "error",
            "items": req.items,
            "columns": req.columns,
            "explanation": "The command could not be applied safely.",
            "error": str(e),
            "needs_clarification": False,
        }


@app.post("/api/apply-prompt")
async def apply_prompt(req: PromptRequest):
    return await chat(ChatRequest(
        items=req.items, columns=req.columns, message=req.prompt))


@app.post("/api/extract-text")
async def extract_text(file: UploadFile = File(...)):
    content = await file.read()
    fname = file.filename or "upload"
    ext = Path(fname).suffix.lstrip(".").lower()

    if ext == "pdf":
        text, method = extract_pdf(content)
    elif ext == "docx":
        text = extract_docx(content)
        method = "docx"
    elif ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"):
        text = ocr_image(Image.open(io.BytesIO(content)))
        method = "image_ocr"
    else:
        text = content.decode("utf-8", errors="ignore")
        method = "text"

    return {"success": True, "text": text, "length": len(text),
            "file_type": ext, "extraction_method": method,
            "engine_version": ENGINE_VERSION}


@app.post("/api/match-products")
async def match_products(req: MatchRequest):
    out = []
    for item in req.items:
        q = norm(item.get("product_name") or item.get("item")
                 or item.get("description"))
        ranked = []
        for p in req.company_products:
            names = [p.get("name", "")]
            if isinstance(p.get("aliases"), list):
                names.extend(p["aliases"])
            score = max(
                [100 if q == norm(n) else fuzz.WRatio(q, norm(n))
                 for n in names if n] or [0])
            ranked.append((score, p))
        ranked.sort(reverse=True, key=lambda x: x[0])
        best = ranked[0] if ranked else (0, None)
        second = ranked[1][0] if len(ranked) > 1 else 0
        accepted = bool(
            best[1] and best[0] >= 88
            and (best[0] - second >= 6 or best[0] >= 97))
        out.append({
            **item,
            "product_id": best[1].get("id") if accepted else None,
            "matched_product_name": best[1].get("name") if accepted else None,
            "match_confidence": round(best[0] / 100, 3),
            "match_status": "matched" if accepted else "review_required",
        })
    return {
        "success": True, "items": out,
        "matched_count": sum(1 for x in out if x["match_status"] == "matched"),
        "review_count": sum(1 for x in out if x["match_status"] == "review_required"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
