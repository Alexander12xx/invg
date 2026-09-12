"""
Smart Document Intelligence Engine v13.0
Dynamic-schema extraction with prompt-driven enrichment.

What changed vs. v12:
  • Documents are no longer forced into a fixed column set.
  • Original columns are preserved in their original order and labels.
  • Computed columns (unit_price, line_total) are appended, not substituted.
  • New endpoint /api/apply-prompt applies natural-language instructions.
  • All responses include a `columns` array describing the table layout.

API contract is backward-compatible: `items` still exists, and each item
still carries canonical fields (product_name, quantity, unit_price, total)
in addition to original columns.

ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import io
import os
import re
import json
import math
import time
import logging
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import PyPDF2
import docx
from PIL import Image, ImageEnhance
import pytesseract
from rapidfuzz import fuzz

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-document-engine")

TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

ENGINE_VERSION = "13.0.0"

# ---------------------------------------------------------------------------
#  AI configuration
# ---------------------------------------------------------------------------
AI_ENABLED = os.getenv("AI_ENABLED", "0") == "1"
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_gemini_model = None
_groq_client = None

def _get_gemini():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        return _gemini_model
    except Exception as e:
        logger.warning(f"Gemini unavailable: {e}")
        return None

def _get_groq():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
        return _groq_client
    except Exception as e:
        logger.warning(f"Groq unavailable: {e}")
        return None


app = FastAPI(title="Smart Document Intelligence Engine", version=ENGINE_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ANALYZE_TIMEOUT_SECONDS = int(os.getenv("ANALYZE_TIMEOUT_SECONDS", "60"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "12"))


# ===========================================================================
#  CANONICAL COLUMN MAP
#  These are the "known" semantic roles a column can play. Any column that
#  doesn't match one of these is preserved as-is under its original label.
# ===========================================================================
CANONICAL_ROLES = {
    "product_name": [
        "product", "product name", "item", "item name", "flower", "flower name",
        "description", "product/service", "particulars", "goods", "articles",
        "flower variety", "variety", "cultivar", "species",
    ],
    "quantity": [
        "quantity", "qty", "qnty", "stems", "total stems", "pieces", "pcs",
        "count", "units", "total quantity", "qty trial only - stems",
        "qty trial only -stems", "qty trial only stems",
    ],
    "boxes": ["boxes", "box", "bx", "cartons", "carton", "ctn", "cases"],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box",
    ],
    "unit_price": [
        "price", "unit price", "unit_price", "cost", "rate",
        "price per stem", "price/stem", "unit cost",
    ],
    "total": ["total", "amount", "line total", "line amount", "extended price"],
    "length_cm": ["length", "length (cm)", "length(cm)", "size", "stem length"],
    "color": ["color", "colour", "shade"],
    "head_size_cm": ["head size", "head size (cm)", "head size(cm)"],
    "n": ["n", "no", "no.", "#", "s/n", "sr", "index"],
}


def classify_column(label: str) -> Optional[str]:
    """Return the canonical role for a column label, or None if unknown."""
    if not label:
        return None
    l = re.sub(r"[^a-z0-9 _\-]|_", " ", str(label).lower()).strip()
    l = re.sub(r"\s+", " ", l)
    if not l:
        return None

    # Exact match
    for role, names in CANONICAL_ROLES.items():
        for n in names:
            if l == re.sub(r"[^a-z0-9 _\-]", " ", n.lower()).strip():
                return role

    # Fuzzy match
    best_role, best_score = None, 0
    for role, names in CANONICAL_ROLES.items():
        for n in names:
            s = fuzz.token_set_ratio(l, re.sub(r"[^a-z0-9 _\-]", " ", n.lower()).strip())
            if s > best_score:
                best_score, best_role = s, role
    return best_role if best_score >= 88 else None


def slugify(label: str, taken: set) -> str:
    """Create a unique snake_case key from a column label."""
    base = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "col"
    key = base
    i = 2
    while key in taken:
        key = f"{base}_{i}"
        i += 1
    taken.add(key)
    return key


# ===========================================================================
#  TABLE STRUCTURE — the new shape
# ===========================================================================
def make_column(key: str, label: str, role: Optional[str], source: str) -> Dict[str, Any]:
    return {"key": key, "label": label, "role": role, "source": source}


def build_columns_from_headers(headers: List[str]) -> List[Dict[str, Any]]:
    taken: set = set()
    cols = []
    for h in headers:
        label = str(h or "").strip() or "Column"
        key = slugify(label, taken)
        role = classify_column(label)
        cols.append(make_column(key, label, role, "original"))
    return cols


def ensure_semantic_columns(columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append missing semantic roles as empty 'added' columns."""
    have = {c.get("role") for c in columns if c.get("role")}
    taken = {c["key"] for c in columns}

    additions = [
        ("unit_price", "Unit Price"),
        ("total", "Line Total"),
    ]
    for role, label in additions:
        if role in have:
            continue
        key = slugify(label, taken)
        columns.append(make_column(key, label, role, "added"))
        have.add(role)
    return columns


# ===========================================================================
#  EXCEL / CSV — dynamic extraction
# ===========================================================================
def find_header_row(df: pd.DataFrame) -> int:
    """Find the row most likely to be the header."""
    best_i, best_score = 0, -1
    for i in range(min(15, len(df))):
        row = [str(v) for v in df.iloc[i].tolist() if pd.notna(v) and str(v).strip()]
        if len(row) < 2:
            continue
        score = sum(1 for v in row if classify_column(v) is not None)
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def extract_from_dataframe_dynamic(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"columns": [], "items": []}

    header_row = find_header_row(df)
    raw_headers = [str(v).strip() if pd.notna(v) else "" for v in df.iloc[header_row].tolist()]

    # Drop trailing empty columns
    while raw_headers and not raw_headers[-1]:
        raw_headers.pop()

    if not raw_headers:
        return {"columns": [], "items": []}

    columns = build_columns_from_headers(raw_headers)

    items: List[Dict[str, Any]] = []
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]
        record: Dict[str, Any] = {}
        empty = True
        for col_idx, col in enumerate(columns):
            if col_idx >= len(row):
                record[col["key"]] = None
                continue
            val = row.iloc[col_idx]
            if pd.isna(val):
                record[col["key"]] = None
                continue
            empty = False
            # Keep numbers as numbers
            if isinstance(val, (int, float)):
                record[col["key"]] = int(val) if float(val).is_integer() else float(val)
            else:
                record[col["key"]] = str(val).strip()
        if empty:
            continue

        # Skip summary rows like "Total"
        first_val = next((v for v in record.values() if v), "")
        if isinstance(first_val, str) and first_val.lower().strip() in (
                "total", "subtotal", "grand total"):
            continue

        # Fill in canonical aliases so downstream code can read `quantity` etc.
        for col in columns:
            role = col.get("role")
            if role and role not in record:
                record[role] = record.get(col["key"])
            elif role and record.get(role) in (None, ""):
                record[role] = record.get(col["key"])

        # Try to derive numeric values from labels
        for col in columns:
            if col.get("role") in ("quantity", "boxes", "pack_rate", "length_cm", "head_size_cm"):
                v = record.get(col["key"])
                if v is not None:
                    try:
                        f = float(v)
                        record[col["key"]] = int(f) if f.is_integer() else f
                    except (TypeError, ValueError):
                        pass

        items.append(record)

    columns = ensure_semantic_columns(columns)
    return {"columns": columns, "items": items}


# ===========================================================================
#  TEXT / PDF / DOCX — best-effort tabular extraction
# ===========================================================================
def parse_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    txt = str(value).strip()
    if not txt:
        return None
    txt = re.sub(r"(?i)\b(?:usd|us\$|kes|ksh|eur|gbp|aed|sar|qar)\b", "", txt)
    txt = txt.replace("$", "").replace("€", "").replace("£", "")
    txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt).strip()
    if not txt:
        return None
    if "," in txt and "." in txt:
        if txt.rfind(",") > txt.rfind("."):
            txt = txt.replace(".", "").replace(",", ".")
        else:
            txt = txt.replace(",", "")
    else:
        txt = txt.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", txt)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_from_text_dynamic(text: str) -> Dict[str, Any]:
    """
    Best-effort dynamic extraction from free-form text.
    Splits each line into cells; finds a header; preserves all columns.
    """
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return {"columns": [], "items": []}

    # Find header: line with the most recognizable column labels
    best_idx, best_score, best_cells = None, 0, []
    for i, line in enumerate(lines[:30]):
        cells = [c.strip() for c in re.split(r"\s*\|\s*|\t+|\s{2,}", line) if c.strip()]
        if len(cells) < 2:
            cells = line.split()
        if len(cells) < 2:
            continue
        score = sum(1 for c in cells if classify_column(c) is not None)
        if score > best_score:
            best_score, best_idx, best_cells = score, i, cells

    if best_idx is None or best_score == 0:
        # Fallback: single-column list
        return {
            "columns": [make_column("item", "Item", None, "original")],
            "items": [{"item": l, "product_name": l} for l in lines],
        }

    columns = build_columns_from_headers(best_cells)
    expected = len(columns)

    items: List[Dict[str, Any]] = []
    for line in lines[best_idx + 1:]:
        if not line.strip():
            continue
        # Skip obvious footers
        low = line.lower()
        if any(k in low for k in ("thank you", "computer-generated", "signature",
                                   "subtotal", "grand total", "total:", "notes:")):
            continue
        cells = [c.strip() for c in re.split(r"\s*\|\s*|\t+|\s{2,}", line) if c.strip()]
        if len(cells) < 2:
            cells = line.split()
        if len(cells) < 2:
            continue

        # Right-peel numbers if there are more than the header count
        if len(cells) > expected:
            nums = []
            rest = cells[:]
            while len(rest) > expected - 1:
                try:
                    nums.insert(0, parse_number(rest[-1]))
                    rest.pop()
                except Exception:
                    break
            cells = rest + [str(n) for n in nums]

        record: Dict[str, Any] = {}
        for j, col in enumerate(columns):
            if j >= len(cells):
                record[col["key"]] = None
                continue
            raw = cells[j]
            role = col.get("role")
            if role in ("quantity", "boxes", "pack_rate", "length_cm",
                        "head_size_cm", "unit_price", "total", "n"):
                n = parse_number(raw)
                record[col["key"]] = n if n is not None else raw
            else:
                record[col["key"]] = raw
        items.append(record)

    columns = ensure_semantic_columns(columns)
    return {"columns": columns, "items": items}


# ===========================================================================
#  PDF / DOCX readers
# ===========================================================================
def extract_pdf_text(content: bytes) -> str:
    if fitz is not None:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            parts = []
            for page in doc:
                t = page.get_text("text", sort=True) or ""
                if t.strip():
                    parts.append(t)
            doc.close()
            text = "\n".join(parts)
            if len(re.sub(r"\s+", "", text)) >= 30:
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        parts = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(parts)
        if len(re.sub(r"\s+", "", text)) >= 30:
            return text
    except Exception:
        pass

    # OCR fallback — lightweight
    if fitz is None:
        return ""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        parts = []
        for idx, page in enumerate(doc):
            if idx >= MAX_OCR_PAGES:
                break
            pix = page.get_pixmap(dpi=150, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = img.convert("L")
            img = ImageEnhance.Contrast(img).enhance(1.5)
            try:
                t = pytesseract.image_to_string(img, lang="eng",
                                                config="--oem 3 --psm 6",
                                                timeout=15)
                if t:
                    parts.append(t)
            except Exception:
                continue
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""


def extract_docx_text(content: bytes) -> str:
    if DocxDocument is None:
        return ""
    try:
        d = DocxDocument(io.BytesIO(content))
        parts = []
        for p in d.paragraphs:
            if p.text.strip():
                parts.append(p.text)
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception:
        return ""


# ===========================================================================
#  CANONICAL ITEM VIEW
#  Maps dynamic records to canonical item fields for backward compat.
# ===========================================================================
def canonical_item(record: Dict[str, Any], columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    def get_by_role(role: str):
        for c in columns:
            if c.get("role") == role:
                return record.get(c["key"])
        return None

    name = get_by_role("product_name") or ""
    # Try to combine product + variety if both exist
    variety = None
    for c in columns:
        if c.get("role") == "product_name" and c["key"].startswith("variety"):
            variety = record.get(c["key"])

    qty = get_by_role("quantity")
    boxes = get_by_role("boxes")
    pack_rate = get_by_role("pack_rate")
    unit_price = get_by_role("unit_price")
    total = get_by_role("total")

    # Try to coerce numeric
    def to_num(x):
        if x is None:
            return None
        try:
            f = float(x)
            return int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            return None

    qty = to_num(qty)
    boxes = to_num(boxes)
    pack_rate = to_num(pack_rate)
    unit_price = to_num(unit_price)
    total = to_num(total)

    if total is None and qty is not None and unit_price is not None:
        total = round(qty * unit_price, 4)

    return {
        "product_name": str(name).strip() if name else "",
        "variety": variety,
        "boxes": boxes,
        "pack_rate": pack_rate,
        "quantity": qty,
        "unit_price": unit_price,
        "total": total,
    }


# ===========================================================================
#  PROMPT APPLICATION
# ===========================================================================
class PromptRequest(BaseModel):
    items: List[Dict[str, Any]]
    columns: List[Dict[str, Any]]
    prompt: str


PROMPT_SYSTEM = """You are a strict data-transformation assistant for invoice tables.

You receive:
  - columns: [{key, label, role, source}, ...]  (role may be null)
  - items: rows, keyed by column.key
  - prompt: a natural-language instruction from the user

You return ONLY valid JSON with this exact shape:
{
  "items": [ <same length as input, same keys, updated values> ],
  "columns": [ <same columns array, optionally with new 'added' columns> ],
  "explanation": "<one sentence on what you did>"
}

Rules:
1. NEVER delete rows unless the prompt explicitly says to.
2. NEVER change the number of columns unless adding a new one. New columns must have "source": "added".
3. Preserve every existing key in every row.
4. Numeric fields must be numbers, not strings.
5. If the prompt is unclear, do nothing and explain.
6. Never invent data.
"""


def _clean_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def apply_prompt_ai(items, columns, prompt):
    payload = json.dumps({"columns": columns, "items": items, "prompt": prompt},
                         ensure_ascii=False)

    providers = []
    if AI_PROVIDER in ("gemini", "auto"):
        providers.append(("gemini", _get_gemini()))
    if AI_PROVIDER in ("groq", "auto"):
        providers.append(("groq", _get_groq()))

    for name, client in providers:
        if client is None:
            continue
        try:
            if name == "gemini":
                resp = client.generate_content(
                    PROMPT_SYSTEM + "\n\nINPUT:\n" + payload,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "max_output_tokens": 8192,
                    },
                )
                raw = resp.text if resp else ""
            else:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "system", "content": PROMPT_SYSTEM},
                              {"role": "user", "content": payload}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=8192,
                )
                raw = resp.choices[0].message.content if resp.choices else ""
            data = _clean_json(raw)
            if data and "items" in data:
                return data
        except Exception as e:
            logger.warning(f"Prompt via {name} failed: {e}")
    return None


def apply_prompt_deterministic(items, columns, prompt):
    """Handle common prompts without AI (fallback)."""
    p = prompt.lower().strip()
    cols = list(columns)
    taken = {c["key"] for c in cols}

    # Detect "add unit price column at X"
    m = re.search(r"(?:add|create)\s+(?:a\s+)?(?:unit\s*price|price)\s*(?:column)?"
                  r"(?:\s*(?:at|=|of|:)\s*([0-9]+(?:\.[0-9]+)?))?", p)
    if m:
        default_price = float(m.group(1)) if m.group(1) else None
        if not any(c.get("role") == "unit_price" for c in cols):
            key = slugify("Unit Price", taken)
            cols.append(make_column(key, "Unit Price", "unit_price", "added"))
            for it in items:
                it[key] = default_price
        else:
            for c in cols:
                if c.get("role") == "unit_price":
                    for it in items:
                        if it.get(c["key"]) in (None, "", 0):
                            it[c["key"]] = default_price

    # Detect "calculate line total" / "add total"
    if re.search(r"\b(total|line total|line amount|amount)\b", p) and \
       re.search(r"\b(calc|comput|add|create)\b", p):
        qty_key = next((c["key"] for c in cols if c.get("role") == "quantity"), None)
        price_key = next((c["key"] for c in cols if c.get("role") == "unit_price"), None)
        if qty_key and price_key:
            total_key = next((c["key"] for c in cols if c.get("role") == "total"), None)
            if not total_key:
                total_key = slugify("Line Total", taken)
                cols.append(make_column(total_key, "Line Total", "total", "added"))
            for it in items:
                try:
                    q = float(it.get(qty_key) or 0)
                    u = float(it.get(price_key) or 0)
                    it[total_key] = round(q * u, 2)
                except (TypeError, ValueError):
                    it[total_key] = None

    # Detect "remove column X"
    m = re.search(r"(?:remove|delete)\s+(?:the\s+)?column\s+([a-z0-9 _\-]+)", p)
    if m:
        target = m.group(1).strip()
        for c in list(cols):
            if target in c["label"].lower() or target == c["key"]:
                for it in items:
                    it.pop(c["key"], None)
                cols.remove(c)

    # Detect "multiply quantity by N"
    m = re.search(r"multiply\s+quantity\s+by\s+([0-9]+(?:\.[0-9]+)?)", p)
    if m:
        factor = float(m.group(1))
        qty_key = next((c["key"] for c in cols if c.get("role") == "quantity"), None)
        if qty_key:
            for it in items:
                try:
                    it[qty_key] = round(float(it[qty_key]) * factor, 2)
                except (TypeError, ValueError, KeyError):
                    pass

    return {"items": items, "columns": cols,
            "explanation": "Applied via deterministic parser."}


# ===========================================================================
#  ROUTES
# ===========================================================================
@app.get("/")
async def root():
    return {
        "service": "Smart Document Intelligence Engine",
        "version": ENGINE_VERSION,
        "status": "operational",
        "ai_enabled": AI_ENABLED,
        "ai_provider": AI_PROVIDER,
        "endpoints": ["/api/ping", "/api/health", "/api/analyze",
                      "/api/apply-prompt", "/api/match-products"],
    }


@app.get("/api/ping")
def ping():
    return {"ok": True, "version": ENGINE_VERSION,
            "ai_enabled": AI_ENABLED, "ai_provider": AI_PROVIDER}


@app.get("/api/health")
def health():
    return {"status": "healthy", "version": ENGINE_VERSION,
            "ai_enabled": AI_ENABLED, "ai_provider": AI_PROVIDER}


def analyze_bytes(content: bytes, fname: str, ext: str,
                  company_id: int, prompt: str = "") -> Dict[str, Any]:
    started = time.perf_counter()
    result: Dict[str, Any] = {"columns": [], "items": [], "extraction_method": ""}
    text_extracted = ""

    if ext in ("xlsx", "xls", "xlsm", "csv"):
        try:
            if ext == "csv":
                try:
                    df = pd.read_csv(io.BytesIO(content), header=None)
                except Exception:
                    df = pd.read_csv(io.BytesIO(content), header=None, sep=";")
            else:
                df = pd.read_excel(io.BytesIO(content), header=None)
            result = extract_from_dataframe_dynamic(df)
            result["extraction_method"] = "spreadsheet_dynamic"
        except Exception as e:
            logger.exception(f"Spreadsheet extraction failed: {e}")

    elif ext == "pdf":
        text_extracted = extract_pdf_text(content)
        result = extract_from_text_dynamic(text_extracted)
        result["extraction_method"] = "pdf_dynamic"

    elif ext in ("docx", "doc"):
        text_extracted = extract_docx_text(content)
        result = extract_from_text_dynamic(text_extracted)
        result["extraction_method"] = "docx_dynamic"

    elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
        try:
            img = Image.open(io.BytesIO(content))
            img = img.convert("L")
            img = ImageEnhance.Contrast(img).enhance(1.5)
            text_extracted = pytesseract.image_to_string(
                img, lang="eng", config="--oem 3 --psm 6", timeout=30)
        except Exception:
            text_extracted = ""
        result = extract_from_text_dynamic(text_extracted)
        result["extraction_method"] = "image_ocr_dynamic"

    else:
        text_extracted = content.decode("utf-8", errors="ignore")
        result = extract_from_text_dynamic(text_extracted)
        result["extraction_method"] = "text_dynamic"

    # Apply initial prompt if provided
    if prompt and result.get("items"):
        applied = apply_prompt_ai(result["items"], result["columns"], prompt)
        if not applied:
            applied = apply_prompt_deterministic(result["items"],
                                                 result["columns"], prompt)
        result["items"] = applied.get("items", result["items"])
        result["columns"] = applied.get("columns", result["columns"])
        result["prompt_applied"] = True
        result["prompt_explanation"] = applied.get("explanation", "")

    # Canonical view for downstream
    canonical_items = []
    for r in result["items"]:
        ci = canonical_item(r, result["columns"])
        ci.update({"_record": r})
        canonical_items.append(ci)

    # Totals
    total_qty = sum((i["quantity"] or 0) for i in canonical_items
                    if i.get("quantity") is not None)
    total_amount = sum((i["total"] or 0) for i in canonical_items
                       if i.get("total") is not None)

    return {
        "success": True,
        "columns": result["columns"],
        "items": canonical_items,
        "raw_items": result["items"],
        "item_count": len(canonical_items),
        "total_quantity": int(total_qty) if total_qty else None,
        "total_amount": round(float(total_amount), 2),
        "text_extracted": text_extracted[:12000],
        "extraction_method": result["extraction_method"],
        "prompt_applied": result.get("prompt_applied", False),
        "prompt_explanation": result.get("prompt_explanation", ""),
        "engine_version": ENGINE_VERSION,
        "file_type": ext,
        "filename": fname,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
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

    logger.info(f"Analyzing {fname} ({ext}), company={company_id}, "
                f"size={len(content)}, prompt_len={len(prompt)}")

    loop = __import__("asyncio").get_running_loop()
    try:
        return await loop.run_in_executor(
            None, analyze_bytes, content, fname, ext, company_id, prompt)
    except Exception as e:
        logger.exception("Analyze crashed")
        return {
            "success": False, "columns": [], "items": [], "raw_items": [],
            "item_count": 0, "error": str(e), "engine_version": ENGINE_VERSION,
        }


@app.post("/api/apply-prompt")
async def apply_prompt_endpoint(req: PromptRequest):
    """Re-apply a prompt to already-extracted items (called from review page)."""
    items = req.items or []
    columns = req.columns or []
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"success": True, "items": items, "columns": columns,
                "explanation": "No prompt provided."}

    # Try AI first
    applied = apply_prompt_ai(items, columns, prompt)
    if not applied:
        applied = apply_prompt_deterministic(items, columns, prompt)

    # Recompute canonical view
    canonical = []
    for r in applied["items"]:
        ci = canonical_item(r, applied["columns"])
        ci["_record"] = r
        canonical.append(ci)

    return {
        "success": True,
        "columns": applied["columns"],
        "items": canonical,
        "raw_items": applied["items"],
        "explanation": applied.get("explanation", ""),
        "applied_via": "ai" if AI_ENABLED else "deterministic",
    }


# ===========================================================================
#  PRODUCT MATCHING (unchanged from v12)
# ===========================================================================
class MatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    company_products: List[Dict[str, Any]]


def normalize_product_for_match(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "").lower()).strip()
    n = re.sub(r"\b\d+(?:\.\d+)?\s*cm\b", " ", n)
    n = re.sub(r"\b(?:box|boxes|bx|carton|cartons|qty|quantity|stems?)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


@app.post("/api/match-products")
async def match_products(req: MatchRequest):
    if not req.items or not req.company_products:
        return {"success": True, "items": req.items, "matched_count": 0,
                "review_count": 0}
    out = []
    for item in req.items:
        iname = normalize_product_for_match(item.get("product_name", ""))
        best, best_score = None, 0
        for prod in req.company_products:
            names = [prod.get("name", "")]
            if isinstance(prod.get("aliases"), list):
                names.extend(prod["aliases"])
            for cand in names:
                cn = normalize_product_for_match(cand)
                if not cn:
                    continue
                s = max(fuzz.ratio(iname, cn),
                        fuzz.token_set_ratio(iname, cn),
                        fuzz.WRatio(iname, cn))
                if s > best_score:
                    best_score, best = s, prod
        conf = best_score / 100.0
        if best and conf >= 0.84:
            out.append({**item, "product_id": best.get("id"),
                        "matched_product_name": best.get("name"),
                        "match_confidence": round(conf, 3),
                        "match_status": "matched"})
        else:
            out.append({**item, "product_id": None,
                        "match_confidence": round(conf, 3),
                        "match_status": "review_required"})
    return {"success": True, "items": out,
            "matched_count": sum(1 for x in out if x["match_status"] == "matched"),
            "review_count": sum(1 for x in out if x["match_status"] == "review_required")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
