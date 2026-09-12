"""
Smart Document Intelligence Engine v13.2
Deterministic extraction + dynamic schema + chat-driven transformations.

Endpoints:
  GET  /                      service info
  GET  /api/ping              liveness
  GET  /api/health            full status
  POST /api/analyze           extract items from a file (file + optional prompt)
  POST /api/apply-prompt      apply a prompt to already-extracted items
  POST /api/chat              conversational operations (deterministic + AI)
  POST /api/match-products    fuzzy product matching
  POST /api/extract-text      return raw text of a file

Design principles:
  1. Deterministic first, AI as helper.
  2. Never invent values.
  3. Every response is well-formed JSON, even on failure.
  4. Hard wall-clock budget on every request.

ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import io, os, re, json, math, time, logging, statistics, tempfile
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
    import openpyxl  # noqa
except Exception:
    openpyxl = None

# Chat engine (deterministic-first operations)
try:
    from chat_engine import process_message as chat_process_message
    _CHAT_ENGINE_AVAILABLE = True
except Exception as e:
    logging.getLogger("smart-document-engine").warning(
        f"chat_engine not available: {e}")
    chat_process_message = None
    _CHAT_ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-document-engine")

TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

ENGINE_VERSION = "13.2.0"

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
MIN_MATCH_CONFIDENCE = float(os.getenv("MIN_MATCH_CONFIDENCE", "0.84"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "12"))
ANALYZE_TIMEOUT_SECONDS = int(os.getenv("ANALYZE_TIMEOUT_SECONDS", "60"))
OCR_DPI = int(os.getenv("OCR_DPI", "150"))


# ===========================================================================
#  PYDANTIC SCHEMAS
# ===========================================================================

class ExtractedLineItem(BaseModel):
    product_name: str = Field(..., min_length=1)
    variety: Optional[str] = None
    farm_code: Optional[str] = None
    boxes: Optional[float] = None
    pack_rate: Optional[float] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None
    specification: Dict[str, Any] = Field(default_factory=dict)
    raw_values: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    validation_warnings: List[str] = Field(default_factory=list)
    row_index: Optional[int] = None

    @field_validator("product_name", "variety", mode="before")
    @classmethod
    def strip_prices(cls, value, info):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if re.search(r"[\$€£]|(?:\b(?:USD|KES|EUR|GBP|AED)\b)", text, re.I):
            if re.search(r"\b(?:price|rate|cost|total|amount|unit\s*price)\b",
                         text, re.I):
                text = re.sub(r"[\$€£]\s*\d+(?:\.\d+)?", "", text)
                text = re.sub(
                    r"(?i)\b(?:price|rate|cost|total|amount)\b\s*[:=]?\s*[\d,.]+",
                    "", text)
                text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) >= 2 else None

    @field_validator("boxes", "pack_rate", "quantity", mode="before")
    @classmethod
    def to_num(cls, value):
        if value is None or value == "":
            return None
        try:
            f = float(value)
            return int(f) if f.is_integer() else f
        except (ValueError, TypeError):
            return None

    @field_validator("unit_price", "total", mode="before")
    @classmethod
    def to_float(cls, value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @model_validator(mode="after")
    def reconcile(self):
        w = list(self.validation_warnings)
        if self.quantity and self.unit_price and self.total:
            expected = self.quantity * self.unit_price
            if abs(expected - self.total) > max(0.05, abs(self.total) * 0.02):
                w.append("quantity_x_unit_price_does_not_match_total")
        if self.boxes and self.pack_rate and self.quantity:
            expected = self.boxes * self.pack_rate
            if abs(expected - self.quantity) > 0.5:
                w.append("boxes_x_pack_rate_does_not_match_quantity")
        self.validation_warnings = sorted(set(w))
        return self


class MatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    company_products: List[Dict[str, Any]]


class PromptRequest(BaseModel):
    items: List[Dict[str, Any]] = []
    columns: List[Dict[str, Any]] = []
    prompt: str = ""


class ChatRequest(BaseModel):
    items: List[Dict[str, Any]] = []
    columns: List[Dict[str, Any]] = []
    message: str = ""


# ===========================================================================
#  HELPERS
# ===========================================================================

def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s.strip().lower())


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


def as_number(v):
    n = parse_number(v)
    if n is None:
        return None
    return int(n) if float(n).is_integer() else n


def empty_field(v) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s == "" or s in {"n/a", "na", "null", "none", "-", "—", "?"}


def safe_sum(values) -> float:
    total = 0.0
    for v in values:
        if v is None:
            continue
        try:
            total += float(v)
        except Exception:
            continue
    return total


# ===========================================================================
#  COLUMN CLASSIFICATION (roles)
# ===========================================================================
CANONICAL_ROLES = {
    "product_name": [
        "product", "product name", "item", "item name", "flower", "flower name",
        "description", "particulars", "goods", "articles", "flower variety",
        "variety", "cultivar", "species",
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
    if not label:
        return None
    l = re.sub(r"[^a-z0-9 _\-]|_", " ", str(label).lower()).strip()
    l = re.sub(r"\s+", " ", l)
    if not l:
        return None
    for role, names in CANONICAL_ROLES.items():
        for n in names:
            if l == re.sub(r"[^a-z0-9 _\-]", " ", n.lower()).strip():
                return role
    best_role, best_score = None, 0
    for role, names in CANONICAL_ROLES.items():
        for n in names:
            s = fuzz.token_set_ratio(l, re.sub(r"[^a-z0-9 _\-]", " ", n.lower()).strip())
            if s > best_score:
                best_score, best_role = s, role
    return best_role if best_score >= 88 else None


def slugify(label: str, taken: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "col"
    key = base
    i = 2
    while key in taken:
        key = f"{base}_{i}"
        i += 1
    taken.add(key)
    return key


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
    have = {c.get("role") for c in columns if c.get("role")}
    taken = {c["key"] for c in columns}
    for role, label in [("unit_price", "Unit Price"), ("total", "Line Total")]:
        if role in have:
            continue
        key = slugify(label, taken)
        columns.append(make_column(key, label, role, "added"))
        have.add(role)
    return columns


# ===========================================================================
#  DYNAMIC EXTRACTION — Excel/CSV
# ===========================================================================
def find_header_row(df: pd.DataFrame) -> int:
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
    raw_headers = [str(v).strip() if pd.notna(v) else ""
                   for v in df.iloc[header_row].tolist()]
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
            if isinstance(val, (int, float)):
                record[col["key"]] = (int(val) if float(val).is_integer()
                                      else float(val))
            else:
                record[col["key"]] = str(val).strip()
        if empty:
            continue

        first_val = next((v for v in record.values() if v), "")
        if isinstance(first_val, str) and first_val.lower().strip() in (
                "total", "subtotal", "grand total"):
            continue

        for col in columns:
            role = col.get("role")
            if role and role not in record:
                record[role] = record.get(col["key"])
            elif role and record.get(role) in (None, ""):
                record[role] = record.get(col["key"])

        for col in columns:
            if col.get("role") in ("quantity", "boxes", "pack_rate",
                                    "length_cm", "head_size_cm"):
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
#  DYNAMIC EXTRACTION — text / PDF / DOCX
# ===========================================================================
def extract_from_text_dynamic(text: str) -> Dict[str, Any]:
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return {"columns": [], "items": []}

    best_idx, best_score, best_cells = None, 0, []
    for i, line in enumerate(lines[:30]):
        cells = [c.strip() for c in re.split(r"\s*\|\s*|\t+|\s{2,}", line)
                 if c.strip()]
        if len(cells) < 2:
            cells = line.split()
        if len(cells) < 2:
            continue
        score = sum(1 for c in cells if classify_column(c) is not None)
        if score > best_score:
            best_score, best_idx, best_cells = score, i, cells

    if best_idx is None or best_score == 0:
        return {
            "columns": [make_column("item", "Item", None, "original")],
            "items": [{"item": l, "product_name": l} for l in lines],
        }

    columns = build_columns_from_headers(best_cells)
    expected = len(columns)

    items: List[Dict[str, Any]] = []
    for line in lines[best_idx + 1:]:
        low = line.lower()
        if any(k in low for k in ("thank you", "computer-generated", "signature",
                                   "subtotal", "grand total", "total:", "notes:")):
            continue
        cells = [c.strip() for c in re.split(r"\s*\|\s*|\t+|\s{2,}", line)
                 if c.strip()]
        if len(cells) < 2:
            cells = line.split()
        if len(cells) < 2:
            continue

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
#  PDF / DOCX / IMAGE READERS
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

    if fitz is None:
        return ""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        parts = []
        for idx, page in enumerate(doc):
            if idx >= MAX_OCR_PAGES:
                break
            pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = img.convert("L")
            img = ImageEnhance.Contrast(img).enhance(1.5)
            try:
                t = pytesseract.image_to_string(
                    img, lang="eng", config="--oem 3 --psm 6", timeout=15)
                if t:
                    parts.append(t)
            except Exception:
                continue
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""


def extract_docx_text(content: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(content))
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


def ocr_image_simple(img: Image.Image) -> str:
    img = img.convert("L")
    w, h = img.size
    longest = max(w, h)
    if longest < 1800:
        s = 1800 / longest
        img = img.resize((int(w * s), int(h * s)))
    img = ImageEnhance.Contrast(img).enhance(1.6)
    for cfg in ("--oem 3 --psm 6", "--oem 3 --psm 4"):
        try:
            txt = pytesseract.image_to_string(img, lang="eng", config=cfg,
                                              timeout=20)
            if txt and len(re.findall(r"[A-Za-z0-9]", txt)) >= 20:
                return txt
        except Exception:
            continue
    return ""


def extract_image_text(content: bytes) -> str:
    try:
        return ocr_image_simple(Image.open(io.BytesIO(content)))
    except Exception:
        return ""


# ===========================================================================
#  CANONICAL ITEM VIEW
# ===========================================================================
def canonical_item(record: Dict[str, Any],
                   columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    def get_by_role(role: str):
        for c in columns:
            if c.get("role") == role:
                return record.get(c["key"])
        return None

    def to_num(x):
        if x is None:
            return None
        try:
            f = float(x)
            return int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            return None

    name = get_by_role("product_name") or ""
    qty = to_num(get_by_role("quantity"))
    boxes = to_num(get_by_role("boxes"))
    pack_rate = to_num(get_by_role("pack_rate"))
    unit_price = to_num(get_by_role("unit_price"))
    total = to_num(get_by_role("total"))

    if total is None and qty is not None and unit_price is not None:
        total = round(qty * unit_price, 4)

    return {
        "product_name": str(name).strip() if name else "",
        "variety": None,
        "boxes": boxes,
        "pack_rate": pack_rate,
        "quantity": qty,
        "unit_price": unit_price,
        "total": total,
    }


# ===========================================================================
#  AI PROMPT APPLICATION (for the /api/apply-prompt endpoint)
# ===========================================================================
PROMPT_SYSTEM = """You are a strict data-transformation assistant for invoice tables.

You receive:
  - columns: [{key, label, role, source}, ...]
  - items: rows, keyed by column.key
  - prompt: a natural-language instruction

Return ONLY valid JSON:
{
  "items": [ <same length as input, same keys, updated values> ],
  "columns": [ <same columns array, optionally with new 'added' columns> ],
  "explanation": "<one short sentence>"
}

Rules:
1. NEVER delete rows unless the prompt says to.
2. NEVER change the number of columns unless adding one (source must be "added").
3. Preserve every existing key in every row.
4. Numbers must be numbers, not strings.
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
    if not AI_ENABLED:
        return None
    payload = json.dumps({"columns": columns, "items": items, "prompt": prompt},
                         ensure_ascii=False)[:30000]

    # Try Gemini
    if AI_PROVIDER in ("gemini", "auto"):
        try:
            model = _get_gemini()
            if model is not None:
                resp = model.generate_content(
                    PROMPT_SYSTEM + "\n\nINPUT:\n" + payload,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "max_output_tokens": 8192,
                    },
                )
                data = _clean_json(resp.text if resp else "")
                if data and "items" in data:
                    return data
        except Exception as e:
            logger.warning(f"Prompt via gemini failed: {e}")

    # Try Groq
    if AI_PROVIDER in ("groq", "auto"):
        try:
            client = _get_groq()
            if client is not None:
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
            logger.warning(f"Prompt via groq failed: {e}")
    return None


def apply_prompt_deterministic(items, columns, prompt):
    """Simple deterministic prompt handler (fallback)."""
    p = prompt.lower().strip()
    cols = list(columns)
    taken = {c["key"] for c in cols}

    # add unit price column at X
    m = re.search(r"(?:add|create)\s+(?:a\s+)?(?:unit\s*price|price)\s*(?:column)?"
                  r"(?:\s*(?:at|=|of|:)\s*([0-9]+(?:\.[0-9]+)?))?", p)
    if m:
        default_price = float(m.group(1)) if m.group(1) else None
        if not any(c.get("role") == "unit_price" for c in cols):
            key = slugify("Unit Price", taken)
            cols.append(make_column(key, "Unit Price", "unit_price", "added"))
            for it in items:
                it[key] = default_price

    # calculate line total
    if re.search(r"\b(total|line total|line amount)\b", p) and \
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

    # remove column X
    m = re.search(r"(?:remove|delete)\s+(?:the\s+)?column\s+([a-z0-9 _\-]+)", p)
    if m:
        target = m.group(1).strip()
        for c in list(cols):
            if target in c["label"].lower() or target == c["key"]:
                for it in items:
                    it.pop(c["key"], None)
                cols.remove(c)

    # multiply quantity by N
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
#  PRODUCT MATCHING
# ===========================================================================
def normalize_product_for_match(name: str) -> str:
    n = norm(name)
    n = re.sub(r"\b\d+(?:\.\d+)?\s*cm\b", " ", n)
    n = re.sub(r"\b(?:box|boxes|bx|carton|cartons|qty|quantity|stems?)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


# ===========================================================================
#  AI TEXT / IMAGE FALLBACKS
# ===========================================================================
AI_TEXT_PROMPT = """Extract line items from this document text. Return ONLY JSON:
{"items":[{"product_name":"...","quantity":null,"unit_price":null,"total":null}]}
Never put numbers or prices inside product_name. Never invent values."""

AI_VISION_PROMPT = """Read this invoice image. Return ONLY JSON:
{"items":[{"product_name":"...","quantity":null,"unit_price":null,"total":null}]}
Never invent values. Use null for missing fields."""


def extract_text_with_ai(text: str) -> List[Dict[str, Any]]:
    if not AI_ENABLED or not text or len(text.strip()) < 20:
        return []
    prompt = AI_TEXT_PROMPT + "\n\nDocument:\n" + text[:12000]
    for provider in (("gemini", _get_gemini), ("groq", _get_groq)):
        name, getter = provider
        client = getter()
        if client is None:
            continue
        try:
            if name == "gemini":
                resp = client.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0,
                                       "response_mime_type": "application/json",
                                       "max_output_tokens": 4096})
                raw = resp.text if resp else ""
            else:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=4096)
                raw = resp.choices[0].message.content if resp.choices else ""
            data = _clean_json(raw)
            if data and "items" in data:
                out = []
                for row in data["items"]:
                    n = str(row.get("product_name") or "").strip()
                    if len(n) < 2:
                        continue
                    out.append({
                        "product_name": n,
                        "boxes": row.get("boxes"),
                        "pack_rate": row.get("pack_rate"),
                        "quantity": row.get("quantity"),
                        "unit_price": row.get("unit_price"),
                        "total": row.get("total"),
                        "specification": {},
                    })
                if out:
                    return out
        except Exception as e:
            logger.warning(f"AI text fallback via {name} failed: {e}")
    return []


def extract_image_with_ai(content: bytes, mime: str = "image/jpeg"):
    if not AI_ENABLED or not GEMINI_API_KEY:
        return []
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(
            [{"mime_type": mime, "data": content}, AI_VISION_PROMPT],
            generation_config={"temperature": 0.0,
                               "response_mime_type": "application/json",
                               "max_output_tokens": 4096})
        data = _clean_json(resp.text if resp else "")
        if not data or "items" not in data:
            return []
        out = []
        for row in data["items"]:
            n = str(row.get("product_name") or "").strip()
            if len(n) < 2:
                continue
            out.append({
                "product_name": n,
                "boxes": row.get("boxes"),
                "pack_rate": row.get("pack_rate"),
                "quantity": row.get("quantity"),
                "unit_price": row.get("unit_price"),
                "total": row.get("total"),
                "specification": {},
            })
        return out
    except Exception as e:
        logger.warning(f"Gemini vision failed: {e}")
        return []


# ===========================================================================
#  AI ANALYSIS ENVELOPE
# ===========================================================================
def confidence_band(c: float) -> str:
    if c >= 0.90: return "high"
    if c >= 0.75: return "medium"
    if c >= 0.55: return "low"
    return "review_required"


def ai_analyze(items: List[Dict], meta: Dict) -> Dict[str, Any]:
    insights, anomalies = [], []
    if not items:
        insights.append({"type": "no_items_detected", "severity": "high",
                         "message": "No line items could be extracted."})
    prices = [float(i["unit_price"]) for i in items if i.get("unit_price")]
    if len(prices) >= 3:
        med = statistics.median(prices)
        for i, it in enumerate(items):
            p = it.get("unit_price")
            if p is None or med == 0:
                continue
            dev = abs(float(p) - med) / med
            if dev >= 0.5:
                insights.append({
                    "type": "price_outlier", "severity": "medium",
                    "item_index": i,
                    "product_name": it.get("product_name"),
                    "message": f"Unit price {p} deviates {dev*100:.0f}% from median.",
                })
    for i, it in enumerate(items):
        for w in it.get("warnings", []):
            anomalies.append({"item_index": i,
                              "product_name": it.get("product_name"),
                              "code": w,
                              "severity": "high" if "not_match" in w else "medium"})
    recs = []
    if any(i["type"] == "no_items_detected" for i in insights):
        recs.append("Re-upload a higher-resolution scan, or paste the text directly.")
    if any(i["type"] == "price_outlier" for i in insights):
        recs.append("Cross-check outlier prices against your rate card.")
    if not recs:
        recs.append("Extraction looks consistent. Proceed to product matching.")
    trust_base = (sum(i.get("confidence", 0.5) for i in items) / len(items)) if items else 0.0
    penalty = sum(0.10 if x.get("severity") == "high" else
                  0.04 if x.get("severity") == "medium" else 0
                  for x in insights)
    trust = round(max(0.0, min(1.0, trust_base - penalty)), 3)
    return {"trust_score": trust,
            "confidence_band": confidence_band(trust),
            "reasoning": f"Parsed {len(items)} item(s).",
            "insights": insights, "anomalies": anomalies,
            "recommendations": recs,
            "analyzed_at": datetime.utcnow().isoformat() + "Z"}


# ===========================================================================
#  ORCHESTRATION
# ===========================================================================
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
        text_extracted = extract_image_text(content)
        result = extract_from_text_dynamic(text_extracted)
        result["extraction_method"] = "image_ocr_dynamic"

    else:
        text_extracted = content.decode("utf-8", errors="ignore")
        result = extract_from_text_dynamic(text_extracted)
        result["extraction_method"] = "text_dynamic"

    # If nothing found, try AI text fallback
    if not result.get("items") and text_extracted and AI_ENABLED:
        ai_items = extract_text_with_ai(text_extracted)
        if ai_items:
            # Build a flat column schema for the AI output
            result["columns"] = build_columns_from_headers(
                ["Product name", "Boxes", "Pack rate", "Quantity",
                 "Unit price", "Line total"])
            flat = []
            for it in ai_items:
                rec = {
                    result["columns"][0]["key"]: it.get("product_name"),
                    result["columns"][1]["key"]: it.get("boxes"),
                    result["columns"][2]["key"]: it.get("pack_rate"),
                    result["columns"][3]["key"]: it.get("quantity"),
                    result["columns"][4]["key"]: it.get("unit_price"),
                    result["columns"][5]["key"]: it.get("total"),
                }
                flat.append(rec)
            result["items"] = flat
            result["extraction_method"] += "+ai_text"

    # If still nothing and image, try Gemini Vision
    if (not result.get("items") and AI_ENABLED
        and ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp")):
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        ai_items = extract_image_with_ai(content, mime)
        if ai_items:
            result["columns"] = build_columns_from_headers(
                ["Product name", "Boxes", "Pack rate", "Quantity",
                 "Unit price", "Line total"])
            flat = []
            for it in ai_items:
                rec = {
                    result["columns"][0]["key"]: it.get("product_name"),
                    result["columns"][1]["key"]: it.get("boxes"),
                    result["columns"][2]["key"]: it.get("pack_rate"),
                    result["columns"][3]["key"]: it.get("quantity"),
                    result["columns"][4]["key"]: it.get("unit_price"),
                    result["columns"][5]["key"]: it.get("total"),
                }
                flat.append(rec)
            result["items"] = flat
            result["extraction_method"] += "+gemini_vision"

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

    # Canonical view
    canonical_items = []
    for r in result["items"]:
        ci = canonical_item(r, result["columns"])
        ci["_record"] = r
        canonical_items.append(ci)

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


# ===========================================================================
#  ENDPOINTS
# ===========================================================================

@app.get("/")
async def root():
    return {
        "service": "Smart Document Intelligence Engine",
        "version": ENGINE_VERSION,
        "status": "operational",
        "ai_enabled": AI_ENABLED,
        "ai_provider": AI_PROVIDER,
        "chat_engine_available": _CHAT_ENGINE_AVAILABLE,
        "endpoints": [
            "/api/ping", "/api/health", "/api/analyze",
            "/api/apply-prompt", "/api/chat",
            "/api/match-products", "/api/extract-text",
        ],
    }


@app.get("/api/ping")
async def ping():
    return {
        "ok": True,
        "version": ENGINE_VERSION,
        "ai_enabled": AI_ENABLED,
        "ai_provider": AI_PROVIDER,
        "gemini_ready": bool(GEMINI_API_KEY),
        "groq_ready": bool(GROQ_API_KEY),
        "chat_engine_available": _CHAT_ENGINE_AVAILABLE,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": ENGINE_VERSION,
        "ocr_available": bool(pytesseract),
        "pdf_available": fitz is not None,
        "ai_enabled": AI_ENABLED,
        "ai_provider": AI_PROVIDER,
        "gemini_ready": bool(GEMINI_API_KEY),
        "groq_ready": bool(GROQ_API_KEY),
        "chat_engine_available": _CHAT_ENGINE_AVAILABLE,
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

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(analyze_bytes, content, fname, ext, company_id, prompt)
            try:
                return await loop.run_in_executor(
                    None, lambda: fut.result(timeout=ANALYZE_TIMEOUT_SECONDS))
            except Exception as te:
                logger.warning(f"Analyze timed out: {te}")
                return {"success": False, "columns": [], "items": [],
                        "raw_items": [], "item_count": 0,
                        "error": "Analysis timed out.",
                        "engine_version": ENGINE_VERSION,
                        "extraction_method": "timeout"}
    except Exception as e:
        logger.exception("Analyze crashed")
        return {"success": False, "columns": [], "items": [],
                "raw_items": [], "item_count": 0,
                "error": str(e), "engine_version": ENGINE_VERSION,
                "extraction_method": "crashed"}


@app.post("/api/apply-prompt")
async def apply_prompt_endpoint(req: PromptRequest):
    items = req.items or []
    columns = req.columns or []
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"success": True, "items": items, "columns": columns,
                "raw_items": items, "explanation": "No prompt provided."}

    applied = apply_prompt_ai(items, columns, prompt)
    if not applied:
        applied = apply_prompt_deterministic(items, columns, prompt)

    canonical = []
    for r in applied["items"]:
        ci = canonical_item(r, applied["columns"])
        ci["_record"] = r
        canonical.append(ci)

    return {"success": True,
            "columns": applied["columns"],
            "items": canonical,
            "raw_items": applied["items"],
            "explanation": applied.get("explanation", ""),
            "applied_via": "ai" if AI_ENABLED else "deterministic"}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Conversational operations on the current table.
    Deterministic first; AI only as fallback; always well-formed JSON.
    """
    items = req.items or []
    columns = req.columns or []
    message = req.message or ""

    if not _CHAT_ENGINE_AVAILABLE or chat_process_message is None:
        return {
            "success": True,
            "items": items,
            "columns": columns,
            "explanation": "Chat engine is not loaded on the server.",
            "applied_via": "error",
        }

    try:
        return chat_process_message(items, columns, message)
    except Exception as e:
        logger.exception("chat failed")
        return {
            "success": True,
            "items": items,
            "columns": columns,
            "explanation": "Something went wrong while processing that instruction.",
            "applied_via": "error",
        }


@app.post("/api/match-products")
async def match_products_endpoint(req: MatchRequest):
    try:
        if not req.items or not req.company_products:
            return {"success": True, "items": req.items,
                    "matched_count": 0, "review_count": 0}
        out = []
        for item in req.items:
            iname = normalize_product_for_match(str(item.get("product_name", "")))
            ranked = []
            for product in req.company_products:
                names = [str(product.get("name", ""))]
                aliases = product.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(str(x) for x in aliases)
                for cand in names:
                    cn = normalize_product_for_match(cand)
                    if not cn:
                        continue
                    score = (100.0 if iname == cn else
                             max(fuzz.ratio(iname, cn),
                                 fuzz.token_set_ratio(iname, cn),
                                 fuzz.WRatio(iname, cn)))
                    ranked.append((score, product))
            ranked.sort(key=lambda x: x[0], reverse=True)
            best = ranked[0] if ranked else (0, None)
            second = ranked[1][0] if len(ranked) > 1 else 0
            conf = best[0] / 100.0
            margin_ok = (best[0] - second) >= 6.0 or best[0] >= 98.0
            accepted = (best[1] is not None and
                        conf >= MIN_MATCH_CONFIDENCE and margin_ok)
            out.append({
                **item,
                "product_id": best[1].get("id") if accepted else None,
                "matched_product_name": best[1].get("name") if accepted else None,
                "match_confidence": round(conf, 3),
                "match_status": "matched" if accepted else "review_required",
            })
        return {
            "success": True, "items": out,
            "matched_count": sum(1 for x in out if x.get("match_status") == "matched"),
            "review_count": sum(1 for x in out if x.get("match_status") == "review_required"),
        }
    except Exception as e:
        logger.exception("Product matching failed")
        raise HTTPException(500, str(e))


@app.post("/api/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    fname = file.filename or ""
    ext = Path(fname).suffix.lstrip(".").lower()

    if ext == "pdf":
        text = extract_pdf_text(content); method = "pdf"
    elif ext in ("docx", "doc"):
        text = extract_docx_text(content); method = "docx"
    elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
        text = extract_image_text(content); method = "image_ocr"
    else:
        text = content.decode("utf-8", errors="ignore"); method = "text"

    return {"success": True, "text": text, "length": len(text),
            "file_type": ext, "extraction_method": method}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
