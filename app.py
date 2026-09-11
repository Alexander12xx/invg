"""
Smart Document Intelligence Engine v12.0
Production-grade extraction with optional Docling deep-extraction.

Architecture:
  1. FAST PATH: PyMuPDF text + regex table parser (our proven engine, <2s)
  2. DEEP PATH: Docling layout-aware extraction (optional, env-gated)
  3. VALIDATION: Pydantic arithmetic checks (never trusts printed totals)

Docling requirements: 8GB+ RAM. Enable with DOCLING_ENABLED=1.
On Render free tier (512MB), leave DOCLING_ENABLED=0 (default).

API contract preserved — no PHP or DB changes needed.

ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import io, os, re, json, math, time, logging, statistics
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import PyPDF2
import docx
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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

# Optional Docling — only import if explicitly enabled
DOCLING_ENABLED = os.getenv("DOCLING_ENABLED", "0") == "1"
DoclingConverter = None
if DOCLING_ENABLED:
    try:
        from docling.document_converter import DocumentConverter
        DoclingConverter = DocumentConverter
        logging.info("Docling enabled for deep extraction")
    except Exception as e:
        logging.warning(f"Docling requested but failed to load: {e}")
        DOCLING_ENABLED = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-document-engine")

TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

ENGINE_VERSION = "12.0.0"

app = FastAPI(
    title="Smart Document Intelligence Engine",
    version=ENGINE_VERSION,
    description="Layout-aware extraction with optional Docling deep path.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MIN_MATCH_CONFIDENCE = float(os.getenv("MIN_MATCH_CONFIDENCE", "0.84"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "30"))
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "12"))
ANALYZE_TIMEOUT_SECONDS = int(os.getenv("ANALYZE_TIMEOUT_SECONDS", "45"))


# ===========================================================================
#  PYDANTIC SCHEMA (with arithmetic validation from your uploaded file)
# ===========================================================================

class ExtractedLineItem(BaseModel):
    """Strictly typed line item. Prices cannot pollute text fields."""
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
        # If it contains currency symbols or price labels, strip them
        if re.search(r"[\$€£]|(?:\b(?:USD|KES|EUR|GBP|AED)\b)", text, re.I):
            if re.search(r"\b(?:price|rate|cost|total|amount|unit\s*price)\b", text, re.I):
                text = re.sub(r"[\$€£]\s*\d+(?:\.\d+)?", "", text)
                text = re.sub(r"(?i)\b(?:price|rate|cost|total|amount)\b\s*[:=]?\s*[\d,.]+", "", text)
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
        """Arithmetic validation — never trust printed totals without checking."""
        w = list(self.validation_warnings)
        # quantity × unit_price should equal total
        if self.quantity and self.unit_price and self.total:
            expected = self.quantity * self.unit_price
            if abs(expected - self.total) > max(0.05, abs(self.total) * 0.02):
                w.append("quantity_x_unit_price_does_not_match_total")
        # boxes × pack_rate should equal quantity
        if self.boxes and self.pack_rate and self.quantity:
            expected = self.boxes * self.pack_rate
            if abs(expected - self.quantity) > 0.5:
                w.append("boxes_x_pack_rate_does_not_match_quantity")
        self.validation_warnings = sorted(set(w))
        return self


class MatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    company_products: List[Dict[str, Any]]


# ===========================================================================
#  HELPERS
# ===========================================================================

def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s.strip().lower())


def clean_ocr_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text


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
    elif re.search(r",\d{1,2}$", txt):
        txt = txt.replace(".", "").replace(",", ".")
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
#  FIELD KNOWLEDGE (expanded)
# ===========================================================================

COLUMN_SYNONYMS = {
    "product": ["product", "product name", "product/service", "product / service",
                "product or service", "item", "item name", "service", "article",
                "articles", "commodity", "goods", "stock item", "particulars",
                "product description", "item description", "description of goods",
                "flower", "flower variety"],
    "variety": ["variety", "flower", "flower name", "flower type", "species",
                "cultivar", "kind", "variety name", "flower variety"],
    "description": ["description", "desc", "details", "item details",
                    "specification", "specifications", "remarks",
                    "product details", "product / service description"],
    "farm_code": ["farm code", "farmcode", "farm reference", "farm ref",
                  "supplier code", "grower code", "grower reference"],
    "boxes": ["boxes", "box", "bx", "cartons", "carton", "ctn", "cases",
              "case", "bundles", "bundle", "packages", "pkg"],
    "pack_rate": ["packrate", "pack rate", "pack_rate", "per box", "per carton",
                  "stems per box", "stems/box", "qty per box",
                  "quantity per box", "stems per carton", "qty/carton"],
    "quantity": ["quantity", "qty", "qnty", "stems", "pcs", "pieces", "count",
                 "total quantity", "total qty", "number of stems",
                 "stem quantity", "invoice quantity", "total stems"],
    "unit_price": ["price", "price per stem", "price/stem", "unit price",
                   "unit price (usd)", "cost", "price per unit",
                   "per stem", "per piece", "amount per stem", "unit cost",
                   "rate per stem", "selling price", "unit selling price"],
    "total": ["total", "total price", "total amount", "line total",
              "line amount", "sub-total", "subtotal", "extended price",
              "line value", "amount"],
    "length": ["length", "length(cm)", "length (cm)", "size", "size(cm)",
               "stem length", "height", "stem size", "length cm"],
    "discount": ["discount", "disc.", "rebate"],
    "tax": ["tax", "vat", "gst", "sales tax"],
}

# Expanded NON_PRODUCT_TERMS — kills COMMERCIAL INVOICE and all headers
NON_PRODUCT_TERMS = {
    # Document titles / headers
    "invoice", "commercial invoice", "tax invoice", "proforma invoice",
    "proforma", "quotation", "quote", "estimate", "receipt", "payment receipt",
    "delivery note", "dispatch note", "packing list", "packing slip",
    "credit note", "credit memo", "purchase order", "statement",
    "invoice details", "invoice number", "order details",
    # Labels / column headers
    "consignee", "consignee details", "consignee name", "consignee address",
    "seller", "seller name", "seller/exporter", "exporter", "exporter name",
    "buyer", "buyer name", "buyer address", "customer", "customer name",
    "customer details", "customer address", "bill to", "ship to",
    "sold to", "deliver to",
    "payment terms", "payment term", "terms of payment",
    "transportation", "transport", "shipment method",
    "notes", "note", "comments", "comment", "remarks",
    "items", "products", "product/service", "product / service",
    "product or service", "particulars",
    "description", "product description", "service description",
    "product / service description",
    "variety", "flower", "flower variety", "flower name",
    "quantity", "qty", "stems", "total stems",
    "price", "unit price", "price per stem", "unit price (usd)",
    "total", "amount", "line total", "line amount",
    "boxes", "packrate", "pack rate", "pack_rate", "length",
    "farm code", "farmcode",
    "country of destination", "destination", "destination country",
    "country of origin", "origin country",
    "point of entry", "port of entry", "port", "airport",
    "date of shipment", "shipment date", "invoice date", "issue date",
    "due date", "payment due", "valid until", "expiry",
    "currency", "currency code", "vat", "tax", "vat rate", "tax rate",
    "subtotal", "sub-total", "grand total", "balance due",
    "awb", "awb number", "awb fee", "air waybill",
    "net weight", "gross weight", "net kg", "gross kg",
}

COMPANY_TERMS = re.compile(
    r"\b(limited|ltd\.?|llc|inc\.?|plc|company|enterprises?|"
    r"investment|trading|holdings?)\b", re.I)

ADDRESS_TERMS = re.compile(
    r"\b(street|st\.|road|rd\.|avenue|ave\.|building|bldg|floor|"
    r"suite|tower|plaza|po box|postal code|p\.o\.)\b", re.I)

PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-().]{7,}\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9\-]+\.(?:com|net|org|io|ke|co\.ke|tc|info|biz)\b", re.I)
HASH_ID_RE = re.compile(r"^#?[A-Z]{2,}\d{4,}[A-Z0-9\-]*$", re.I)


def is_boilerplate(line: str) -> bool:
    if not line:
        return True
    s = line.strip()
    if re.search(r"computer[- ]generated|no\s+signature\s+required|"
                 r"^\s*thank\s+you|^\s*generated\s+on|^\s*scan\s+to\s+verify|"
                 r"^\s*page\s+\d+|^\s*paid\s*$|^\s*sent\s*$|"
                 r"^\s*(?:sub[- ]?total|grand\s+total|balance\s+due)\b|"
                 r"prices?\s+are\s+inclusive|delivery\s+cost\s+is\s+the\s+buyer",
                 s, re.I):
        return True
    if URL_RE.search(s) or EMAIL_RE.search(s):
        return True
    if HASH_ID_RE.match(s):
        return True
    return False


def is_contact_or_address(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    if PHONE_RE.search(s) and len(re.findall(r"\d", s)) >= 7:
        return True
    if EMAIL_RE.search(s) or URL_RE.search(s):
        return True
    if ADDRESS_TERMS.search(s) and len(s.split()) <= 8:
        return True
    return False


HEADER_WORDS = {
    "flower", "variety", "length", "pack", "rate", "boxes", "box",
    "total", "stems", "stem", "unit", "price", "amount", "qty",
    "quantity", "description", "product", "service", "item", "no",
    "cartons", "carton", "bundles", "bundle", "val", "value",
}


def looks_like_product(name: str) -> bool:
    n = str(name or "").strip()
    if len(n) < 2 or not re.search(r"[A-Za-z]{3,}", n):
        return False
    low = norm(n)
    if low in NON_PRODUCT_TERMS:
        return False
    # Reject rows made entirely of header tokens
    toks = re.findall(r"[A-Za-z]+", n.lower())
    if toks and all(t in HEADER_WORDS for t in toks):
        return False
    if "@" in n or URL_RE.search(n):
        return False
    if ADDRESS_TERMS.search(n):
        return False
    if COMPANY_TERMS.search(n) and not re.search(
            r"\b(rose|roses|flower|flowers|plant|goods|supplies|"
            r"celocia|carnation|chrysanthemum|tulip|lily|orchid|gerbera|"
            r"alstroemeria|alstromeria|sunflower|eustoma|hydrangea|"
            r"birds?\s+of\s+paradise|strelitzia)\b", n, re.I):
        return False
    if is_boilerplate(n) or is_contact_or_address(n):
        return False
    if re.fullmatch(r"[\d\s.,:/()%$€£+\-]+", n):
        return False
    return True


def is_header_or_metadata(line: str) -> bool:
    s = str(line or "").strip()
    if not s:
        return True
    if re.match(r"^(?:invoice|inv|quote|qtn|proforma|po|flr)[\s/#-]*[\w/-]+$", norm(s)):
        return True
    if re.fullmatch(r"[\d\s.,:/()%$€£-]+", s):
        return True
    return False


# ===========================================================================
#  HEADER MATCHING
# ===========================================================================

def match_header(label: str) -> Optional[str]:
    l = norm(label)
    l = re.sub(r"^[#*]+", "", l)
    l = re.sub(r"[*:.#]+$", "", l).strip()
    if not l:
        return None
    if l in {"#", "no", "no.", "s/n", "sn", "sr"}:
        return "row_index"
    for field, names in COLUMN_SYNONYMS.items():
        if l in {norm(x) for x in names}:
            return field
    priority = ["pack_rate", "unit_price", "farm_code", "quantity", "boxes",
                "length", "total", "product", "variety", "description"]
    for field in priority:
        for n in COLUMN_SYNONYMS[field]:
            nn = norm(n)
            if len(nn) >= 5 and (nn in l or l in nn):
                return field
    best_field, best_score = None, 0
    for field, names in COLUMN_SYNONYMS.items():
        for n in names:
            s = fuzz.token_set_ratio(l, norm(n))
            if s > best_score:
                best_score, best_field = s, field
    return best_field if best_score >= 88 else None


# ===========================================================================
#  HEADER TOKENIZATION (the fix for two-line headers)
# ===========================================================================

HEADER_TOKEN_SET = {
    "flower", "variety", "length", "pack", "rate", "boxes", "box",
    "total", "stems", "stem", "unit", "price", "amount", "qty",
    "quantity", "description", "product", "service", "item",
    "cartons", "carton", "bundles", "bundle", "no", "val", "value",
}


def _tokenize_header_block(block_text: str) -> List[str]:
    """Split a header block into column-ish tokens with two-word merging."""
    tokens = re.findall(r"[A-Za-z#/]+", block_text)
    tokens = [t for t in tokens if t.lower() in HEADER_TOKEN_SET or t == "#"]
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        pair = (cur.lower(), nxt.lower()) if nxt else ("", "")
        if pair in {
            ("unit", "price"), ("pack", "rate"), ("line", "total"),
            ("total", "stems"), ("total", "amount"), ("flower", "variety"),
            ("product", "description"), ("product", "service"),
            ("service", "description"),
        }:
            merged.append(f"{cur} {nxt}")
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged


def find_table_header(lines: List[str]) -> Tuple[Optional[int], List[Optional[str]]]:
    """Find a header using 1..4 adjacent physical lines. Handles two-line headers."""
    best = None
    limit = min(len(lines), 200)
    for i in range(limit):
        for span in (2, 3, 1, 4):
            if i + span > limit:
                continue
            block = [x.strip() for x in lines[i:i+span] if x.strip()]
            if not block:
                continue
            # Try explicit separators first
            joined = " | ".join(block)
            cols = re.split(r"\s*\|\s*", joined)
            cols = [c.strip() for c in cols if c.strip()]
            # If too few, tokenize as header words
            if len(cols) < 3:
                cols = _tokenize_header_block(" ".join(block))
            # If still too few, try line-by-line
            if len(cols) < 3:
                for line in block:
                    cols.extend(_tokenize_header_block(line))
            mapped = [match_header(c) for c in cols]
            known = [m for m in mapped if m]
            unique = set(known)
            if (len(unique) >= 3 and
                any(x in unique for x in ("product", "variety", "description")) and
                any(x in unique for x in ("quantity", "boxes", "unit_price",
                                           "total", "pack_rate"))):
                score = len(unique) * 10 + len(known)
                if best is None or score > best[0]:
                    best = (score, i, mapped, span)
    if best:
        idx, mapped = best[1], best[2]
        # Drop leading row_index column if present
        if mapped and mapped[0] == "row_index":
            mapped = mapped[1:]
        return idx, mapped
    return None, []


# ===========================================================================
#  TABLE PARSING
# ===========================================================================

NUMERIC_FIELDS = {"boxes", "pack_rate", "quantity", "unit_price", "total"}


def split_columns(line: str) -> List[str]:
    line = str(line or "").strip()
    if not line:
        return []
    if "|" in line:
        return [x.strip() for x in re.split(r"\s*\|\s*", line) if x.strip()]
    if "\t" in line:
        return [x.strip() for x in line.split("\t") if x.strip()]
    parts = [x.strip() for x in re.split(r"\s{2,}", line) if x.strip()]
    return parts if len(parts) >= 2 else [line]


def strip_row_index(text: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^\s*(\d{1,3})\s*[.)]?\s+(.*\S)\s*$", text or "")
    if m and len(m.group(2)) >= 2:
        return m.group(2).strip(), int(m.group(1))
    return text, None


def _peel_numeric_tokens(s: str, max_peel: int) -> Tuple[str, List[str]]:
    """Peel up to max_peel numeric tokens from the RIGHT side of s."""
    nums: List[str] = []
    remaining = s.strip()
    num_re = re.compile(
        r"(?:\s|^)((?:[$€£]\s*)?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?"
        r"(?:\s*(?:KES|KSH|USD|EUR|GBP|AED|SAR|QAR))?)\s*$",
        re.IGNORECASE)
    for _ in range(max_peel):
        m = num_re.search(remaining)
        if not m or m.start(1) == 0:
            break
        nums.insert(0, m.group(1).strip())
        remaining = remaining[:m.start(1)].rstrip()
    return remaining.strip(), nums


def value_for_field(field: str, raw: str) -> Any:
    if empty_field(raw):
        return None
    if field in NUMERIC_FIELDS:
        return as_number(parse_number(raw))
    if field == "length":
        m = re.search(r"\d+(?:\.\d+)?", raw)
        return f"{m.group(0)}cm" if m else None
    return raw.strip()


def parse_table_row(line: str, headers: List[Optional[str]]) -> Optional[Dict[str, Any]]:
    cols = split_columns(line)

    # Single-cell rows (flattened PDFs) — peel numbers off the right
    if len(cols) < 2:
        num_slots = sum(1 for h in headers if h in NUMERIC_FIELDS or h == "length")
        head, nums = _peel_numeric_tokens(line, max_peel=max(1, num_slots))
        if not nums:
            return None
        head, ridx = strip_row_index(head)
        if not looks_like_product(head):
            return None
        cols = [head] + nums

    vals: Dict[str, Any] = {}
    raw_values: Dict[str, str] = {}
    for i, cell in enumerate(cols):
        field = headers[i] if i < len(headers) else None
        if not field:
            raw_values[f"unmapped_{i+1}"] = cell
            continue
        if field in vals and vals[field] not in (None, ""):
            raw_values[f"duplicate_{field}_{i+1}"] = cell
            continue
        vals[field] = value_for_field(field, cell)

    name = None
    for field in ("product", "variety", "description"):
        if vals.get(field) and looks_like_product(str(vals[field])):
            name = str(vals[field]).strip()
            break
    if not name:
        for cell in cols:
            if looks_like_product(cell) and not is_header_or_metadata(cell):
                name = cell
                break
    if not name:
        return None

    name, ridx = strip_row_index(name)

    item = {
        "product_name": name,
        "boxes": vals.get("boxes"),
        "pack_rate": vals.get("pack_rate"),
        "quantity": vals.get("quantity"),
        "unit_price": vals.get("unit_price"),
        "total": vals.get("total"),
        "specification": {},
    }
    if vals.get("length") is not None:
        item["specification"]["length"] = vals["length"]
    if vals.get("farm_code"):
        item["farm_code"] = str(vals["farm_code"]).strip()
    if ridx is not None:
        item["row_index"] = ridx
    if raw_values:
        item["raw_values"] = raw_values
    return item


# ===========================================================================
#  TEXT PARSER (multi-pass)
# ===========================================================================

def parse_order_text(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = clean_ocr_text(text)
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    items: List[Dict[str, Any]] = []

    # Pass 1: Table-based extraction
    header_idx, headers = find_table_header(lines)
    if header_idx is not None:
        for line in lines[header_idx + 1:]:
            if is_header_or_metadata(line) or is_boilerplate(line):
                continue
            item = parse_table_row(line, headers)
            if item:
                items.append(item)
        if items:
            return clean_items(items), extract_meta_from_text(text)

    # Pass 2: Free-form line-by-line
    for line in lines:
        if is_header_or_metadata(line) or is_boilerplate(line):
            continue
        # Try peeling numbers off right
        num_slots = 6
        head, nums = _peel_numeric_tokens(line, max_peel=num_slots)
        if len(nums) >= 2:
            head, ridx = strip_row_index(head)
            if looks_like_product(head):
                # Assign heuristically: last=total, 2nd-last=price, 3rd-last=qty
                item = {
                    "product_name": head,
                    "boxes": None, "pack_rate": None,
                    "quantity": as_number(nums[-3]) if len(nums) >= 3 else None,
                    "unit_price": as_number(nums[-2]),
                    "total": as_number(nums[-1]),
                    "specification": {},
                }
                if len(nums) >= 4:
                    item["pack_rate"] = as_number(nums[-4])
                if len(nums) >= 5:
                    item["boxes"] = as_number(nums[-5])
                if len(nums) >= 6:
                    item["specification"]["length"] = f"{int(float(nums[-6]))}cm"
                items.append(item)

    return clean_items(items), extract_meta_from_text(text)


# ===========================================================================
#  METADATA EXTRACTION
# ===========================================================================

META_LABELS = {
    "invoice_number": ["invoice number", "invoice no", "invoice #", "inv no",
                       "quotation number", "quote number", "proforma number",
                       "reference", "ref no"],
    "date": ["date", "invoice date", "issue date", "document date"],
    "due_date": ["due date", "payment due", "valid until"],
    "currency": ["currency", "currency code"],
    "consignee_name": ["consignee", "bill to", "ship to", "customer name"],
    "seller_name": ["seller", "vendor", "supplier", "exporter"],
    "purchase_order_no": ["purchase order", "po no", "order number"],
    "payment_terms": ["payment terms", "terms of payment"],
    "awb_number": ["awb number", "air waybill", "tracking number"],
    "net_weight": ["net weight", "net kg"],
    "notes": ["notes", "comments", "remarks"],
}

CURRENCY_WORDS = {
    "usd": "USD", "us dollar": "USD", "dollar": "USD",
    "kes": "KES", "ksh": "KES",
    "eur": "EUR", "euro": "EUR", "gbp": "GBP",
    "aed": "AED", "sar": "SAR", "qar": "QAR",
}

DOC_TYPES = {
    "invoice": ["invoice", "tax invoice", "commercial invoice"],
    "quotation": ["quotation", "quote", "estimate"],
    "proforma": ["proforma", "pro forma", "proforma invoice"],
    "receipt": ["receipt", "payment receipt"],
    "delivery_note": ["delivery note", "dispatch note"],
    "packing_list": ["packing list", "packing slip"],
    "credit_note": ["credit note", "credit memo"],
    "purchase_order": ["purchase order"],
}


def match_meta_label(label: str) -> Optional[str]:
    key = norm(label)
    key = re.sub(r"[*:#.]+$", "", key).strip()
    if not key:
        return None
    for meta, names in META_LABELS.items():
        for n in names:
            nn = norm(n)
            if key == nn or key.startswith(nn) or nn.startswith(key):
                return meta
    return None


def detect_document_type(text: str) -> Optional[str]:
    low = norm(text[:12000])
    scores = {t: max([fuzz.partial_ratio(low, norm(w)) for w in ws] or [0])
              for t, ws in DOC_TYPES.items()}
    typ, score = max(scores.items(), key=lambda x: x[1])
    return typ if score >= 70 else None


def extract_meta_from_text(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for i, line in enumerate(lines):
        m = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,70}?)\s*"
            r"(?:[:#]\s*|-\s+|\|\s*)(.*?)\s*$", line)
        if m:
            label, value = m.group(1), m.group(2).strip()
            field = match_meta_label(label)
            if field and value and not empty_field(value):
                meta[field] = value
    # Currency detection
    if "currency" in meta:
        c = norm(meta["currency"])
        for word, code in CURRENCY_WORDS.items():
            if word in c:
                meta["currency"] = code
                break
    else:
        for word, code in CURRENCY_WORDS.items():
            if re.search(rf"\b{word}\b", norm(text)):
                meta["currency"] = code
                break
    # Hash IDs
    if "invoice_number" not in meta:
        m = re.search(r"#\s*([A-Z]{2,}-?\d{4,}[A-Z0-9\-]*)", text)
        if m:
            meta["invoice_number"] = m.group(1)
    meta["document_type"] = detect_document_type(text)
    return meta


# ===========================================================================
#  VALIDATION / CLEANING
# ===========================================================================

def validate_item(item: Dict[str, Any]) -> List[str]:
    w = []
    b, p, q, u, t = (item.get("boxes"), item.get("pack_rate"),
                     item.get("quantity"), item.get("unit_price"),
                     item.get("total"))
    if b is not None and b <= 0: w.append("boxes_not_positive")
    if p is not None and p <= 0: w.append("pack_rate_not_positive")
    if q is not None and q <= 0: w.append("quantity_not_positive")
    if u is not None and u < 0: w.append("unit_price_negative")
    if t is not None and t < 0: w.append("total_negative")
    if q and u and t:
        try:
            if abs(float(q) * float(u) - float(t)) > max(0.05, abs(float(t)) * 0.02):
                w.append("quantity_x_unit_price_does_not_match_total")
        except Exception:
            pass
    if b and p and q:
        try:
            if abs(float(b) * float(p) - float(q)) > 0.5:
                w.append("boxes_x_pack_rate_does_not_match_quantity")
        except Exception:
            pass
    return w


def clean_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("product_name", "")).strip()
        if not looks_like_product(name):
            continue
        out: Dict[str, Any] = {
            "product_name": name,
            "boxes": as_number(raw.get("boxes")),
            "pack_rate": as_number(raw.get("pack_rate")),
            "quantity": as_number(raw.get("quantity")),
            "unit_price": as_number(raw.get("unit_price")),
            "total": as_number(raw.get("total")),
        }
        if isinstance(raw.get("specification"), dict) and raw["specification"]:
            out["specification"] = raw["specification"]
        for key in ("farm_code", "product_id", "row_index"):
            if raw.get(key) is not None and not empty_field(raw.get(key)):
                out[key] = raw[key]
        if raw.get("raw_values"):
            out["raw_values"] = raw["raw_values"]
        # Safe derivation
        if out["quantity"] is None and out["boxes"] and out["pack_rate"]:
            try:
                out["quantity"] = as_number(float(out["boxes"]) * float(out["pack_rate"]))
            except Exception:
                pass
        if out["total"] is None and out["quantity"] and out["unit_price"]:
            try:
                out["total"] = round(float(out["quantity"]) * float(out["unit_price"]), 4)
            except Exception:
                pass
        conf = 0.95
        if out["boxes"] is None: conf -= 0.02
        if out["pack_rate"] is None: conf -= 0.02
        if out["quantity"] is None: conf -= 0.04
        if out["unit_price"] is None: conf -= 0.04
        warns = validate_item(out)
        conf -= min(0.30, 0.08 * len(warns))
        out["confidence"] = round(max(0.0, min(1.0, conf)), 3)
        if warns:
            out["warnings"] = warns
        cleaned.append(out)
    return cleaned


# ===========================================================================
#  DOCLING DEEP EXTRACTION (optional)
# ===========================================================================

def extract_with_docling(content: bytes, fname: str) -> List[Dict[str, Any]]:
    """Use Docling for layout-aware extraction. Only call if DOCLING_ENABLED."""
    if not DOCLING_ENABLED or DoclingConverter is None:
        return []
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=Path(fname).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        converter = DoclingConverter()
        result = converter.convert(tmp_path)
        os.unlink(tmp_path)
        doc = result.document
        items = []
        for table in doc.tables:
            df = table.export_to_dataframe()
            df.columns = [str(c).lower().strip() for c in df.columns]
            qty_col = next((c for c in df.columns if 'qty' in c or 'quantity' in c or 'stems' in c), None)
            price_col = next((c for c in df.columns if 'price' in c or 'unit' in c or 'rate' in c), None)
            desc_col = next((c for c in df.columns if 'item' in c or 'desc' in c or 'variety' in c or 'flower' in c), None)
            total_col = next((c for c in df.columns if 'total' in c or 'amount' in c), None)
            if desc_col and qty_col:
                for _, row in df.iterrows():
                    try:
                        desc = str(row[desc_col]).strip()
                        if not looks_like_product(desc):
                            continue
                        item = {
                            "product_name": desc,
                            "quantity": as_number(parse_number(row[qty_col])),
                            "unit_price": as_number(parse_number(row[price_col])) if price_col else None,
                            "total": as_number(parse_number(row[total_col])) if total_col else None,
                            "boxes": None, "pack_rate": None,
                            "specification": {},
                        }
                        items.append(item)
                    except Exception:
                        continue
        return items
    except Exception as e:
        logger.warning(f"Docling extraction failed: {e}")
        return []


# ===========================================================================
#  PDF / IMAGE EXTRACTION
# ===========================================================================

def extract_text_from_pdf(content: bytes) -> Tuple[str, str]:
    """Fast path: PyMuPDF text first. OCR only if needed."""
    # PyMuPDF
    if fitz is not None:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            parts = []
            text_pages = 0
            for page in doc:
                txt = page.get_text("text", sort=True) or ""
                if len(re.sub(r"\s+", "", txt)) >= 20:
                    text_pages += 1
                if txt.strip():
                    parts.append(txt)
            doc.close()
            text = "\n".join(parts).strip()
            if len(re.sub(r"\s+", "", text)) >= 30:
                return text, "pdf_text_fitz"
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
    # PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        parts = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(parts).strip()
        if len(re.sub(r"\s+", "", text)) >= 30:
            return text, "pdf_text_pypdf2"
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")
    # Targeted OCR
    if fitz is None:
        return "", "pdf_text_empty"
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        ocr_parts = []
        for idx, page in enumerate(doc):
            if idx >= MAX_OCR_PAGES:
                break
            pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            txt = ocr_image(img)
            if txt:
                ocr_parts.append(txt)
        return "\n".join(ocr_parts), "pdf_ocr"
    except Exception as e:
        logger.warning(f"PDF OCR failed: {e}")
        return "", "pdf_text_empty"


def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    w, h = img.size
    longest = max(w, h)
    if longest < 1800:
        scale = 1800 / longest
        img = img.resize((int(w * scale), int(h * scale)))
    img = ImageEnhance.Contrast(img).enhance(1.6)
    return img


def ocr_image(img: Image.Image) -> str:
    processed = preprocess_image(img)
    for cfg in ("--oem 3 --psm 6", "--oem 3 --psm 4"):
        try:
            txt = pytesseract.image_to_string(processed, lang="eng", config=cfg, timeout=20)
            if txt and len(re.findall(r"[A-Za-z0-9]", txt)) >= 20:
                return txt
        except Exception:
            continue
    return ""


def extract_text_from_image(content: bytes) -> Tuple[str, str]:
    try:
        img = Image.open(io.BytesIO(content))
        return ocr_image(img), "image_ocr"
    except Exception:
        return "", "image_error"


def extract_text_from_docx(content: bytes) -> str:
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


def extract_text_from_json(content: bytes) -> Tuple[str, str]:
    try:
        obj = json.loads(content.decode("utf-8", errors="ignore"))
        return json.dumps(obj, ensure_ascii=False, indent=2), "json"
    except Exception:
        return content.decode("utf-8", errors="ignore"), "text"


# ===========================================================================
#  SPREADSHEET
# ===========================================================================

def extract_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    items = []
    # Find header row
    header_row = -1
    header_map = {0: "product"}
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        mapping = {}
        for idx, val in enumerate(row):
            if pd.isna(val):
                continue
            field = match_header(str(val))
            if field and field != "row_index" and field not in mapping.values():
                mapping[idx] = field
        fields = set(mapping.values())
        if (len(fields) >= 3 and
            any(x in fields for x in ("product", "variety", "description")) and
            any(x in fields for x in ("quantity", "boxes", "unit_price", "total", "pack_rate"))):
            header_row = i
            header_map = mapping
            break
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]
        vals = {}
        for idx, field in header_map.items():
            if idx < len(row) and not pd.isna(row.iloc[idx]):
                vals[field] = row.iloc[idx]
        if not vals:
            continue
        name = None
        for f in ("product", "variety", "description"):
            c = vals.get(f)
            if not empty_field(c) and looks_like_product(str(c)):
                name = str(c).strip()
                break
        if not name:
            continue
        name, ridx = strip_row_index(name)
        item = {
            "product_name": name,
            "boxes": as_number(vals.get("boxes")),
            "pack_rate": as_number(vals.get("pack_rate")),
            "quantity": as_number(vals.get("quantity")),
            "unit_price": as_number(vals.get("unit_price")),
            "total": as_number(vals.get("total")),
            "specification": {},
        }
        if ridx is not None:
            item["row_index"] = ridx
        if not empty_field(vals.get("farm_code")):
            item["farm_code"] = str(vals["farm_code"]).strip()
        items.append(item)
    return clean_items(items)


def extract_from_spreadsheet(content: bytes, ext: str) -> List[Dict[str, Any]]:
    try:
        if ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(content), header=None)
            except Exception:
                df = pd.read_csv(io.BytesIO(content), header=None, sep=";")
        else:
            df = pd.read_excel(io.BytesIO(content), header=None)
        return extract_from_dataframe(df)
    except Exception as e:
        logger.exception("Spreadsheet read failed")
        return []


# ===========================================================================
#  AI ANALYSIS LAYER
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
            if p is None or med == 0: continue
            dev = abs(float(p) - med) / med
            if dev >= 0.5:
                insights.append({"type": "price_outlier", "severity": "medium",
                                 "item_index": i,
                                 "product_name": it.get("product_name"),
                                 "message": f"Unit price {p} deviates {dev*100:.0f}% from median ({med:.2f})."})
    for i, it in enumerate(items):
        for w in it.get("warnings", []):
            anomalies.append({"item_index": i,
                              "product_name": it.get("product_name"),
                              "code": w,
                              "severity": "high" if "not_match" in w else "medium"})
    recs = []
    if any(i["type"] == "no_items_detected" for i in insights):
        recs.append("Re-upload a higher-resolution scan.")
    if any(i["type"] == "price_outlier" for i in insights):
        recs.append("Cross-check outlier prices against your rate card.")
    if not recs:
        recs.append("Extraction looks consistent. Proceed to product matching.")
    trust_base = (sum(i.get("confidence", 0.5) for i in items) / len(items)) if items else 0.0
    penalty = sum(0.10 if x.get("severity") == "high" else
                  0.04 if x.get("severity") == "medium" else 0
                  for x in insights)
    trust = round(max(0.0, min(1.0, trust_base - penalty)), 3)
    return {
        "trust_score": trust,
        "confidence_band": confidence_band(trust),
        "reasoning": f"Parsed {len(items)} item(s). Trust {trust:.2f} ({confidence_band(trust)}).",
        "insights": insights,
        "anomalies": anomalies,
        "recommendations": recs,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
    }


# ===========================================================================
#  PRODUCT MATCHING
# ===========================================================================

def normalize_product_for_match(name: str) -> str:
    n = norm(name)
    n = re.sub(r"\b\d+(?:\.\d+)?\s*cm\b", " ", n)
    n = re.sub(r"\b(?:box|boxes|bx|carton|cartons|qty|quantity|stems?)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


@app.post("/api/match-products")
async def match_products_endpoint(req: MatchRequest):
    try:
        if not req.items or not req.company_products:
            return {"success": True, "items": req.items, "matched_count": 0, "review_count": 0}
        out = []
        for item in req.items:
            iname = normalize_product_for_match(str(item.get("product_name", "")))
            ranked = []
            for product in req.company_products:
                names = [str(product.get("name", ""))]
                aliases = product.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(str(x) for x in aliases)
                for candidate in names:
                    cname = normalize_product_for_match(candidate)
                    if not cname:
                        continue
                    if iname == cname:
                        score = 100.0
                    else:
                        score = max(fuzz.ratio(iname, cname),
                                    fuzz.token_set_ratio(iname, cname),
                                    fuzz.WRatio(iname, cname))
                    ranked.append((score, product))
            ranked.sort(key=lambda x: x[0], reverse=True)
            best = ranked[0] if ranked else (0, None)
            second = ranked[1][0] if len(ranked) > 1 else 0
            confidence = best[0] / 100.0
            margin_ok = (best[0] - second) >= 6.0 or best[0] >= 98.0
            accepted = best[1] is not None and confidence >= MIN_MATCH_CONFIDENCE and margin_ok
            out.append({
                **item,
                "product_id": best[1].get("id") if accepted else None,
                "matched_product_name": best[1].get("name") if accepted else None,
                "match_confidence": round(confidence, 3),
                "match_status": "matched" if accepted else "review_required",
            })
        return {
            "success": True,
            "items": out,
            "matched_count": sum(1 for x in out if x.get("match_status") == "matched"),
            "review_count": sum(1 for x in out if x.get("match_status") == "review_required"),
        }
    except Exception as e:
        logger.exception("Product matching failed")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
#  ENDPOINTS
# ===========================================================================

@app.get("/api/ping")
async def ping():
    return {"ok": True, "version": ENGINE_VERSION,
            "docling_enabled": DOCLING_ENABLED,
            "t": datetime.utcnow().isoformat() + "Z"}


@app.get("/")
async def root():
    return {
        "service": "Smart Document Intelligence Engine",
        "version": ENGINE_VERSION,
        "docling_enabled": DOCLING_ENABLED,
        "status": "operational",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": ENGINE_VERSION,
        "ocr_available": bool(pytesseract),
        "pdf_available": fitz is not None,
        "docling_enabled": DOCLING_ENABLED,
    }


def _analyze_sync(content: bytes, fname: str, ext: str, company_id: int) -> Dict[str, Any]:
    started = time.perf_counter()
    items: List[Dict[str, Any]] = []
    text_extracted = ""
    method = ""
    meta: Dict[str, Any] = {}

    # Spreadsheet fast path
    if ext in ("xlsx", "xls", "xlsm", "csv"):
        items = extract_from_spreadsheet(content, ext)
        method = "spreadsheet"
    elif ext == "pdf":
        text_extracted, method = extract_text_from_pdf(content)
        items, meta = parse_order_text(text_extracted)
    elif ext in ("docx", "doc"):
        text_extracted = extract_text_from_docx(content)
        method = "docx"
        items, meta = parse_order_text(text_extracted)
    elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
        text_extracted, method = extract_text_from_image(content)
        items, meta = parse_order_text(text_extracted)
    elif ext == "json":
        text_extracted, method = extract_text_from_json(content)
        items, meta = parse_order_text(text_extracted)
    else:
        text_extracted = content.decode("utf-8", errors="ignore")
        method = "text"
        items, meta = parse_order_text(text_extracted)

    # Docling deep path (optional) — only if enabled AND fast path found nothing
    if DOCLING_ENABLED and not items and ext in ("pdf", "png", "jpg", "jpeg"):
        try:
            docling_items = extract_with_docling(content, fname)
            if docling_items:
                items = clean_items(docling_items)
                method = f"{method}_docling"
        except Exception as e:
            logger.warning(f"Docling path failed: {e}")

    cleaned = clean_items(items)
    if text_extracted and not meta:
        meta = extract_meta_from_text(text_extracted)

    # Apply strict schema with arithmetic validation
    routed = []
    for raw in cleaned:
        try:
            v = ExtractedLineItem(**raw)
            d = v.model_dump()
            d["confidence_band"] = confidence_band(d["confidence"])
            routed.append(d)
        except Exception as e:
            logger.warning(f"Item failed strict routing: {e}")

    total_boxes = safe_sum(x.get("boxes") for x in routed)
    total_qty = safe_sum(x.get("quantity") for x in routed)
    total_amount = safe_sum(x.get("total") for x in routed)

    warnings: List[str] = []
    for x in routed:
        warnings.extend(x.get("validation_warnings", []))

    elapsed = round(time.perf_counter() - started, 3)

    return {
        "success": True,
        "items": routed,
        "metadata": meta,
        "document_type": meta.get("document_type"),
        "total_boxes": int(total_boxes) if total_boxes else None,
        "total_quantity": int(total_qty) if total_qty else None,
        "total_amount": round(float(total_amount), 2),
        "currency": meta.get("currency", "USD"),
        "item_count": len(routed),
        "review_required": any(
            x.get("confidence", 0) < 0.80 or x.get("validation_warnings")
            for x in routed),
        "warnings": sorted(set(warnings)),
        "analysis": ai_analyze(routed, meta),
        "text_extracted": text_extracted[:12000] if text_extracted else "",
        "extraction_method": method,
        "file_type": ext,
        "filename": fname,
        "engine_version": ENGINE_VERSION,
        "diagnostics": {
            "elapsed_seconds": elapsed,
            "input_bytes": len(content),
            "docling_used": DOCLING_ENABLED and "docling" in method,
        },
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    company_id: int = Form(0),
    file_type: Optional[str] = Form(None),
):
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    fname = file.filename or "upload"
    ext = (file_type or Path(fname).suffix.lstrip(".")).lower()

    logger.info(f"Analyzing {fname} ({ext}), company={company_id}, size={len(content)}")

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, _analyze_sync, content, fname, ext, company_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze failed")
        # Never return 500 — return a well-formed empty response
        return {
            "success": False,
            "items": [],
            "metadata": {},
            "item_count": 0,
            "error": str(e),
            "engine_version": ENGINE_VERSION,
            "extraction_method": "crashed",
        }


@app.post("/api/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    fname = file.filename or ""
    ext = Path(fname).suffix.lstrip(".").lower()

    if ext == "pdf":
        text, method = extract_text_from_pdf(content)
    elif ext in ("docx", "doc"):
        text, method = extract_text_from_docx(content), "docx"
    elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
        text, method = extract_text_from_image(content)
    elif ext == "json":
        text, method = extract_text_from_json(content)
    else:
        text, method = content.decode("utf-8", errors="ignore"), "text"

    return {
        "success": True,
        "text": text,
        "length": len(text),
        "file_type": ext,
        "extraction_method": method,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
