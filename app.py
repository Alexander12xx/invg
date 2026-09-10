"""
Smart Document Intelligence Engine v8.0
Structured-table-first extraction with anchored column geometry.

Fixes over v7:
  • Table header anchoring prevents header words leaking into product rows
  • Multi-line row stitching (item description wrapping across lines)
  • Row-index stripping (leading "1 ", "2 ", ... never part of product name)
  • Strict boilerplate/contact/link/address rejection
  • Currency codes stripped inside numeric cells
  • Free-form fallback is gated: only runs when no table was detected

ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io, os, re, json, math, logging, statistics
from datetime import datetime

import pandas as pd
import PyPDF2
import docx
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
from rapidfuzz import fuzz

try:
    import fitz
except Exception:
    fitz = None

try:
    import openpyxl  # noqa
except Exception:
    openpyxl = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-document-engine")

TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

ENGINE_VERSION = "8.0.0"

app = FastAPI(title="Smart Document Intelligence Engine",
              version=ENGINE_VERSION,
              description="Structured-table-first, confidence-aware extraction engine.")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MIN_MATCH_CONFIDENCE = float(os.getenv("MIN_MATCH_CONFIDENCE", "0.60"))


# ===========================================================================
#  STRICT SCHEMA (unchanged contract, safer defaults)
# ===========================================================================

PRICE_PATTERN = re.compile(
    r"(?:\$|€|£|KES|KSH|USD|EUR|GBP|AED|SAR|QAR)?\s*"
    r"\b\d+(?:[\.,]\d{1,4})?\b\s*"
    r"(?:USD|KES|EUR|GBP|AED|SAR|QAR)?", re.IGNORECASE)
CURRENCY_SYMBOLS = re.compile(
    r"[\$€£]|(?:\b(?:USD|KES|KSH|EUR|GBP|AED|SAR|QAR)\b)", re.IGNORECASE)
PRICE_LABEL_WORDS = re.compile(
    r"\b(?:price|rate|cost|total|amount|subtotal|value|"
    r"unit\s*price|price\s*/?\s*stem|per\s*stem)\b", re.IGNORECASE)


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
        if CURRENCY_SYMBOLS.search(text) or PRICE_LABEL_WORDS.search(text):
            cleaned = PRICE_PATTERN.sub(" ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:,;.|")
            return cleaned if len(cleaned) >= 2 else "UNKNOWN_ITEM"
        return text

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
            exp = self.quantity * self.unit_price
            if abs(exp - self.total) > max(0.02, abs(self.total) * 0.02):
                w.append("quantity_x_unit_price_does_not_match_total")
        if self.boxes and self.pack_rate and self.quantity:
            exp = self.boxes * self.pack_rate
            if abs(exp - self.quantity) > 0.5:
                w.append("boxes_x_pack_rate_does_not_match_quantity")
        self.validation_warnings = sorted(set(w))
        return self


class MatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    company_products: List[Dict[str, Any]]


# ===========================================================================
#  NUMERIC / TEXT HELPERS
# ===========================================================================

def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s.strip().lower())


def clean_ocr_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\|", " |", text)
    text = re.sub(r"\|\s+", "| ", text)
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
    txt = re.sub(r"[\$€£]", "", txt)
    txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt).strip()
    if not txt:
        return None
    # Handle 1,234.56 and 1.234,56
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


# ===========================================================================
#  BOILERPLATE / JUNK REJECTION  (BIG upgrade)
# ===========================================================================

BOILERPLATE_PATTERNS = [
    r"^\s*thank\s+you\b",
    r"^\s*thanks\b",
    r"^\s*welcome\b",
    r"^\s*please\s+note\b",
    r"^\s*note\s*[:!]",
    r"computer[- ]generated",
    r"no\s+signature\s+required",
    r"^\s*generated\s+on\b",
    r"^\s*scan\s+to\s+verify\b",
    r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$",
    r"^\s*invoice\s*(?:no|number|#)?\b[:\s]*[a-z0-9\-/]+",
    r"^\s*(?:quotation|proforma|receipt|delivery\s+note)\b",
    r"^\s*paid\s*$", r"^\s*sent\s*$", r"^\s*unpaid\s*$", r"^\s*draft\s*$",
    r"^\s*(?:order|invoice)\s+details\s*$",
    r"^\s*created\s+by\b",
    r"^\s*bill\s+to\b", r"^\s*ship\s+to\b", r"^\s*consignee\b",
    r"^\s*customer\s+details?\s*$",
    r"^\s*terms?\s*(?:&|and)\s*conditions\b",
    r"prices?\s+are\s+inclusive\s+of\s+vat",
    r"delivery\s+cost\s+is\s+the\s+buyer",
    r"total\s+price\s+per\s+item\s+is\s+inclusive",
    r"other\s+charges\s*:?",
    r"^\s*sub[- ]?total\b",
    r"^\s*grand\s+total\b",
    r"^\s*balance\s+due\b",
    r"^\s*due\s+date\b",
    r"^\s*payment\s+terms\b",
]

BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

PHONE_RE = re.compile(
    r"(?:\+?\d[\d\s\-().]{6,}\d)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9\-]+\.(?:com|net|org|io|ke|co\.ke|tc|info|biz)\b", re.I)
HASH_ID_RE = re.compile(r"^#?[A-Z]{2,}\d{4,}[A-Z0-9\-]*$", re.I)
ADDRESS_RE = re.compile(
    r"\b(street|st\.|road|rd\.|avenue|ave\.|boulevard|blvd\.|building|bldg|"
    r"floor|suite|tower|plaza|po\s*box|p\.o\.|postal\s*code|zip|"
    r"nairobi|dubai|amsterdam|bogota|quito|miami|johannesburg|"
    r"kenya|uae|united\s+arab\s+emirates|netherlands|colombia|ecuador)\b",
    re.IGNORECASE)
NAME_HEADER_RE = re.compile(
    r"^\s*(?:#\s*)?(?:product|service|description|variety|item|qty|"
    r"quantity|unit\s*price|price|total|amount|boxes?|cartons?|"
    r"pack\s*rate|rate)\b.*$", re.IGNORECASE)

KNOWN_COMPANY_TERMS = re.compile(
    r"\b(limited|ltd\.?|llc|inc\.?|plc|gmbh|bv|nv|s\.?a\.?|s\.?l\.?|"
    r"company|enterprises?|investment|trading|holdings?|imports?|"
    r"exports?|flowers?|roses?|logistics|freight|agency)\b", re.IGNORECASE)


def is_boilerplate(line: str) -> bool:
    if not line:
        return True
    s = line.strip()
    if BOILERPLATE_RE.search(s):
        return True
    if URL_RE.search(s):
        return True
    if EMAIL_RE.search(s):
        return True
    if HASH_ID_RE.match(s):
        return True
    # a line dominated by digits/symbols
    letters = re.findall(r"[A-Za-z]", s)
    if len(letters) < 3 and len(s) < 30:
        return True
    return False


def is_contact_or_address(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if PHONE_RE.search(s) and len(re.findall(r"\d", s)) >= 7:
        return True
    if EMAIL_RE.search(s) or URL_RE.search(s):
        return True
    if ADDRESS_RE.search(s) and len(s.split()) <= 8:
        return True
    return False


def is_table_header_line(line: str) -> bool:
    """Detects rows like: '# PRODUCT / SERVICE DESCRIPTION QTY UNIT PRICE TOTAL'"""
    s = re.sub(r"[|]", " ", line).strip()
    if not s:
        return False
    toks = re.split(r"\s{2,}| \| ", s)
    if len(toks) < 2:
        toks = s.split()
    joined = norm(s)
    hits = 0
    for key in ("product", "service", "description", "qty", "quantity",
                "unit price", "price", "total", "amount", "boxes",
                "pack rate", "rate"):
        if key in joined:
            hits += 1
    return hits >= 2


def strip_leading_row_index(name: str) -> Tuple[str, Optional[int]]:
    """'12 Welding goggles/Shields' -> ('Welding goggles/Shields', 12)"""
    m = re.match(r"^\s*(\d{1,3})\s*[.)]?\s+(.*\S)\s*$", name)
    if m and len(m.group(2)) >= 2:
        return m.group(2), int(m.group(1))
    return name, None


def looks_like_product(name: str) -> bool:
    n = str(name or "").strip()
    if len(n) < 2 or not re.search(r"[A-Za-z]{3,}", n):
        return False
    if is_boilerplate(n) or is_contact_or_address(n):
        return False
    if NAME_HEADER_RE.match(n) and len(n.split()) <= 6:
        return False
    # all-caps street-like
    if ADDRESS_RE.search(n):
        return False
    return True


# ===========================================================================
#  HEADER KNOWLEDGE (unchanged core, plus '#' index)
# ===========================================================================

COLUMN_SYNONYMS = {
    "product": ["product", "product name", "product/service", "product / service",
                "product or service", "item", "item name", "service", "article",
                "articles", "commodity", "goods", "stock item", "particulars",
                "product description", "item description", "description of goods"],
    "variety": ["variety", "flower", "flower name", "flower type", "species",
                "cultivar", "kind", "variety name"],
    "description": ["description", "desc", "details", "item details",
                    "specification", "specifications", "remarks",
                    "product details", "product / service description"],
    "farm_code": ["farm code", "farmcode", "farm reference", "farm ref",
                  "supplier code", "grower code", "grower reference"],
    "boxes": ["boxes", "box", "bx", "cartons", "carton", "ctn", "cases",
              "case", "bundles", "bundle", "packages", "pkg",
              "number of boxes", "no. of boxes", "no boxes", "box qty",
              "carton qty", "cartons qty"],
    "pack_rate": ["packrate", "pack rate", "pack_rate", "per box", "per carton",
                  "stems per box", "stems/box", "qty per box",
                  "quantity per box", "stems per carton", "qty/carton",
                  "quantity/carton", "conversion rate", "pack rate per box"],
    "quantity": ["quantity", "qty", "qnty", "stems", "pcs", "pieces", "count",
                 "total quantity", "total qty", "number of stems",
                 "stem quantity", "invoice quantity", "qty invoice",
                 "quantity invoice"],
    "unit_price": ["price", "price per stem", "price/stem", "unit price",
                   "unit price (usd)", "unit price(usd)", "cost",
                   "price per unit", "per stem", "per piece",
                   "amount per stem", "unit cost", "priceperstem",
                   "rate per stem", "selling price", "unit selling price"],
    "total": ["total", "total price", "total amount", "line total",
              "line amount", "sub-total", "subtotal", "extended price",
              "total (usd)", "total(usd)", "line value"],
    "length": ["length", "length(cm)", "length (cm)", "size", "size(cm)",
               "stem length", "height", "stem size", "length cm"],
    "discount": ["discount", "disc.", "rebate"],
    "tax": ["tax", "vat", "gst", "sales tax"],
}

META_LABELS = {
    "invoice_number": ["invoice number", "invoice no", "invoice #",
                       "invoice no.", "inv no", "inv #", "quotation number",
                       "quotation no", "quote number", "proforma number",
                       "proforma no", "document number", "doc no",
                       "reference", "ref no", "reference number",
                       "document ref"],
    "date": ["date", "date of shipment", "shipment date", "invoice date",
             "issue date", "document date", "quotation date"],
    "due_date": ["due date", "payment due", "valid until", "valid till",
                 "expiry", "expires", "expiration"],
    "currency": ["currency", "currency code", "ccy"],
    "vat_rate": ["vat rate", "vat rate (%)", "tax rate", "tax %", "vat %"],
    "country_destination": ["country of destination", "destination country",
                            "destination", "country dest", "ship to country",
                            "country destination"],
    "point_of_entry": ["point of entry", "port of entry", "entry point",
                       "port", "airport", "arrival port"],
    "country_origin": ["country of origin", "origin country"],
    "consignee_name": ["consignee name", "consignee", "bill to", "ship to",
                       "customer name", "client name", "buyer name",
                       "consignee details"],
    "consignee_address": ["consignee address", "bill to address",
                          "ship to address", "customer address",
                          "buyer address", "delivery address"],
    "seller_name": ["seller name", "seller", "vendor", "supplier",
                    "exporter", "seller / exporter", "exporter name"],
    "purchase_order_no": ["purchase order no", "purchase order #",
                          "purchase order number", "purchase order",
                          "po no", "po #", "po number", "customer po",
                          "order number", "purchase order no."],
    "payment_terms": ["payment terms", "payment term", "terms of payment",
                      "payment"],
    "transportation": ["transportation", "transport", "shipment method",
                       "mode of transport", "shipping method",
                       "transportation details"],
    "awb_number": ["awb number", "awb no", "awb", "air waybill",
                   "tracking number", "waybill", "airway bill"],
    "net_weight": ["net weight", "net weight (kgs)", "net weight (kg)",
                   "net kg", "net weight kgs"],
    "gross_weight": ["gross weight", "gross kg", "gross weight (kg)"],
    "notes": ["notes", "note", "comments", "comment", "remarks"],
}

DOC_TYPES = {
    "invoice": ["invoice", "tax invoice", "commercial invoice"],
    "quotation": ["quotation", "quote", "estimate"],
    "proforma": ["proforma", "pro forma", "pro-forma", "proforma invoice"],
    "receipt": ["receipt", "payment receipt"],
    "delivery_note": ["delivery note", "delivery", "dispatch note"],
    "packing_list": ["packing list", "packing slip"],
    "credit_note": ["credit note", "credit memo"],
    "purchase_order": ["purchase order", "purchase order form"],
    "statement": ["statement", "account statement"],
}


def match_header(label: str) -> Optional[str]:
    l = norm(label)
    l = re.sub(r"^[#*]+", "", l)
    l = re.sub(r"[*:.#]+$", "", l).strip()
    if not l:
        return None
    if l in {"#", "no", "no.", "s/n", "sn"}:
        return "row_index"
    for field, names in COLUMN_SYNONYMS.items():
        if l in {norm(x) for x in names}:
            return field
    priority = ["pack_rate", "unit_price", "farm_code", "quantity",
                "boxes", "length", "total", "product", "variety", "description"]
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
    best_meta, best_score = None, 0
    for meta, names in META_LABELS.items():
        for n in names:
            s = fuzz.token_set_ratio(key, norm(n))
            if s > best_score:
                best_score, best_meta = s, meta
    return best_meta if best_score >= 91 else None


# ===========================================================================
#  DOCUMENT TYPE / META
# ===========================================================================

def detect_document_type(text: str) -> Optional[str]:
    low = norm(text[:12000])
    scores = {t: max([fuzz.partial_ratio(low, norm(w)) for w in ws] or [0])
              for t, ws in DOC_TYPES.items()}
    typ, score = max(scores.items(), key=lambda x: x[1])
    return typ if score >= 70 else None


def extract_meta_from_text(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    lines = [x.strip() for x in clean_ocr_text(text).splitlines() if x.strip()]
    for i, line in enumerate(lines):
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,70}?)\s*"
                     r"(?:[:#]\s*|-\s+|\|\s*)(.*?)\s*$", line)
        if m:
            label, value = m.group(1), m.group(2).strip()
            field = match_meta_label(label)
            if field and value and not empty_field(value):
                meta[field] = value
                continue
        field = match_meta_label(line.rstrip(":#"))
        if field and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and not match_meta_label(nxt) and not re.match(r"^[A-Za-z].*[:#]", nxt):
                if field not in meta:
                    meta[field] = nxt
    if "currency" in meta:
        c = norm(meta["currency"])
        for word, code in {"usd": "USD", "kes": "KES", "ksh": "KES",
                           "eur": "EUR", "euro": "EUR", "gbp": "GBP",
                           "pound": "GBP", "aed": "AED", "sar": "SAR",
                           "qar": "QAR", "dollar": "USD"}.items():
            if word in c:
                meta["currency"] = code
                break
    else:
        for word, code in {"kes": "KES", "ksh": "KES", "usd": "USD",
                           "eur": "EUR", "gbp": "GBP", "aed": "AED"}.items():
            if re.search(rf"\b{word}\b", norm(text)):
                meta["currency"] = code
                break
    meta["document_type"] = detect_document_type(text)
    return meta


# ===========================================================================
#  TABLE DETECTION (anchor row, not just header)
# ===========================================================================

def find_table_anchor(lines: List[str]) -> Optional[int]:
    """Find index of the most header-like line."""
    best_idx, best_score = None, 0
    for i, line in enumerate(lines):
        if not is_table_header_line(line):
            continue
        mapped = [match_header(c) for c in split_columns(line)]
        known = [m for m in mapped if m and m != "row_index"]
        if len(known) < 2:
            continue
        score = len(known)
        # Must include a product-ish or numeric field to count
        if any(k in known for k in ("product", "description", "variety", "quantity",
                                     "unit_price", "total", "boxes", "pack_rate")):
            if score > best_score:
                best_score, best_idx = score, i
    return best_idx


def split_columns(line: str) -> List[str]:
    if "|" in line:
        parts = [x.strip() for x in re.split(r"\s*\|\s*", line)]
    elif "\t" in line:
        parts = [x.strip() for x in line.split("\t")]
    else:
        parts = [x.strip() for x in re.split(r"\s{2,}", line.strip())]
    return [x for x in parts if x != ""]


def header_columns_for(anchor_line: str) -> List[str]:
    """
    Given '# PRODUCT / SERVICE DESCRIPTION QTY UNIT PRICE TOTAL'
    produce a canonical column list in order.
    """
    s = anchor_line.strip()
    s = re.sub(r"^[#\s]+", "", s)
    # strip trailing boilerplate if present
    # split on 2+ spaces or pipe or tabs (columns tend to be wide-spaced)
    cols = re.split(r"\s{2,}|\s*\|\s*|\t+", s)
    if len(cols) < 2:
        cols = re.split(r"\s+", s)
    # merge known two-word labels
    merged: List[str] = []
    i = 0
    while i < len(cols):
        cur = cols[i].strip()
        if not cur:
            i += 1
            continue
        nxt = cols[i + 1].strip() if i + 1 < len(cols) else ""
        combined = norm(cur + " " + nxt) if nxt else ""
        if combined in {"unit price", "pack rate", "line total",
                         "product service", "service description",
                         "product description"}:
            merged.append(cur + " " + nxt)
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged


def header_map_for(anchor_line: str) -> List[Optional[str]]:
    cols = header_columns_for(anchor_line)
    mapped = []
    for c in cols:
        m = match_header(c)
        mapped.append(m)
    # If any column ends up None but a token like 'UNIT' or 'PRICE' is adjacent, try joined again.
    # Deduplicate; keep positional.
    return mapped


# ===========================================================================
#  TOKEN / CELL EXTRACTION FOR TABLE ROWS
# ===========================================================================

NUMBER_RE = re.compile(r"^\s*(?:[$€£]|KES|KSH|USD|EUR|GBP|AED|SAR|QAR)?\s*"
                       r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\s*"
                       r"(?:[$€£]|KES|KSH|USD|EUR|GBP|AED|SAR|QAR)?\s*$",
                       re.IGNORECASE)


def is_numeric_cell(cell: str) -> bool:
    if not cell:
        return False
    return bool(NUMBER_RE.match(cell.strip()))


def split_row_into_cells(row: str) -> List[str]:
    """
    Try to tokenize a data row into cells based on wide gaps or pipes.
    """
    if "|" in row:
        parts = [x.strip() for x in re.split(r"\s*\|\s*", row)]
        return [p for p in parts if p != ""]
    if "\t" in row:
        parts = [x.strip() for x in row.split("\t")]
        return [p for p in parts if p != ""]
    parts = [x.strip() for x in re.split(r"\s{2,}", row.strip())]
    if len(parts) >= 3:
        return [p for p in parts if p != ""]
    return [row.strip()]


def assign_cells_to_columns(cells: List[str], headers: List[Optional[str]],
                            numeric_fields: set) -> Dict[str, Any]:
    """
    Positional assignment with right-alignment preference for numeric fields.
    """
    result: Dict[str, Any] = {}
    raw: Dict[str, str] = {}

    # Identify which header slots are numeric in this table
    numeric_slots = [i for i, h in enumerate(headers)
                     if h in numeric_fields]
    product_slots = [i for i, h in enumerate(headers)
                     if h in ("product", "description", "variety")]

    # Split incoming cells into text-ish and numeric-ish buckets
    nums, texts = [], []
    for c in cells:
        if is_numeric_cell(c):
            nums.append(c)
        else:
            texts.append(c)

    # Assign numerics to numeric slots from the right (usually qty, price, total ...)
    assigned = {}
    for slot, val in zip(reversed(numeric_slots), reversed(nums)):
        assigned[slot] = val

    # Assign remaining text to product-ish slots first
    remaining_texts = [c for c in texts]
    for slot in product_slots:
        if slot in assigned:
            continue
        if remaining_texts:
            assigned[slot] = remaining_texts.pop(0)

    # Anything left: fill empty header slots in order
    for i, h in enumerate(headers):
        if h is None or i in assigned:
            continue
        if remaining_texts:
            assigned[i] = remaining_texts.pop(0)

    for i, val in assigned.items():
        field = headers[i]
        if not field or field == "row_index":
            continue
        if field in numeric_fields:
            n = parse_number(val)
            if n is not None:
                result[field] = as_number(n)
            else:
                raw[f"unparsed_{field}"] = val
        elif field == "length":
            m = re.search(r"\d+(?:\.\d+)?", str(val))
            if m:
                result.setdefault("specification", {})["length"] = f"{m.group(0)}cm"
        else:
            result[field] = str(val).strip()

    if raw:
        result.setdefault("raw_values", {}).update(raw)
    return result


# ===========================================================================
#  TABLE EXTRACTION — the main improvement
# ===========================================================================

NUMERIC_FIELDS = {"boxes", "pack_rate", "quantity", "unit_price", "total"}


def extract_table_items(lines: List[str], anchor_idx: int) -> List[Dict[str, Any]]:
    anchor_line = lines[anchor_idx]
    headers = header_map_for(anchor_line)
    if not any(headers):
        return []

    items: List[Dict[str, Any]] = []
    pending_cells: List[str] = []

    def flush(pending):
        if not pending:
            return
        row_dict = assign_cells_to_columns(pending, headers, NUMERIC_FIELDS)
        name = None
        for key in ("product", "description", "variety"):
            if row_dict.get(key) and looks_like_product(str(row_dict[key])):
                name = str(row_dict[key]).strip()
                break
        if not name:
            for c in pending:
                if looks_like_product(c):
                    name = c.strip()
                    break
        if not name:
            return
        name, ridx = strip_leading_row_index(name)
        if not looks_like_product(name):
            return
        item = {
            "product_name": name,
            "boxes": row_dict.get("boxes"),
            "pack_rate": row_dict.get("pack_rate"),
            "quantity": row_dict.get("quantity"),
            "unit_price": row_dict.get("unit_price"),
            "total": row_dict.get("total"),
            "specification": row_dict.get("specification", {}) or {},
        }
        if row_dict.get("farm_code"):
            item["farm_code"] = str(row_dict["farm_code"]).strip()
        if row_dict.get("raw_values"):
            item["raw_values"] = row_dict["raw_values"]
        if ridx is not None:
            item["row_index"] = ridx
        items.append(item)

    for line in lines[anchor_idx + 1:]:
        s = line.strip()
        if not s:
            continue

        # A new header → restart column map
        if is_table_header_line(s) and any(match_header(c) for c in split_columns(s)):
            flush(pending_cells); pending_cells = []
            headers = header_map_for(s)
            continue

        # Pure boilerplate outside table
        if is_boilerplate(s):
            continue

        cells = split_row_into_cells(s)

        # Single-cell wide-gap fallback: try to split numeric tokens off the right
        if len(cells) == 1:
            # Extract trailing numerics greedily
            m = re.findall(
                r"(?:[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:KES|USD|EUR|GBP|AED)?)",
                s, re.IGNORECASE)
            text_part = s
            if m:
                text_part = s
                for tok in reversed(m):
                    # cut from end
                    idx = text_part.rfind(tok)
                    if idx >= 0:
                        text_part = text_part[:idx].rstrip()
                text_part = text_part.strip(" -:,;")
                cells = [text_part] + [tok.strip() for tok in m]

        # If line has at least one numeric-looking cell, or a leading index, treat as new row
        has_number = any(is_numeric_cell(c) for c in cells)
        has_lead_index = bool(re.match(r"^\d{1,3}[.)]?\s+", s))

        if pending_cells and not (has_number or has_lead_index):
            # Continuation of previous row's description
            pending_cells[0] = (pending_cells[0] + " " + cells[0]).strip()
            continue

        if pending_cells:
            flush(pending_cells)
        pending_cells = cells

    flush(pending_cells)
    return items


# ===========================================================================
#  FREE-FORM FALLBACK (only used when no table found)
# ===========================================================================

FIELD_PATTERNS = {
    "boxes": r"(?:no\.?\s*of\s*)?(?:boxes?|bx|cartons?|ctn|cases?|bundles?|packages?)",
    "pack_rate": r"(?:pack\s*rate|packrate|stems?\s*(?:per|/)\s*(?:box|carton)|"
                 r"qty\s*(?:per|/)\s*(?:box|carton)|quantity\s*(?:per|/)\s*(?:box|carton))",
    "quantity": r"(?:quantity|qty|qnty|total\s+qty|total\s+quantity|invoice\s+qty|"
                r"invoice\s+quantity|stems?|pieces?|pcs|units?)",
    "unit_price": r"(?:price\s*(?:per|/)\s*(?:stem|piece|unit)|price\s*per\s*stem|"
                  r"price/stem|unit\s*price|unit\s*cost|rate\s*per\s*stem|"
                  r"selling\s*price|cost\s*per\s*unit)",
    "total": r"(?:line\s+total|total\s+amount|total\s+price|line\s+amount|"
             r"extended\s+price|amount)",
    "length": r"(?:length(?:\s*\(?(?:cm|cms)\)?)?|stem\s+length|size)\b",
    "farm_code": r"(?:farm\s*code|farm\s*ref(?:erence)?|grower\s*code|supplier\s*code)",
}


def extract_labeled_fields(line: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field in ("pack_rate", "unit_price", "farm_code", "quantity",
                  "boxes", "length", "total"):
        pat = FIELD_PATTERNS[field]
        m = re.search(rf"\b{pat}\b\s*(?:[:=]\s*|\s+)([^|,;]+)", line, re.I)
        if not m:
            continue
        raw = m.group(1).strip()
        nxt = re.search(r"\s+(?:pack\s*rate|packrate|qty|quantity|boxes?|"
                        r"cartons?|price|unit\s*price|total|farm\s*code|length)\b",
                        raw, re.I)
        if nxt:
            raw = raw[:nxt.start()].strip()
        if field in {"boxes", "pack_rate", "quantity", "unit_price", "total"}:
            v = parse_number(raw)
            if v is not None:
                result[field] = as_number(v)
        elif field == "length":
            lm = re.search(r"\d+(?:\.\d+)?", raw)
            if lm:
                result["specification"] = {"length": f"{lm.group(0)}cm"}
        elif field == "farm_code":
            parts = raw.split()
            if parts:
                result["farm_code"] = parts[0]
    return result


def strip_known_annotations(name: str) -> str:
    s = name
    for pat in [
        r"\bpack\s*rate\b\s*[:=]?\s*[\d,.]+",
        r"\bpackrate\b\s*[:=]?\s*[\d,.]+",
        r"\b(?:qty|quantity|qnty)\b\s*[:=]?\s*[\d,.]+",
        r"\b(?:boxes?|bx|cartons?|ctn)\b\s*[:=]?\s*[\d,.]+",
        r"\b(?:price|rate|unit\s*price|price\s*/\s*stem|price\s*per\s*stem)"
        r"\b\s*[:=@]?\s*[\d,.]+",
        r"\b(?:total|amount)\b\s*[:=]?\s*[\d,.]+",
        r"\b\d+(?:\.\d+)?\s*cm\b",
    ]:
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(r"(?i)(?<=\d)\s*(?:usd|us\$|kes|ksh|eur|gbp|aed|sar|qar)\b", " ", s)
    s = re.sub(r"[\$€£]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" -:,;.|")


def parse_inline_line(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s or is_boilerplate(s) or is_contact_or_address(s):
        return None
    if is_table_header_line(s):
        return None
    fields = extract_labeled_fields(s)
    if "boxes" not in fields:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:boxes?|bx|cartons?|ctn)\b", s, re.I)
        if m:
            fields["boxes"] = as_number(m.group(1))
    if "specification" not in fields:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*cm\b", s, re.I)
        if m:
            fields["specification"] = {"length": f"{m.group(1)}cm"}
    name = strip_known_annotations(s)
    name = re.sub(r"^(\d{1,3})[.)]?\s+", "", name)  # strip leading index
    if not looks_like_product(name):
        return None
    item = {
        "product_name": name.strip(),
        "boxes": fields.get("boxes"),
        "pack_rate": fields.get("pack_rate"),
        "quantity": fields.get("quantity"),
        "unit_price": fields.get("unit_price"),
        "total": fields.get("total"),
        "specification": fields.get("specification", {}) or {},
    }
    if fields.get("farm_code"):
        item["farm_code"] = fields["farm_code"]
    return item


# ===========================================================================
#  TOP-LEVEL TEXT PARSER (table-first)
# ===========================================================================

def parse_order_text(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = clean_ocr_text(text)
    lines = [x.rstrip() for x in text.splitlines()]
    meta = extract_meta_from_text(text)

    anchor = find_table_anchor(lines)
    if anchor is not None:
        table_items = extract_table_items(lines, anchor)
        if table_items:
            return clean_items(table_items), meta

    # Fallback (only reached when no table header was detected)
    items = []
    for line in lines:
        s = line.strip()
        if not s or len(s) < 3:
            continue
        if is_boilerplate(s) or is_contact_or_address(s):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,70}\s*[:#]", s):
            if match_meta_label(re.split(r"[:#]", s, 1)[0]):
                continue
        item = parse_inline_line(s)
        if item:
            items.append(item)

    return clean_items(items), meta


# ===========================================================================
#  EXCEL / CSV  (table-aware by construction)
# ===========================================================================

def find_excel_header(df: pd.DataFrame) -> Tuple[Optional[int], Dict[int, str]]:
    best = None
    for i in range(min(50, len(df))):
        row = df.iloc[i]
        mapping: Dict[int, str] = {}
        for idx, val in enumerate(row):
            if pd.isna(val):
                continue
            field = match_header(str(val))
            if field and field != "row_index" and field not in mapping.values():
                mapping[idx] = field
        fields = set(mapping.values())
        score = len(fields)
        if (score >= 2
                and any(x in fields for x in ("product", "variety", "description"))
                and any(x in fields for x in ("quantity", "boxes", "unit_price",
                                               "total", "pack_rate"))):
            if best is None or score > best[0]:
                best = (score, i, mapping)
    return (best[1], best[2]) if best else (None, {})


def extract_from_dataframe(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if df is None or df.empty:
        return [], {}
    header_row, header_map = find_excel_header(df)
    if header_row is None:
        return [], {}
    items = []
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]
        vals: Dict[str, Any] = {}
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
        name, ridx = strip_leading_row_index(name)
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
        if not empty_field(vals.get("length")):
            lv = str(vals["length"]).strip()
            m = re.search(r"\d+(?:\.\d+)?", lv)
            if m:
                item["specification"]["length"] = f"{m.group(0)}cm"
        items.append(item)
    return clean_items(items), {}


def extract_from_excel(content: bytes, ext: str):
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
        return [], {"error": str(e)}


# ===========================================================================
#  BINARY TEXT EXTRACTION
# ===========================================================================

def extract_text_from_pdf(content: bytes) -> Tuple[str, str]:
    text_parts = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                pass
    except Exception as e:
        logger.warning("PyPDF2 failed: %s", e)
    text = "\n".join(text_parts).strip()
    if len(re.sub(r"\s+", "", text)) >= 30:
        return text, "pdf_text"
    if fitz is None:
        return text, "pdf_text_empty_or_short"
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        parts = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(ocr_image(img))
        return "\n".join(parts), "pdf_ocr"
    except Exception:
        logger.exception("PDF OCR failed")
        return text, "pdf_text"


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
        logger.exception("DOCX failed")
        return ""


def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) < 1800:
        scale = min(2.5, 1800 / max(w, h))
        img = img.resize((int(w * scale), int(h * scale)))
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = ImageEnhance.Sharpness(gray).enhance(1.4)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray


def ocr_image(img: Image.Image) -> str:
    processed = preprocess_image(img)
    configs = ["--oem 3 --psm 6", "--oem 3 --psm 4",
               "--oem 3 --psm 11", "--oem 3 --psm 3"]
    results = []
    for cfg in configs:
        try:
            t = pytesseract.image_to_string(processed, lang="eng", config=cfg)
            if t:
                results.append(t)
        except Exception as e:
            logger.warning("OCR config failed: %s", e)
    if not results:
        return ""
    return max(results, key=lambda x: len(re.findall(r"[A-Za-z0-9]", x)))


def extract_text_from_image(content: bytes) -> Tuple[str, str]:
    try:
        img = Image.open(io.BytesIO(content))
        return ocr_image(img), "image_ocr"
    except Exception:
        logger.exception("Image extraction failed")
        return "", "image_error"


def extract_text_from_json(content: bytes) -> Tuple[str, str]:
    try:
        obj = json.loads(content.decode("utf-8", errors="ignore"))
        return json.dumps(obj, ensure_ascii=False, indent=2), "json"
    except Exception:
        return content.decode("utf-8", errors="ignore"), "text"


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
            exp = float(q) * float(u)
            if abs(exp - float(t)) > max(0.02, abs(float(t)) * 0.02):
                w.append("quantity_x_unit_price_does_not_match_total")
        except Exception:
            pass
    if b and p and q:
        try:
            exp = float(b) * float(p)
            if abs(exp - float(q)) > 0.5:
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

        # Non-destructive arithmetic fill
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
        if out.get("raw_values"): conf -= 0.10

        warns = validate_item(out)
        conf -= min(0.30, 0.08 * len(warns))
        out["confidence"] = round(max(0.0, min(1.0, conf)), 3)
        if warns:
            out["warnings"] = warns
        cleaned.append(out)
    return cleaned


# ===========================================================================
#  AI ANALYSIS LAYER
# ===========================================================================

FLOWER_FAMILIES = {
    "rose": ["rose", "roses", "rosa"],
    "carnation": ["carnation", "carnations", "dianthus"],
    "chrysanthemum": ["chrysanthemum", "chrysanthemums", "mum", "spray mum",
                      "spray mums", "chysanthemum", "chysanthemums"],
    "tulip": ["tulip", "tulips"],
    "lily": ["lily", "lilies", "lilium", "oriantal", "oriental", "original"],
    "orchid": ["orchid", "orchids", "phalaenopsis"],
    "gerbera": ["gerbera", "gerberas"],
    "alstroemeria": ["alstroemeria", "alstromeria", "astomeria", "astromeria"],
    "gypsophila": ["gypsophila", "gypso", "gyp", "excelence", "excellence"],
    "hypericum": ["hypericum", "hypericums"],
    "solidago": ["solidago"],
    "limonium": ["limonium", "limoniya", "statice", "static"],
    "eucalyptus": ["eucalyptus"],
    "ruscus": ["ruscus"],
    "leather_leaf": ["leather leaf", "leatherleaf"],
    "hydrangea": ["hydrangea", "hydringa"],
}


def infer_flower_family(name: str) -> Optional[str]:
    low = norm(name)
    for family, terms in FLOWER_FAMILIES.items():
        for t in terms:
            if re.search(rf"\b{re.escape(t)}\b", low):
                return family
    return None


def confidence_band(c: float) -> str:
    if c >= 0.90: return "high"
    if c >= 0.75: return "medium"
    if c >= 0.55: return "low"
    return "review_required"


def ai_analyze(items, meta):
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
                  0.04 if x.get("severity") == "medium" else
                  0.02 if x.get("severity") == "low" else 0
                  for x in insights)
    trust = round(max(0.0, min(1.0, trust_base - penalty)), 3)
    doc_type = meta.get("document_type") or "unclassified document"
    reasoning = (f"Parsed a {doc_type} with {len(items)} line item(s). "
                 f"Detected {len(anomalies)} anomaly(ies), {len(insights)} insight(s). "
                 f"Trust {trust:.2f} ({confidence_band(trust)}).")
    return {"trust_score": trust, "confidence_band": confidence_band(trust),
            "reasoning": reasoning, "insights": insights, "anomalies": anomalies,
            "recommendations": recs,
            "analyzed_at": datetime.utcnow().isoformat() + "Z"}


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
            return {"success": True, "items": req.items, "matched_count": 0}
        out = []
        for item in req.items:
            iname = normalize_product_for_match(str(item.get("product_name", "")))
            best, best_score = None, 0
            for product in req.company_products:
                names = [str(product.get("name", ""))]
                aliases = product.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(str(x) for x in aliases)
                for candidate in names:
                    cname = normalize_product_for_match(candidate)
                    if not cname: continue
                    score = max(fuzz.ratio(iname, cname),
                                fuzz.token_set_ratio(iname, cname),
                                fuzz.partial_ratio(iname, cname) if len(iname) >= 5 else 0)
                    if score > best_score:
                        best_score, best = score, product
            conf = best_score / 100.0
            if best and conf >= MIN_MATCH_CONFIDENCE:
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
    except Exception as e:
        logger.exception("Product matching failed")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
#  ENDPOINTS
# ===========================================================================

@app.get("/")
async def root():
    return {"service": "Smart Document Intelligence Engine",
            "version": ENGINE_VERSION, "status": "operational",
            "capabilities": ["structured_table_extraction", "multi_line_row_stitching",
                             "boilerplate_rejection", "contact_and_link_filtering",
                             "row_index_stripping", "confidence_scoring",
                             "anomaly_detection", "trust_score",
                             "product_matching", "currency_detection"],
            "endpoints": ["/api/health", "/api/analyze (POST)",
                          "/api/match-products (POST)", "/api/extract-text (POST)"]}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "smart-import-engine",
            "version": ENGINE_VERSION,
            "ocr_available": bool(pytesseract),
            "scanned_pdf_ocr_available": fitz is not None}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...),
                  company_id: int = Form(0),
                  file_type: Optional[str] = Form(None)):
    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413,
                                detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
        fname = file.filename or "upload"
        ext = (file_type or Path(fname).suffix.lstrip(".")).lower()
        logger.info("Processing %s (%s), company=%s, size=%s",
                    fname, ext, company_id, len(content))

        items, metadata, text_extracted, method = [], {}, "", ""
        if ext in ("xlsx", "xls", "xlsm", "csv"):
            items, metadata = extract_from_excel(content, ext)
            method = "spreadsheet"
        elif ext == "pdf":
            text_extracted, method = extract_text_from_pdf(content)
            items, metadata = parse_order_text(text_extracted)
        elif ext in ("docx", "doc"):
            text_extracted = extract_text_from_docx(content); method = "docx"
            items, metadata = parse_order_text(text_extracted)
        elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
            text_extracted, method = extract_text_from_image(content)
            items, metadata = parse_order_text(text_extracted)
        elif ext == "json":
            text_extracted, method = extract_text_from_json(content)
            items, metadata = parse_order_text(text_extracted)
        else:
            text_extracted = content.decode("utf-8", errors="ignore"); method = "text"
            items, metadata = parse_order_text(text_extracted)

        cleaned = clean_items(items)

        routed = []
        for raw in cleaned:
            try:
                v = ExtractedLineItem(**raw)
                d = v.model_dump()
                d["flower_family"] = infer_flower_family(d["product_name"])
                d["confidence_band"] = confidence_band(d["confidence"])
                routed.append(d)
            except Exception as e:
                logger.warning("Item failed strict routing: %s", e)

        total_boxes = int(sum(float(x.get("boxes") or 0) for x in routed)) or None
        total_qty = int(sum(float(x.get("quantity") or 0) for x in routed)) or None
        total_amount = round(sum(float(x.get("total") or 0) for x in routed), 2)

        warnings: List[str] = []
        for x in routed:
            warnings.extend(x.get("validation_warnings", []))

        analysis = ai_analyze(routed, metadata)

        return {"success": True, "items": routed, "metadata": metadata,
                "document_type": metadata.get("document_type"),
                "total_boxes": total_boxes, "total_quantity": total_qty,
                "total_amount": total_amount,
                "currency": metadata.get("currency", "USD"),
                "item_count": len(routed),
                "review_required": any(x.get("confidence", 0) < 0.80 or
                                       x.get("validation_warnings") for x in routed),
                "warnings": sorted(set(warnings)),
                "analysis": analysis,
                "text_extracted": text_extracted[:12000] if text_extracted else "",
                "extraction_method": method, "file_type": ext,
                "filename": fname, "engine_version": ENGINE_VERSION}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    try:
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
        return {"success": True, "text": text, "length": len(text),
                "file_type": ext, "extraction_method": method}
    except Exception as e:
        logger.exception("extract-text failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
