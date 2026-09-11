"""
Smart Document Intelligence Engine v9.1
Production-hardened: v6.1 reliability + v9.0 intelligence, with hard time budgets.

Design principles:
  1. Never block longer than ANALYZE_TIMEOUT_SECONDS (default 45s).
  2. Try text layers before OCR — OCR is the last resort.
  3. Layout engine first, text reconstruction second, v6.1 parser as safety net.
  4. Every phase wrapped; partial results beat no results.
  5. No startup warm-ups that delay cold-start responses.
  6. Never invent boxes, pack_rate, quantity or price.

API contract identical to v6.1 and v9.0 — no PHP or DB changes needed.

ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import Counter
import io
import os
import re
import json
import math
import time
import logging
import statistics
import concurrent.futures

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
    import openpyxl  # noqa: F401
except Exception:
    openpyxl = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-document-engine")

TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

ENGINE_VERSION = "9.1.0"

app = FastAPI(
    title="Smart Document Intelligence Engine",
    version=ENGINE_VERSION,
    description="Layout-aware extraction with hard time budgets and safe fallbacks.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MIN_MATCH_CONFIDENCE = float(os.getenv("MIN_MATCH_CONFIDENCE", "0.60"))
ANALYZE_TIMEOUT_SECONDS = int(os.getenv("ANALYZE_TIMEOUT_SECONDS", "45"))
OCR_TIMEOUT_SECONDS = int(os.getenv("OCR_TIMEOUT_SECONDS", "15"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "6"))


# ===========================================================================
#  SCHEMA
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
            return cleaned if len(cleaned) >= 2 else None
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
            if abs(exp - self.total) > max(0.05, abs(self.total) * 0.02):
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
#  HELPERS
# ===========================================================================

def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
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


def safe_int(x):
    if x is None:
        return None
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except Exception:
        return x


# ===========================================================================
#  FIELD KNOWLEDGE (unchanged from v6.1 — proven correct)
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
                    "product details", "product / service description",
                    "service description"],
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
                 "quantity invoice", "total stems"],
    "unit_price": ["price", "price per stem", "price/stem", "unit price",
                   "unit price (usd)", "unit price(usd)", "cost",
                   "price per unit", "per stem", "per piece",
                   "amount per stem", "unit cost", "priceperstem",
                   "rate per stem", "selling price", "unit selling price"],
    "total": ["total", "total price", "total amount", "line total",
              "line amount", "sub-total", "subtotal", "extended price",
              "total (usd)", "total(usd)", "line value", "amount"],
    "length": ["length", "length(cm)", "length (cm)", "size", "size(cm)",
               "stem length", "height", "stem size", "length cm"],
    "discount": ["discount", "disc.", "rebate"],
    "tax": ["tax", "vat", "gst", "sales tax"],
}

META_LABELS = {
    "invoice_number": ["invoice number", "invoice no", "invoice #", "invoice no.",
                       "inv no", "inv #", "quotation number", "quotation no",
                       "quote number", "proforma number", "proforma no",
                       "document number", "doc no", "reference", "ref no",
                       "reference number", "document ref"],
    "date": ["date", "date of shipment", "shipment date", "invoice date",
             "issue date", "document date", "quotation date"],
    "due_date": ["due date", "payment due", "valid until", "valid till",
                 "expiry", "expires", "expiration"],
    "currency": ["currency", "currency code", "ccy"],
    "vat_rate": ["vat rate", "vat rate (%)", "tax rate", "tax %", "vat %"],
    "country_destination": ["country of destination", "destination country",
                            "destination", "country dest", "ship to country"],
    "point_of_entry": ["point of entry", "port of entry", "entry point",
                       "port", "airport", "arrival port"],
    "country_origin": ["country of origin", "origin country"],
    "consignee_name": ["consignee name", "consignee", "bill to", "ship to",
                       "customer name", "client name", "buyer name"],
    "consignee_address": ["consignee address", "bill to address",
                          "ship to address", "customer address",
                          "buyer address", "delivery address"],
    "seller_name": ["seller name", "seller", "vendor", "supplier", "exporter"],
    "purchase_order_no": ["purchase order no", "purchase order #",
                          "purchase order number", "purchase order",
                          "po no", "po #", "po number", "customer po",
                          "order number"],
    "payment_terms": ["payment terms", "payment term", "terms of payment"],
    "transportation": ["transportation", "transport", "shipment method",
                       "mode of transport", "shipping method"],
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

CURRENCY_WORDS = {
    "usd": "USD", "us dollar": "USD", "dollar": "USD",
    "kes": "KES", "ksh": "KES", "kenya shilling": "KES",
    "eur": "EUR", "euro": "EUR", "gbp": "GBP", "pound": "GBP",
    "aed": "AED", "sar": "SAR", "qar": "QAR",
}


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
#  PRODUCT / BOILERPLATE CLASSIFIERS
# ===========================================================================

PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-().]{7,}\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9\-]+\.(?:com|net|org|io|ke|co\.ke|tc|info|biz)\b", re.I)
HASH_ID_RE = re.compile(r"^#?[A-Z]{2,}\d{4,}[A-Z0-9\-]*$", re.I)
ADDRESS_RE = re.compile(
    r"\b(street|st\.|road|rd\.|avenue|ave\.|boulevard|blvd\.|building|bldg|"
    r"floor|suite|tower|plaza|po\s*box|p\.o\.|postal\s*code|zip|"
    r"nairobi|dubai|amsterdam|bogota|quito|miami|johannesburg|"
    r"kenya|uae|united\s+arab\s+emirates|netherlands|colombia|ecuador)\b",
    re.IGNORECASE)

BOILERPLATE_RE = re.compile("|".join([
    r"^\s*thank\s+you\b", r"^\s*thanks\b", r"^\s*welcome\b",
    r"computer[- ]generated", r"no\s+signature\s+required",
    r"^\s*generated\s+on\b", r"^\s*scan\s+to\s+verify\b",
    r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$",
    r"^\s*paid\s*$", r"^\s*sent\s*$", r"^\s*unpaid\s*$", r"^\s*draft\s*$",
    r"^\s*(?:order|invoice)\s+details\s*$",
    r"^\s*(?:bill\s+to|ship\s+to|sold\s+to|consignee|consignor)\b",
    r"^\s*sub[- ]?total\b", r"^\s*grand\s+total\b",
    r"^\s*balance\s+due\b", r"^\s*due\s+date\b",
    r"^\s*payment\s+terms\b",
    r"prices?\s+are\s+inclusive\s+of\s+vat",
    r"delivery\s+cost\s+is\s+the\s+buyer",
    r"total\s+price\s+per\s+item\s+is\s+inclusive",
    r"verify\s+at\s*:",
    r"^\s*other\s+charges\b", r"^\s*awb\s+fee\b",
]), re.IGNORECASE)

COMPANY_TERMS = re.compile(
    r"\b(limited|ltd\.?|llc|inc\.?|plc|company|enterprises?|"
    r"investment|trading|holdings?)\b", re.I)

NON_PRODUCT_TERMS = {
    "invoice", "invoice details", "invoice number", "quotation", "proforma",
    "receipt", "delivery note", "consignee", "consignee details",
    "seller", "seller name", "buyer", "customer", "customer details",
    "payment terms", "transportation", "transport", "notes", "items",
    "products", "product/service", "description", "variety", "quantity",
    "price", "price per stem", "unit price", "total", "amount",
    "boxes", "packrate", "pack rate", "farm code", "length",
    "country of destination", "country of origin", "point of entry",
    "date of shipment", "currency", "purchase order", "purchase order #",
    "subtotal", "grand total", "vat", "tax",
}


def is_boilerplate(line: str) -> bool:
    if not line:
        return True
    s = line.strip()
    if BOILERPLATE_RE.search(s):
        return True
    if URL_RE.search(s) or EMAIL_RE.search(s):
        return True
    if HASH_ID_RE.match(s):
        return True
    letters = re.findall(r"[A-Za-z]", s)
    if len(letters) < 3 and len(s) < 30:
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
    if ADDRESS_RE.search(s) and len(s.split()) <= 8:
        return True
    return False


def looks_like_product(name: str) -> bool:
    n = str(name or "").strip()
    if len(n) < 2 or not re.search(r"[A-Za-z]{3,}", n):
        return False
    low = norm(n)
    if low in NON_PRODUCT_TERMS:
        return False
    if "@" in n or URL_RE.search(n):
        return False
    if ADDRESS_RE.search(n):
        return False
    if COMPANY_TERMS.search(n) and not re.search(
            r"\b(rose|roses|flower|flowers|plant|goods|supplies|"
            r"carnation|chrysanthemum|tulip|lily|orchid|gerbera|"
            r"alstroemeria|sunflower|eustoma|hydrangea)\b", n, re.I):
        return False
    if is_boilerplate(n) or is_contact_or_address(n):
        return False
    if match_header(n) and len(n.split()) <= 4:
        return False
    if re.fullmatch(r"[\d\s.,:/()%$€£+\-]+", n):
        return False
    return True


def is_header_or_metadata(line: str) -> bool:
    s = str(line or "").strip()
    if not s:
        return True
    if match_meta_label(s.rstrip(":#")):
        return True
    if match_header(s) and len(s.split()) <= 5:
        return True
    low = norm(s)
    if low in NON_PRODUCT_TERMS:
        return True
    if re.match(r"^(?:invoice|inv|quote|qtn|proforma|po|flr)[\s/#-]*[\w/-]+$", low):
        return True
    if re.fullmatch(r"[\d\s.,:/()%$€£-]+", s):
        return True
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s):
        return True
    return False


# ===========================================================================
#  METADATA
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
        m = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,70}?)\s*"
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
        for word, code in CURRENCY_WORDS.items():
            if word in c:
                meta["currency"] = code
                break
    else:
        for word, code in CURRENCY_WORDS.items():
            if re.search(rf"\b{word}\b", norm(text)):
                meta["currency"] = code
                break

    if "invoice_number" not in meta:
        m = re.search(r"#\s*([A-Z]{2,}-?\d{4,}[A-Z0-9\-]*)", text)
        if m:
            meta["invoice_number"] = m.group(1)

    meta["document_type"] = detect_document_type(text)
    return meta


# ===========================================================================
#  LAYOUT ENGINE — Word positions
# ===========================================================================

class Word:
    __slots__ = ("text", "x0", "y0", "x1", "y1", "page")

    def __init__(self, text, x0, y0, x1, y1, page=0):
        self.text = text
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.page = page

    @property
    def cx(self): return (self.x0 + self.x1) / 2.0

    @property
    def cy(self): return (self.y0 + self.y1) / 2.0


def cluster_rows(words: List[Word], y_tol: float = 4.0) -> List[List[Word]]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (w.page, w.cy, w.x0))
    rows: List[List[Word]] = []
    cur = [words[0]]
    cur_y = words[0].cy
    for w in words[1:]:
        if w.page != cur[0].page or abs(w.cy - cur_y) > y_tol:
            rows.append(sorted(cur, key=lambda x: x.x0))
            cur = [w]
            cur_y = w.cy
        else:
            cur.append(w)
            cur_y = (cur_y * (len(cur) - 1) + w.cy) / len(cur)
    if cur:
        rows.append(sorted(cur, key=lambda x: x.x0))
    return rows


def row_text(row: List[Word]) -> str:
    return " ".join(w.text for w in row)


def detect_column_gaps(rows: List[List[Word]],
                       min_gap: float = 10.0) -> List[Tuple[float, float]]:
    gap_hits: List[Tuple[float, float]] = []
    for row in rows:
        for i in range(len(row) - 1):
            g = row[i + 1].x0 - row[i].x1
            if g >= min_gap:
                gap_hits.append((row[i].x1, row[i + 1].x0))
    if not gap_hits:
        return []
    gap_hits.sort()
    clusters: List[Tuple[float, float]] = []
    for gs, ge in gap_hits:
        if clusters and gs <= clusters[-1][1] + 2:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], ge))
        else:
            clusters.append((gs, ge))
    boundaries = [(a + b) / 2.0 for a, b in clusters]
    page_min = min(w.x0 for row in rows for w in row)
    page_max = max(w.x1 for row in rows for w in row)
    edges = [page_min] + boundaries + [page_max + 1]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def assign_words_to_columns(row: List[Word],
                            cols: List[Tuple[float, float]]) -> List[str]:
    buckets: List[List[Word]] = [[] for _ in cols]
    for w in row:
        placed = False
        for i, (a, b) in enumerate(cols):
            if a <= w.cx < b:
                buckets[i].append(w)
                placed = True
                break
        if not placed:
            best_i, best_d = 0, 1e9
            for i, (a, b) in enumerate(cols):
                mid = (a + b) / 2.0
                d = abs(w.cx - mid)
                if d < best_d:
                    best_d, best_i = d, i
            buckets[best_i].append(w)
    return [" ".join(w.text for w in sorted(b, key=lambda x: x.x0)).strip()
            for b in buckets]


HEADER_TOKENS = {
    "flower", "variety", "length", "pack", "rate", "boxes", "box",
    "total", "stems", "stem", "unit", "price", "amount", "qty", "quantity",
    "description", "product", "service", "item", "no", "#", "s/n",
    "cartons", "carton", "bundles", "bundle", "val", "value",
}


def header_score(row_str: str) -> int:
    s = norm(row_str)
    if "$" in s or len(re.findall(r"\d", s)) >= 3:
        return 0
    toks = re.findall(r"[a-z#/]+", s)
    if not toks:
        return 0
    hits = sum(1 for t in toks if t in HEADER_TOKENS)
    return hits if hits >= 1 and hits >= len(toks) - 1 else 0


def header_columns_for(stitched: str) -> List[str]:
    s = stitched.strip()
    s = re.sub(r"^[#\s]+", "", s)
    cols = re.split(r"\s{3,}|\s*\|\s*|\t+", s)
    cols = [c.strip() for c in cols if c.strip()]
    if len(cols) < 2:
        cols = s.split()
    merged, i = [], 0
    while i < len(cols):
        cur = cols[i]
        nxt = cols[i + 1] if i + 1 < len(cols) else ""
        combo = norm(cur + " " + nxt) if nxt else ""
        if combo in {"unit price", "pack rate", "line total",
                     "total stems", "total amount", "product service",
                     "service description", "flower variety"}:
            merged.append(cur + " " + nxt)
            i += 2
            continue
        merged.append(cur)
        i += 1
    return [c for c in merged if norm(c) not in {"#", "no", "no.", "s/n", "sr"}]


def stitch_header_rows(rows: List[List[Word]]) -> Tuple[Optional[int], Optional[int], List[Optional[str]]]:
    best = None
    i = 0
    while i < len(rows):
        rt = row_text(rows[i])
        if header_score(rt) == 0:
            i += 1
            continue
        start = i
        end = i + 1
        while end < len(rows) and header_score(row_text(rows[end])) > 0:
            end += 1
        parts = [row_text(rows[k]) for k in range(start, end)]
        stitched = " ".join(parts)
        cols = header_columns_for(stitched)
        mapped = [match_header(c) for c in cols]
        known = [m for m in mapped if m and m != "row_index"]
        has_product = any(k in known for k in ("product", "description", "variety"))
        has_numeric = any(k in known for k in ("quantity", "total",
                                                "unit_price", "boxes", "pack_rate"))
        if has_product and has_numeric and len(set(known)) >= 3:
            score = len(set(known)) + (3 if has_product else 0) + (2 if has_numeric else 0)
            if best is None or score > best[0]:
                best = (score, start, end, mapped)
        i = end
    if not best:
        return None, None, []
    _, start, end, mapped = best
    return start, end, mapped


# ===========================================================================
#  CELL ASSIGNMENT
# ===========================================================================

NUMERIC_FIELDS = {"boxes", "pack_rate", "quantity", "unit_price", "total"}


def is_numeric_cell(cell: str) -> bool:
    if not cell:
        return False
    s = cell.strip()
    s = re.sub(r"[\$€£]", "", s)
    s = re.sub(r"(?i)\b(?:usd|kes|ksh|eur|gbp|aed|sar|qar)\b", "", s).strip()
    if not s:
        return False
    return bool(re.fullmatch(r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?", s))


def strip_row_index(text: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^\s*(\d{1,3})\s*[.)]?\s+(.*\S)\s*$", text or "")
    if m and len(m.group(2)) >= 2:
        return m.group(2).strip(), int(m.group(1))
    return text, None


def resolve_row_cells(cells: List[str],
                      headers: List[Optional[str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    raw: Dict[str, str] = {}

    cells = list(cells)
    while cells and not cells[-1]:
        cells.pop()
    if not cells:
        return result

    numeric_slots = [i for i, h in enumerate(headers) if h in NUMERIC_FIELDS]
    product_slots = [i for i, h in enumerate(headers)
                     if h in ("product", "description", "variety")]
    length_slots = [i for i, h in enumerate(headers) if h == "length"]

    num_cells, txt_cells = [], []
    for c in cells:
        (num_cells if is_numeric_cell(c) else txt_cells).append(c)

    assigned: Dict[int, str] = {}

    if txt_cells and product_slots:
        head, ridx = strip_row_index(txt_cells[0])
        if head:
            assigned[product_slots[0]] = head
        if ridx is not None:
            result["row_index"] = ridx
        leftover_text = txt_cells[1:]
    else:
        leftover_text = list(txt_cells)

    for slot in product_slots[1:]:
        if slot in assigned or not leftover_text:
            continue
        assigned[slot] = leftover_text.pop(0)

    remaining_nums = list(num_cells)
    for slot in length_slots:
        for k, v in enumerate(remaining_nums):
            n = parse_number(v)
            if n is not None and 20 <= n <= 250:
                assigned[slot] = v
                remaining_nums.pop(k)
                break

    open_slots = [s for s in numeric_slots if s not in assigned]
    for slot, val in zip(open_slots, remaining_nums):
        assigned[slot] = val

    consumed = set(assigned.values())
    for v in num_cells:
        if v not in consumed:
            raw[f"unmapped_numeric_{len(raw)+1}"] = v

    for i, val in assigned.items():
        field = headers[i]
        if not field or field == "row_index":
            continue
        if field in NUMERIC_FIELDS:
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


def flush_row(cells: List[str], headers: List[Optional[str]],
              items: List[Dict[str, Any]]) -> None:
    if not cells or not any(c.strip() for c in cells):
        return
    row_dict = resolve_row_cells(cells, headers)
    name = None
    for key in ("product", "description", "variety"):
        v = row_dict.get(key)
        if v and looks_like_product(str(v)):
            name = str(v).strip()
            break
    if not name:
        for c in cells:
            if looks_like_product(c):
                name = c.strip()
                break
    if not name:
        return
    name, ridx = strip_row_index(name)
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
    ridx_final = row_dict.get("row_index", ridx)
    if ridx_final is not None:
        item["row_index"] = ridx_final
    items.append(item)


# ===========================================================================
#  LAYOUT-BASED TABLE RECONSTRUCTION
# ===========================================================================

def reconstruct_table_from_grid(rows: List[List[Word]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    h_start, h_end, headers = stitch_header_rows(rows)
    if h_start is None:
        return []

    data_rows = rows[h_end:]
    cols = detect_column_gaps(data_rows, min_gap=10.0)
    if not cols:
        cols = [(min(w.x0 for row in rows for w in row),
                 max(w.x1 for row in rows for w in row) + 1)]

    items: List[Dict[str, Any]] = []
    pending: Optional[List[str]] = None

    for row in data_rows:
        rtext = row_text(row).strip()
        if not rtext:
            continue
        if header_score(rtext) > 0:
            _, _, new_headers = stitch_header_rows([row])
            if new_headers and sum(1 for h in new_headers if h and h != "row_index") >= 3:
                if pending:
                    flush_row(pending, headers, items)
                    pending = None
                headers = new_headers
            continue
        if is_boilerplate(rtext):
            continue
        if re.match(r"^\s*(?:sub\s*total|grand\s*total|awb\s*fee|other\s*charges|"
                    r"vat|tax|balance|amount\s+due|notes?|delivery|discount)\b",
                    rtext, re.I):
            continue

        cells = assign_words_to_columns(row, cols)
        has_num = any(is_numeric_cell(c) for c in cells)
        has_lead_index = bool(re.match(r"^\d{1,3}[.)]?\s+", rtext))

        if pending and not (has_num or has_lead_index):
            pending[0] = (pending[0] + " " + cells[0]).strip()
            continue
        if pending:
            flush_row(pending, headers, items)
        pending = cells

    if pending:
        flush_row(pending, headers, items)
    return items


# ===========================================================================
#  TEXT-BASED RECONSTRUCTION (no word positions available)
# ===========================================================================

def extract_table_from_text(text: str) -> List[Dict[str, Any]]:
    lines = [x.rstrip() for x in text.splitlines()]

    # Find header block
    h_start = h_end = None
    headers: List[Optional[str]] = []
    i = 0
    while i < len(lines):
        if header_score(lines[i]) > 0:
            start = i
            end = i + 1
            while end < len(lines) and header_score(lines[end]) > 0:
                end += 1
            stitched = " ".join(lines[k] for k in range(start, end))
            cols = header_columns_for(stitched)
            mapped = [match_header(c) for c in cols]
            known = [m for m in mapped if m and m != "row_index"]
            if (any(k in known for k in ("product", "description", "variety"))
                    and any(k in known for k in ("quantity", "total", "unit_price",
                                                  "boxes", "pack_rate"))
                    and len(set(known)) >= 3):
                h_start, h_end, headers = start, end, mapped
                break
            i = end
        else:
            i += 1

    if h_start is None:
        return []

    expected_cols = sum(1 for h in headers if h)

    def split_row(s: str) -> List[str]:
        s = s.strip()
        if not s:
            return []
        if "|" in s:
            parts = [p.strip() for p in re.split(r"\s*\|\s*", s) if p.strip()]
            if len(parts) >= 2:
                return parts
        if "\t" in s:
            parts = [p.strip() for p in s.split("\t") if p.strip()]
            if len(parts) >= 2:
                return parts
        wide = [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]
        if len(wide) >= max(2, expected_cols - 1):
            return wide
        cells: List[str] = []
        remaining = s
        for _ in range(expected_cols - 1):
            m = re.search(
                r"(?:\s|^)((?:[$€£]\s*)?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?"
                r"(?:\s*(?:KES|KSH|USD|EUR|GBP|AED|SAR|QAR))?)\s*$",
                remaining, re.IGNORECASE)
            if not m or m.start(1) == 0:
                break
            cells.insert(0, m.group(1).strip())
            remaining = remaining[:m.start(1)].rstrip()
        head = re.sub(r"^\d{1,3}[.)]?\s+", "", remaining).strip() if remaining else ""
        return ([head] + cells) if cells else [s]

    items: List[Dict[str, Any]] = []
    pending: List[str] = []

    for line in lines[h_end:]:
        s = line.strip()
        if not s:
            continue
        if header_score(s) > 0:
            start = lines.index(line)
            end = start + 1
            while end < len(lines) and header_score(lines[end]) > 0:
                end += 1
            stitched = " ".join(lines[k] for k in range(start, end))
            new_headers = [match_header(c) for c in header_columns_for(stitched)]
            if sum(1 for h in new_headers if h and h != "row_index") >= 3:
                if pending:
                    flush_row(pending, headers, items)
                    pending = []
                headers = new_headers
                expected_cols = sum(1 for h in headers if h)
            continue
        if is_boilerplate(s):
            continue
        if re.match(r"^\s*(?:sub\s*total|grand\s*total|awb\s*fee|other\s*charges|"
                    r"vat|tax|balance|amount\s+due|notes?|delivery|discount)\b",
                    s, re.I):
            continue

        cells = split_row(s)
        has_num = any(is_numeric_cell(c) for c in cells)
        has_lead_index = bool(re.match(r"^\d{1,3}[.)]?\s+", s))

        if pending and not (has_num or has_lead_index):
            pending[0] = (pending[0] + " " + cells[0]).strip()
            continue
        if pending:
            flush_row(pending, headers, items)
        pending = cells

    if pending:
        flush_row(pending, headers, items)
    return items


# ===========================================================================
#  v6.1-STYLE FREE-FORM FALLBACK (safety net)
# ===========================================================================

def parse_inline_line_v61(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s or is_header_or_metadata(s):
        return None
    if is_boilerplate(s) or is_contact_or_address(s):
        return None

    # Peel trailing numerics
    nums: List[str] = []
    remaining = s
    for _ in range(6):
        m = re.search(
            r"(?:\s|^)((?:[$€£]\s*)?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?"
            r"(?:\s*(?:KES|KSH|USD|EUR|GBP|AED|SAR|QAR))?)\s*$",
            remaining, re.IGNORECASE)
        if not m or m.start(1) == 0:
            break
        nums.insert(0, m.group(1).strip())
        remaining = remaining[:m.start(1)].rstrip()

    name, ridx = strip_row_index(remaining.strip())
    if not looks_like_product(name):
        return None

    parsed_nums = [parse_number(n) for n in nums]
    parsed_nums = [n for n in parsed_nums if n is not None]

    item: Dict[str, Any] = {
        "product_name": name,
        "boxes": None, "pack_rate": None, "quantity": None,
        "unit_price": None, "total": None,
        "specification": {},
    }
    if ridx is not None:
        item["row_index"] = ridx

    # Heuristic: assign from the right: total, unit_price, then quantity/etc.
    if len(parsed_nums) >= 3:
        item["total"] = as_number(parsed_nums[-1])
        item["unit_price"] = as_number(parsed_nums[-2])
        item["quantity"] = as_number(parsed_nums[-3])
        if len(parsed_nums) >= 4:
            item["pack_rate"] = as_number(parsed_nums[-4])
        if len(parsed_nums) >= 5:
            item["boxes"] = as_number(parsed_nums[-5])
    elif len(parsed_nums) == 2:
        item["unit_price"] = as_number(parsed_nums[-2])
        item["total"] = as_number(parsed_nums[-1])
    elif len(parsed_nums) == 1:
        item["quantity"] = as_number(parsed_nums[0])

    return item


# ===========================================================================
#  EXCEL / CSV
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
        if not empty_field(vals.get("length")):
            lv = str(vals["length"]).strip()
            m = re.search(r"\d+(?:\.\d+)?", lv)
            if m:
                item["specification"]["length"] = f"{m.group(0)}cm"
        items.append(item)
    return clean_items(items), {}


def extract_from_spreadsheet(content: bytes, ext: str):
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
    "sunflower": ["sunflower", "sunflowers"],
    "eustoma": ["eustoma", "lisianthus"],
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
        recs.append("Re-upload a higher-resolution scan or a text-based PDF.")
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
#  FILE READING — every path wrapped, budgeted, and text-first
# ===========================================================================

def _pymupdf_words_and_text(content: bytes) -> Tuple[List[Word], str]:
    if fitz is None:
        return [], ""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return [], ""
    words: List[Word] = []
    chunks: List[str] = []
    try:
        for page_no, page in enumerate(doc):
            try:
                for w in page.get_text("words"):
                    x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
                    if txt and txt.strip():
                        words.append(Word(txt, x0, y0, x1, y1, page=page_no))
                chunks.append(page.get_text("text") or "")
            except Exception:
                continue
    except Exception:
        logger.exception("PyMuPDF layout failed")
    return words, "\n".join(chunks)


def _pypdf2_text(content: bytes) -> str:
    parts = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                pass
    except Exception:
        logger.warning("PyPDF2 failed")
    return "\n".join(parts).strip()


def _preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    w, h = img.size
    if max(w, h) > 2000:
        s = 2000 / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    elif max(w, h) < 1200:
        s = 1200 / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    img = ImageEnhance.Contrast(img).enhance(1.5)
    return img


def _ocr_text(img: Image.Image) -> str:
    try:
        processed = _preprocess_image(img)
        return pytesseract.image_to_string(
            processed, lang="eng", config="--oem 3 --psm 6",
            timeout=OCR_TIMEOUT_SECONDS) or ""
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        return ""


def _ocr_words(img: Image.Image) -> List[Word]:
    try:
        processed = _preprocess_image(img)
        data = pytesseract.image_to_data(
            processed, lang="eng", config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT, timeout=OCR_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("OCR words failed: %s", e)
        return []
    out: List[Word] = []
    for i in range(len(data.get("text", []))):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0.0
        if conf < 25:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append(Word(txt, x, y, x + w, y + h, page=0))
    return out


def _docx_text(content: bytes) -> str:
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


def read_document(content: bytes, ext: str) -> Tuple[List[Word], str, str]:
    """
    Return (words, text, method). Text-first for PDFs.
    Never raises; worst case returns ("", "", "failed").
    """
    try:
        if ext == "pdf":
            words, txt = _pymupdf_words_and_text(content)
            if len(re.sub(r"\s+", "", txt)) >= 15:
                return words, txt, "pdf_layout" if words else "pdf_text"
            txt2 = _pypdf2_text(content)
            if len(re.sub(r"\s+", "", txt2)) >= 15:
                return [], txt2, "pdf_text"
            # Genuine scan → OCR, but capped and budgeted
            if fitz is None:
                return [], txt or txt2, "pdf_text_empty"
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                all_words: List[Word] = []
                texts: List[str] = []
                for idx, page in enumerate(doc):
                    if idx >= MAX_OCR_PAGES:
                        break
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5),
                                              alpha=False)
                        img = Image.frombytes("RGB",
                                              [pix.width, pix.height],
                                              pix.samples)
                        all_words.extend(_ocr_words(img))
                        texts.append(_ocr_text(img))
                    except Exception:
                        continue
                return all_words, "\n".join(texts), "pdf_ocr"
            except Exception:
                logger.exception("PDF OCR failed")
                return [], txt or txt2, "pdf_text_empty"

        if ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
            try:
                img = Image.open(io.BytesIO(content))
                return _ocr_words(img), _ocr_text(img), "image_ocr"
            except Exception:
                logger.exception("Image failed")
                return [], "", "image_error"

        if ext in ("docx", "doc"):
            return [], _docx_text(content), "docx"

        if ext == "json":
            try:
                obj = json.loads(content.decode("utf-8", errors="ignore"))
                return [], json.dumps(obj, ensure_ascii=False, indent=2), "json"
            except Exception:
                return [], content.decode("utf-8", errors="ignore"), "text"

        return [], content.decode("utf-8", errors="ignore"), "text"
    except Exception:
        logger.exception("read_document failed")
        return [], "", "failed"


# ===========================================================================
#  PIPELINE
# ===========================================================================

def run_pipeline(content: bytes, fname: str, ext: str,
                 deadline: float) -> Dict[str, Any]:
    """
    deadline is a time.monotonic() timestamp. Every phase checks against it.
    """
    def out_of_time() -> bool:
        return time.monotonic() > deadline

    # Spreadsheets: fast path
    if ext in ("xlsx", "xls", "xlsm", "csv"):
        items, meta = extract_from_spreadsheet(content, ext)
        return {"items": items, "metadata": meta or {},
                "text_extracted": "", "extraction_method": "spreadsheet"}

    words, text, method = read_document(content, ext)

    # Metadata always extracted from raw text
    meta: Dict[str, Any] = {}
    if text:
        try:
            meta = extract_meta_from_text(text)
        except Exception:
            logger.exception("Meta extraction failed")

    items: List[Dict[str, Any]] = []

    # Phase 1: layout engine (only if we have word positions)
    if words and not out_of_time():
        try:
            rows = cluster_rows(words, y_tol=4.0)
            items = reconstruct_table_from_grid(rows)
            if items:
                return {"items": clean_items(items), "metadata": meta,
                        "text_extracted": text, "extraction_method": method}
        except Exception:
            logger.exception("Layout reconstruction failed")

    # Phase 2: text-based table reconstruction
    if text and not items and not out_of_time():
        try:
            items = extract_table_from_text(text)
            if items:
                return {"items": clean_items(items), "metadata": meta,
                        "text_extracted": text, "extraction_method": method}
        except Exception:
            logger.exception("Text table reconstruction failed")

    # Phase 3: v6.1 free-form fallback (safety net)
    if text and not items and not out_of_time():
        try:
            for line in text.splitlines():
                if out_of_time():
                    break
                item = parse_inline_line_v61(line)
                if item:
                    items.append(item)
        except Exception:
            logger.exception("Free-form fallback failed")

    return {"items": clean_items(items), "metadata": meta,
            "text_extracted": text, "extraction_method": method}


def analyze_sync(content: bytes, fname: str, ext: str) -> Dict[str, Any]:
    """
    Run pipeline with a hard wall-clock budget. On overrun, return whatever
    partial items were produced (or an empty list) rather than erroring.
    """
    deadline = time.monotonic() + ANALYZE_TIMEOUT_SECONDS
    pipeline: Dict[str, Any] = {"items": [], "metadata": {},
                                 "text_extracted": "", "extraction_method": "pending"}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_pipeline, content, fname, ext, deadline)
            try:
                pipeline = fut.result(timeout=ANALYZE_TIMEOUT_SECONDS + 5)
            except concurrent.futures.TimeoutError:
                logger.warning("Pipeline exceeded %ss; returning partial result",
                               ANALYZE_TIMEOUT_SECONDS)
                pipeline["extraction_method"] = "timed_out_partial"
    except Exception:
        logger.exception("Pipeline crashed; returning empty result")
        pipeline["extraction_method"] = "error_partial"

    routed: List[Dict[str, Any]] = []
    for raw in pipeline.get("items", []):
        try:
            v = ExtractedLineItem(**raw)
            d = v.model_dump()
            d["flower_family"] = infer_flower_family(d["product_name"])
            d["confidence_band"] = confidence_band(d["confidence"])
            routed.append(d)
        except Exception as e:
            logger.warning("Item failed strict routing: %s", e)

    total_boxes = safe_int(safe_sum(x.get("boxes") for x in routed))
    total_qty = safe_int(safe_sum(x.get("quantity") for x in routed))
    total_amount = round(safe_sum(x.get("total") for x in routed), 2)

    warnings: List[str] = []
    for x in routed:
        warnings.extend(x.get("validation_warnings", []))

    meta = pipeline.get("metadata") or {}
    analysis = ai_analyze(routed, meta)

    return {
        "success": True,
        "items": routed,
        "metadata": meta,
        "document_type": meta.get("document_type"),
        "total_boxes": total_boxes,
        "total_quantity": total_qty,
        "total_amount": total_amount,
        "currency": meta.get("currency", "USD"),
        "item_count": len(routed),
        "review_required": any(
            x.get("confidence", 0) < 0.80 or x.get("validation_warnings")
            for x in routed),
        "warnings": sorted(set(warnings)),
        "analysis": analysis,
        "text_extracted": (pipeline.get("text_extracted") or "")[:12000],
        "extraction_method": pipeline.get("extraction_method"),
        "file_type": ext,
        "filename": fname,
        "engine_version": ENGINE_VERSION,
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
            "capabilities": [
                "layout_aware_extraction", "table_grid_reconstruction",
                "multi_line_row_stitching", "semantic_row_tagging",
                "boilerplate_suppression", "currency_detection",
                "confidence_scoring", "anomaly_detection", "trust_score",
                "product_matching", "pdf_layout", "pdf_text", "pdf_ocr",
                "image_ocr", "docx", "xlsx", "csv", "json", "text",
                "hard_time_budget", "partial_result_safety",
            ],
            "endpoints": ["/api/health", "/api/ping", "/api/analyze (POST)",
                          "/api/match-products (POST)", "/api/extract-text (POST)"]}


@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "smart-import-engine",
            "version": ENGINE_VERSION,
            "ocr_available": bool(pytesseract),
            "pdf_layout_available": fitz is not None,
            "analyze_timeout_seconds": ANALYZE_TIMEOUT_SECONDS}


@app.get("/api/ping")
def ping():
    return {"ok": True, "t": datetime.utcnow().isoformat() + "Z",
            "version": ENGINE_VERSION}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...),
                  company_id: int = Form(0),
                  file_type: Optional[str] = Form(None)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
    fname = file.filename or "upload"
    ext = (file_type or Path(fname).suffix.lstrip(".")).lower()

    logger.info("Analyzing %s (%s), company=%s, size=%s, budget=%ss",
                fname, ext, company_id, len(content), ANALYZE_TIMEOUT_SECONDS)

    try:
        result = await run_in_threadpool(analyze_sync, content, fname, ext)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze crashed")
        # Last-resort: never return 500 — return a well-formed empty response
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

    def _work():
        words, text, method = read_document(content, ext)
        return {"success": True, "text": text, "length": len(text),
                "file_type": ext, "extraction_method": method,
                "word_count": len(words)}

    return await run_in_threadpool(_work)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
