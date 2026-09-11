"""
Smart Document Intelligence Engine v10.0
ALTECH SOFTWARE DEVELOPERS

A local/open-source, confidence-aware document understanding engine for:
- invoices, quotations, proformas, receipts, delivery notes, purchase orders
- flower/export orders and semi-structured business documents
- PDF (text + layout + scanned/OCR), DOCX, XLSX/XLSM, CSV, JSON and plain text
- pasted/free-form text with no clean table layout

Design:
1. Multi-source extraction: text, PDF geometry, OCR, DOCX tables, spreadsheets.
2. Layout-aware table reconstruction using word coordinates when available.
3. Schema/label intelligence with exact, normalized and fuzzy matching.
4. Multi-hypothesis parsing instead of one brittle parser.
5. Evidence/provenance for every important field.
6. Conservative reconciliation: arithmetic is used to validate, not blindly invent.
7. Product matching is alias-aware and confidence-gated.
8. Explicit ambiguity is returned for human review rather than silently guessing.

No paid AI API is required.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import mimetypes
import os
import re
import statistics
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import PyPDF2
import docx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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


ENGINE_VERSION = "10.0.0"
logger = logging.getLogger("smart-document-engine")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ANALYZE_TIMEOUT_SECONDS = float(os.getenv("ANALYZE_TIMEOUT_SECONDS", "55"))
MIN_PRODUCT_MATCH = float(os.getenv("MIN_PRODUCT_MATCH", "0.84"))

ALLOWED_EXTENSIONS = {
    "pdf", "docx", "doc", "xlsx", "xlsm", "xls", "csv", "json",
    "txt", "text", "md", "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"
}

TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD


app = FastAPI(
    title="Smart Document Intelligence Engine",
    version=ENGINE_VERSION,
    description="Open-source, confidence-aware document and order extraction engine.",
)

# Configure origins instead of shipping a wildcard in production.
cors_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [x.strip() for x in cors_raw.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# FIELD KNOWLEDGE
# ---------------------------------------------------------------------------

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "product": [
        "product", "product name", "product/service", "product / service",
        "product or service", "item", "item name", "service", "article",
        "articles", "commodity", "goods", "stock item", "particulars",
        "product description", "item description", "item/service",
    ],
    "variety": [
        "variety", "flower", "flower name", "flower type", "species",
        "cultivar", "kind", "variety name", "flower variety",
    ],
    "description": [
        "description", "desc", "details", "item details", "specification",
        "specifications", "remarks", "product details", "description/details",
    ],
    "farm_code": [
        "farm code", "farmcode", "farm reference", "farm ref",
        "supplier code", "grower code", "grower reference", "grower ref",
    ],
    "boxes": [
        "boxes", "box", "bx", "cartons", "carton", "ctn", "cases", "case",
        "bundles", "bundle", "packages", "pkg", "packs",
        "number of boxes", "no. of boxes", "no boxes", "box qty",
        "carton qty", "cartons qty",
    ],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box", "quantity per box",
        "stems per carton", "qty/carton", "quantity/carton",
        "conversion rate", "pack rate per box", "pack",
    ],
    "quantity": [
        "quantity", "qty", "qnty", "stems", "pcs", "pieces", "count",
        "total quantity", "total qty", "number of stems", "stem quantity",
        "invoice quantity", "qty invoice", "quantity invoice", "units",
        "unit quantity", "ordered quantity",
    ],
    "unit_price": [
        "price", "price per stem", "price/stem", "unit price",
        "unit price (usd)", "unit price(usd)", "cost", "price per unit",
        "per stem", "per piece", "amount per stem", "unit cost",
        "priceperstem", "rate per stem", "selling price",
        "unit selling price", "rate", "unit rate",
    ],
    "total": [
        "total", "total price", "total amount", "line total", "line amount",
        "sub-total", "subtotal", "extended price", "total (usd)",
        "total(usd)", "line value", "amount",
    ],
    "length": [
        "length", "length(cm)", "length (cm)", "size", "size(cm)",
        "stem length", "height", "stem size", "length cm", "cm",
    ],
    "discount": ["discount", "disc.", "rebate"],
    "tax": ["tax", "vat", "gst", "sales tax"],
}

META_LABELS: Dict[str, List[str]] = {
    "invoice_number": [
        "invoice number", "invoice no", "invoice #", "invoice no.",
        "inv no", "inv #", "quotation number", "quotation no",
        "quote number", "proforma number", "proforma no",
        "document number", "doc no", "reference", "ref no",
        "reference number", "document ref", "order number",
    ],
    "date": [
        "date", "date of shipment", "shipment date", "invoice date",
        "issue date", "document date", "quotation date", "order date",
    ],
    "due_date": [
        "due date", "payment due", "valid until", "valid till", "expiry",
        "expires", "expiration",
    ],
    "currency": ["currency", "currency code", "ccy"],
    "vat_rate": ["vat rate", "vat rate (%)", "tax rate", "tax %", "vat %"],
    "country_destination": [
        "country of destination", "destination country", "destination",
        "country dest", "ship to country", "country destination",
    ],
    "point_of_entry": [
        "point of entry", "port of entry", "entry point", "port", "airport",
        "arrival port",
    ],
    "country_origin": ["country of origin", "origin country"],
    "consignee_name": [
        "consignee name", "consignee", "bill to", "ship to",
        "customer name", "client name", "buyer name", "consignee details",
    ],
    "consignee_address": [
        "consignee address", "bill to address", "ship to address",
        "customer address", "buyer address", "delivery address",
    ],
    "seller_name": [
        "seller name", "seller", "vendor", "supplier", "exporter",
        "seller / exporter", "exporter name",
    ],
    "purchase_order_no": [
        "purchase order no", "purchase order #", "purchase order number",
        "purchase order", "po no", "po #", "po number", "customer po",
    ],
    "payment_terms": [
        "payment terms", "payment term", "terms of payment", "payment",
    ],
    "transportation": [
        "transportation", "transport", "shipment method",
        "mode of transport", "shipping method", "transportation details",
    ],
    "awb_number": [
        "awb number", "awb no", "awb", "air waybill", "tracking number",
        "waybill", "airway bill",
    ],
    "net_weight": [
        "net weight", "net weight (kgs)", "net weight (kg)", "net kg",
        "net weight kgs",
    ],
    "notes": ["notes", "note", "comments", "comment", "remarks"],
}

DOC_TYPES = {
    "invoice": ["invoice", "tax invoice", "commercial invoice"],
    "quotation": ["quotation", "quote", "estimate"],
    "proforma": ["proforma", "pro forma", "pro-forma", "proforma invoice"],
    "receipt": ["receipt", "payment receipt"],
    "delivery_note": ["delivery note", "delivery", "dispatch note"],
    "credit_note": ["credit note", "credit memo"],
    "purchase_order": ["purchase order", "purchase order form", "po"],
}

CURRENCY_WORDS = {
    "usd": "USD", "us dollar": "USD", "dollar": "USD", "us$": "USD",
    "kes": "KES", "ksh": "KES", "kenya shilling": "KES",
    "eur": "EUR", "euro": "EUR", "gbp": "GBP", "pound": "GBP",
    "aed": "AED", "sar": "SAR", "qar": "QAR", "jpy": "JPY",
}

NON_PRODUCT_TERMS = {
    "invoice", "invoice details", "invoice number", "quotation", "proforma",
    "receipt", "delivery note", "consignee", "consignee details", "seller",
    "seller name", "buyer", "customer", "customer details", "payment terms",
    "transportation", "transport", "notes", "items", "products",
    "product/service", "description", "variety", "quantity", "price",
    "price per stem", "unit price", "total", "amount", "boxes", "packrate",
    "pack rate", "farm code", "length", "country of destination",
    "country of origin", "point of entry", "date of shipment", "currency",
    "purchase order", "purchase order #", "subtotal", "grand total", "vat",
    "tax",
}

COMPANY_TERMS = re.compile(
    r"\b(limited|ltd\.?|llc|inc\.?|plc|company|enterprises?|investment|"
    r"trading|holdings?)\b", re.I
)
ADDRESS_TERMS = re.compile(
    r"\b(street|st\.|road|rd\.|avenue|ave\.|building|bldg|floor|suite|"
    r"tower|plaza|po box|postal code|p\.o\.)\b", re.I
)


# ---------------------------------------------------------------------------
# SMALL UTILITIES
# ---------------------------------------------------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def norm(value: Any) -> str:
    s = "" if value is None else str(value)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def compact_norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(value))


def empty_field(value: Any) -> bool:
    if value is None:
        return True
    return norm(value) in {"", "n/a", "na", "null", "none", "-", "—"}


def as_number(value: Any) -> Optional[float]:
    n = parse_number(value)
    if n is None:
        return None
    return int(n) if float(n).is_integer() else n


def parse_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)

    txt = str(value).strip()
    if not txt:
        return None

    # Handle common OCR substitutions without changing ordinary text.
    txt = txt.replace("O", "0") if re.fullmatch(r"[\sO0-9,.$€£%+\-]+", txt, re.I) else txt
    txt = re.sub(
        r"(?i)\b(?:usd|us\$|kes|ksh|eur|gbp|aed|sar|qar|jpy)\b", "", txt
    )
    txt = txt.replace("$", "").replace("€", "").replace("£", "").strip()
    txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt)

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


def safe_sum(values: Iterable[Any]) -> float:
    total = 0.0
    for v in values:
        n = parse_number(v)
        if n is not None:
            total += n
    return total


def safe_int(value: Any) -> Any:
    n = parse_number(value)
    if n is None:
        return value
    return int(n) if float(n).is_integer() else n


def clean_ocr_text(text: str) -> str:
    text = (text or "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\|", " |", text)
    text = re.sub(r"\|\s+", "| ", text)
    return text


# ---------------------------------------------------------------------------
# LABEL INTELLIGENCE
# ---------------------------------------------------------------------------

def match_header(label: str) -> Optional[str]:
    l = norm(label)
    l = re.sub(r"[*:.#]+$", "", l).strip()
    if not l:
        return None

    exact = {norm(x): field for field, names in COLUMN_SYNONYMS.items() for x in names}
    if l in exact:
        return exact[l]

    # Composite headers frequently appear as "total stems", "price usd",
    # "qty/stems", etc. Prefer semantically distinctive fields first.
    priority = [
        "pack_rate", "unit_price", "farm_code", "quantity", "boxes",
        "length", "total", "product", "variety", "description"
    ]
    for field in priority:
        for candidate in COLUMN_SYNONYMS[field]:
            c = norm(candidate)
            if len(c) >= 4 and (c in l or l in c):
                return field

    best_field, best_score = None, 0.0
    for field, names in COLUMN_SYNONYMS.items():
        for candidate in names:
            c = norm(candidate)
            score = max(
                fuzz.ratio(l, c),
                fuzz.token_set_ratio(l, c),
                fuzz.WRatio(l, c),
            )
            if score > best_score:
                best_score, best_field = score, field

    return best_field if best_score >= 86 else None


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

    best_meta, best_score = None, 0.0
    for meta, names in META_LABELS.items():
        for n in names:
            score = max(fuzz.ratio(key, norm(n)), fuzz.WRatio(key, norm(n)))
            if score > best_score:
                best_score, best_meta = score, meta
    return best_meta if best_score >= 90 else None


def detect_document_type(text: str) -> Optional[str]:
    low = norm(text[:20000])
    if not low:
        return None

    # Phrase hits are stronger than generic fuzzy matching.
    phrase_scores = {}
    for typ, words in DOC_TYPES.items():
        phrase_scores[typ] = max((100 if norm(w) in low else 0 for w in words), default=0)

    best_typ, best = max(phrase_scores.items(), key=lambda x: x[1])
    if best >= 100:
        return best_typ

    # Avoid the old partial_ratio(text, "po") problem where any short word
    # can accidentally classify a document as a purchase order.
    scores = {}
    for typ, words in DOC_TYPES.items():
        scores[typ] = max(
            (fuzz.partial_ratio(norm(w), low[:12000]) for w in words if len(norm(w)) >= 4),
            default=0,
        )
    typ, score = max(scores.items(), key=lambda x: x[1])
    return typ if score >= 82 else None


# ---------------------------------------------------------------------------
# PRODUCT SAFETY / QUALITY
# ---------------------------------------------------------------------------

def looks_like_product(name: str) -> bool:
    n = str(name or "").strip()
    if len(n) < 2 or not re.search(r"[A-Za-z]{2,}", n):
        return False
    low = norm(n)
    if low in NON_PRODUCT_TERMS:
        return False
    if "@" in n or re.search(r"https?://|www\.", n, re.I):
        return False
    if ADDRESS_TERMS.search(n):
        return False
    if COMPANY_TERMS.search(n) and not re.search(
        r"\b(rose|roses|flower|flowers|plant|goods|supplies)\b", n, re.I
    ):
        return False
    if not re.search(r"[A-Za-z]{3,}", n):
        return False
    return True


def is_probable_header(line: str) -> bool:
    s = str(line or "").strip()
    if not s:
        return True
    if match_meta_label(s.rstrip(":#")):
        return True
    mapped = [match_header(x) for x in split_columns(s)]
    if len([x for x in mapped if x]) >= 2 and not re.search(r"\d", s):
        return True
    low = norm(s)
    if low in NON_PRODUCT_TERMS:
        return True
    if re.fullmatch(r"[\d\s.,:/()%$€£-]+", s):
        return True
    return False


def is_boilerplate(s: str) -> bool:
    low = norm(s)
    if low in NON_PRODUCT_TERMS:
        return True
    if re.search(r"\b(grand total|subtotal|sub total|amount due|balance due)\b", low):
        return True
    return False


def is_contact_or_address(s: str) -> bool:
    if "@" in s or re.search(r"https?://|www\.", s, re.I):
        return True
    if ADDRESS_TERMS.search(s):
        return True
    return False


# ---------------------------------------------------------------------------
# META EXTRACTION
# ---------------------------------------------------------------------------

def extract_meta_from_text(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    lines = [x.strip() for x in clean_ocr_text(text).splitlines() if x.strip()]

    for i, line in enumerate(lines):
        m = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,80}?)\s*"
            r"(?:[:#]\s*|-\s+|\|\s*)(.*?)\s*$",
            line,
        )
        if m:
            label, value = m.group(1), m.group(2).strip()
            field_name = match_meta_label(label)
            if field_name and value and not empty_field(value):
                meta[field_name] = value
                continue

        # "Invoice Number" on one line, value on the next.
        field_name = match_meta_label(line.rstrip(":#"))
        if field_name and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and not match_meta_label(nxt):
                if field_name not in meta and len(nxt) < 160:
                    meta[field_name] = nxt

    if "currency" in meta:
        c = norm(meta["currency"])
        for word, code in CURRENCY_WORDS.items():
            if word in c:
                meta["currency"] = code
                break

    meta["document_type"] = detect_document_type(text)
    return meta


# ---------------------------------------------------------------------------
# FIELD EXTRACTION FROM FREE FORM TEXT
# ---------------------------------------------------------------------------

FIELD_PATTERNS = {
    "boxes": r"(?:no\.?\s*of\s*)?(?:boxes?|bx|cartons?|ctn|cases?|bundles?|packages?)",
    "pack_rate": r"(?:pack\s*rate|packrate|stems?\s*(?:per|/)\s*(?:box|carton)|qty\s*(?:per|/)\s*(?:box|carton)|quantity\s*(?:per|/)\s*(?:box|carton))",
    "quantity": r"(?:quantity|qty|qnty|total\s+qty|total\s+quantity|invoice\s+qty|invoice\s+quantity|stems?|pieces?|pcs|units?)",
    "unit_price": r"(?:price\s*(?:per|/)\s*(?:stem|piece|unit)|price\s*per\s*stem|price/stem|unit\s*price|unit\s*cost|rate\s*per\s*stem|selling\s*price|cost\s*per\s*unit|unit\s*rate)",
    "total": r"(?:line\s+total|total\s+amount|total\s+price|line\s+amount|extended\s+price|amount)",
    "length": r"(?:length(?:\s*\(?(?:cm|cms)\)?)?|stem\s+length|size)\b",
    "farm_code": r"(?:farm\s*code|farm\s*ref(?:erence)?|grower\s*code|supplier\s*code)",
}


def extract_labeled_fields(line: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    patterns = [
        ("pack_rate", FIELD_PATTERNS["pack_rate"]),
        ("unit_price", FIELD_PATTERNS["unit_price"]),
        ("farm_code", FIELD_PATTERNS["farm_code"]),
        ("quantity", FIELD_PATTERNS["quantity"]),
        ("boxes", FIELD_PATTERNS["boxes"]),
        ("length", FIELD_PATTERNS["length"]),
        ("total", FIELD_PATTERNS["total"]),
    ]

    for field_name, label_pat in patterns:
        m = re.search(
            rf"\b{label_pat}\b\s*(?:[:=]\s*|\s+)([^|,;]+)",
            line, re.I,
        )
        if not m:
            continue
        raw = m.group(1).strip()

        # Stop at the next recognized label.
        next_labels = re.search(
            r"\s+(?:pack\s*rate|packrate|qty|quantity|boxes?|cartons?|"
            r"price|unit\s*price|total|farm\s*code|length)\b",
            raw, re.I,
        )
        if next_labels:
            raw = raw[:next_labels.start()].strip()

        if field_name in {"boxes", "pack_rate", "quantity", "unit_price", "total"}:
            value = as_number(raw)
            if value is not None:
                result[field_name] = value
        elif field_name == "length":
            lm = re.search(r"\d+(?:\.\d+)?", raw)
            if lm:
                result["specification"] = {"length": f"{lm.group(0)}cm"}
        elif field_name == "farm_code":
            if raw.split():
                result["farm_code"] = raw.split()[0]

    return result


def strip_known_annotations(name: str) -> str:
    s = name
    patterns = [
        r"\bpack\s*rate\b\s*[:=]?\s*[\d,.]+",
        r"\bpackrate\b\s*[:=]?\s*[\d,.]+",
        r"\b(?:qty|quantity|qnty)\b\s*[:=]?\s*[\d,.]+",
        r"\b(?:boxes?|bx|cartons?|ctn)\b\s*[:=]?\s*[\d,.]+",
        r"\b(?:price|rate|unit\s*price|price\s*/\s*stem|price\s*per\s*stem)\b\s*[:=@]?\s*[\d,.]+",
        r"\b(?:total|amount)\b\s*[:=]?\s*[\d,.]+",
        r"\b\d+(?:\.\d+)?\s*cm\b",
    ]
    for pat in patterns:
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(
        r"(?i)(?<=\d)\s*(?:usd|us\$|kes|ksh|eur|gbp|aed|sar|qar|jpy)\b",
        " ", s
    )
    return re.sub(r"\s+", " ", s).strip(" -:,;.|")


def parse_inline_line(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s or is_header_or_metadata(s):
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
    if fields:
        name = re.sub(
            r"\b(?:boxes?|bx|cartons?|ctn|qty|quantity|qnty|pack\s*rate|"
            r"packrate|price|rate|total|amount|length)\b",
            " ", name, flags=re.I,
        )
        name = re.sub(r"\s+", " ", name).strip(" -:,;.|")

    if not looks_like_product(name):
        return None

    return {
        "product_name": name,
        "boxes": fields.get("boxes"),
        "pack_rate": fields.get("pack_rate"),
        "quantity": fields.get("quantity"),
        "unit_price": fields.get("unit_price"),
        "total": fields.get("total"),
        "specification": fields.get("specification", {}),
        "evidence": [{"type": "inline_line", "text": s}],
        "source_confidence": 0.78,
    }


def is_header_or_metadata(s: str) -> bool:
    if is_probable_header(s):
        return True
    low = norm(s)
    if re.match(r"^(?:invoice|inv|quote|qtn|proforma|po)[\s/#-]*[\w/-]+$", low):
        return True
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s):
        return True
    return False


# ---------------------------------------------------------------------------
# TABLE / ROW SPLITTING
# ---------------------------------------------------------------------------

def split_columns(line: str) -> List[str]:
    if "|" in line:
        parts = [x.strip() for x in re.split(r"\s*\|\s*", line)]
    elif "\t" in line:
        parts = [x.strip() for x in line.split("\t")]
    else:
        parts = [x.strip() for x in re.split(r"\s{2,}", line.strip())]
    return [x for x in parts if x]


def flexible_row_split(row: str, expected_cols: int = 8) -> List[str]:
    """
    Multi-hypothesis splitter:
    - explicit pipes/tabs
    - wide whitespace
    - right-peeling numeric suffixes
    - fallback to tokens
    """
    s = (row or "").strip()
    if not s:
        return []

    explicit = split_columns(s)
    if len(explicit) >= 2:
        return explicit

    # Right-peel numeric cells. This is deliberately bounded so product names
    # containing numbers do not get consumed indefinitely.
    cells: List[str] = []
    remaining = s
    numeric_tail = re.compile(
        r"(?:^|\s)((?:[$€£]\s*)?-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?"
        r"(?:\s*(?:KES|KSH|USD|EUR|GBP|AED|SAR|QAR|JPY))?)\s*$",
        re.I,
    )
    for _ in range(max(1, min(expected_cols - 1, 10))):
        m = numeric_tail.search(remaining)
        if not m or m.start(1) <= 0:
            break
        cells.insert(0, m.group(1).strip())
        remaining = remaining[:m.start(1)].rstrip()

    head = re.sub(r"^\d{1,4}[.)]?\s+", "", remaining).strip()
    if cells:
        return [head] + cells

    tokens = s.split()
    return tokens if len(tokens) <= expected_cols else [s]


def find_table_header(lines: Sequence[str]) -> Tuple[Optional[int], List[Optional[str]]]:
    best: Optional[Tuple[float, int, List[Optional[str]]]] = None

    # Combine consecutive short header lines too, e.g.:
    # "Flower Variety Length" / "Pack" / "Rate" / "Boxes Total Stems..."
    for i in range(len(lines)):
        combined_candidates = [lines[i]]
        if i + 1 < len(lines):
            combined_candidates.append(lines[i] + " " + lines[i + 1])
        if i + 2 < len(lines):
            combined_candidates.append(lines[i] + " " + lines[i + 1] + " " + lines[i + 2])

        for candidate in combined_candidates:
            cols = split_columns(candidate)
            if len(cols) < 2:
                # For space-separated headers, use a word-window.
                words = candidate.split()
                if len(words) >= 3:
                    cols = words

            mapped = [match_header(c) for c in cols]
            known = [m for m in mapped if m]
            distinct = set(known)
            if (
                len(distinct) >= 3
                and any(x in distinct for x in ("product", "variety", "description"))
                and any(x in distinct for x in ("quantity", "boxes", "unit_price", "total", "pack_rate"))
            ):
                score = len(distinct) * 10 + len(known)
                if best is None or score > best[0]:
                    best = (score, i, mapped)

    return (best[1], best[2]) if best else (None, [])


def value_for_field(field_name: str, raw: str) -> Any:
    if empty_field(raw):
        return None
    if field_name in {"boxes", "pack_rate", "quantity", "unit_price", "total"}:
        return as_number(raw)
    if field_name == "length":
        m = re.search(r"\d+(?:\.\d+)?", raw)
        return f"{m.group(0)}cm" if m else None
    return raw.strip()


def parse_table_cells(cells: List[str], headers: List[Optional[str]]) -> Optional[Dict[str, Any]]:
    if not cells:
        return None

    vals: Dict[str, Any] = {}
    raw_values: Dict[str, str] = {}
    evidence = []

    for i, cell in enumerate(cells):
        if i >= len(headers):
            raw_values[f"unmapped_{i+1}"] = cell
            continue
        field_name = headers[i]
        if not field_name:
            raw_values[f"unmapped_{i+1}"] = cell
            continue
        if field_name in vals and vals[field_name] not in (None, ""):
            raw_values[f"duplicate_{field_name}_{i+1}"] = cell
            continue
        vals[field_name] = value_for_field(field_name, cell)
        evidence.append({"field": field_name, "raw": cell, "column_index": i})

    name = None
    for field_name in ("product", "variety", "description"):
        candidate = vals.get(field_name)
        if candidate and looks_like_product(str(candidate)):
            name = str(candidate).strip()
            break

    if not name:
        for cell in cells:
            if looks_like_product(cell) and not is_header_or_metadata(cell):
                name = cell
                break

    if not name:
        return None

    item = {
        "product_name": name,
        "boxes": vals.get("boxes"),
        "pack_rate": vals.get("pack_rate"),
        "quantity": vals.get("quantity"),
        "unit_price": vals.get("unit_price"),
        "total": vals.get("total"),
        "specification": {},
        "evidence": evidence,
        "source_confidence": 0.84,
    }

    if vals.get("length") is not None:
        item["specification"]["length"] = vals["length"]
    if vals.get("farm_code"):
        item["farm_code"] = str(vals["farm_code"]).strip()
    if raw_values:
        item["raw_values"] = raw_values

    return item


def parse_table_lines(lines: Sequence[str]) -> List[Dict[str, Any]]:
    header_idx, headers = find_table_header(lines)
    if header_idx is None:
        return []

    items = []
    expected = max(2, len(headers))
    for line in lines[header_idx + 1:]:
        s = line.strip()
        if not s:
            continue
        if is_boilerplate(s):
            continue

        cells = split_columns(s)
        if len(cells) < 2:
            cells = flexible_row_split(s, expected)

        # Handle OCR rows with fewer cells: if a row contains enough numeric
        # evidence, try the flexible splitter even when whitespace splitting
        # produced a misleading result.
        candidates = [cells]
        alt = flexible_row_split(s, expected)
        if alt != cells:
            candidates.append(alt)

        best_item = None
        best_score = -1
        for cand in candidates:
            item = parse_table_cells(cand, headers)
            if not item:
                continue
            score = 0
            for k in ("quantity", "boxes", "pack_rate", "unit_price", "total"):
                if item.get(k) is not None:
                    score += 2
            if item.get("product_name"):
                score += 3
            if item.get("raw_values"):
                score -= 1
            if score > best_score:
                best_score, best_item = score, item

        if best_item:
            items.append(best_item)

    return items


def parse_order_text(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = clean_ocr_text(text)
    lines = [x.rstrip() for x in text.splitlines()]
    meta = extract_meta_from_text(text)

    items = parse_table_lines(lines)

    # Free-form parser is independent of table parsing.
    free_items = []
    for line in lines:
        s = line.strip()
        if not s or len(s) < 3 or is_header_or_metadata(s):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,80}\s*[:#]", s):
            if match_meta_label(re.split(r"[:#]", s, 1)[0]):
                continue
        item = parse_inline_line(s)
        if item:
            free_items.append(item)

    # If the table parser found rows, keep it as the primary source but add
    # non-duplicate free-form rows.
    if items:
        existing = {compact_norm(x["product_name"]) for x in items}
        for item in free_items:
            key = compact_norm(item["product_name"])
            if key and key not in existing:
                items.append(item)
                existing.add(key)
    else:
        items = free_items

    return clean_items(items), meta


# ---------------------------------------------------------------------------
# PDF GEOMETRY
# ---------------------------------------------------------------------------

@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block: int = 0
    line: int = 0
    word: int = 0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def pdf_words(content: bytes) -> Tuple[List[Word], str]:
    if fitz is None:
        return [], ""
    words: List[Word] = []
    text_pages: List[str] = []
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            text_pages.append(page.get_text("text") or "")
            for w in page.get_text("words"):
                if len(w) >= 5 and str(w[4]).strip():
                    words.append(
                        Word(
                            float(w[0]), float(w[1]), float(w[2]), float(w[3]),
                            str(w[4]), int(w[5]), int(w[6]), int(w[7])
                        )
                    )
        return words, "\n".join(text_pages)
    except Exception:
        logger.exception("PDF geometry extraction failed")
        return [], ""


def cluster_word_rows(words: List[Word], y_tol: float = 3.5) -> List[List[Word]]:
    rows: List[List[Word]] = []
    for word in sorted(words, key=lambda w: (w.y0, w.x0)):
        placed = False
        for row in rows[-3:]:
            cy = statistics.mean(w.y0 for w in row)
            if abs(word.y0 - cy) <= y_tol:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])
    for row in rows:
        row.sort(key=lambda w: w.x0)
    return rows


def row_text(row: Sequence[Word]) -> str:
    return " ".join(w.text for w in sorted(row, key=lambda z: z.x0))


def layout_table_parse(words: List[Word]) -> List[Dict[str, Any]]:
    """
    Geometry-aware parsing with a fallback that does not depend on detecting
    global column gaps. It is intentionally page/layout agnostic.
    """
    if not words:
        return []

    rows = cluster_word_rows(words)
    texts = [row_text(r) for r in rows]
    header_idx, headers = find_table_header(texts)
    if header_idx is None:
        return []

    # Estimate column anchors from the header. For each header word, use its
    # center; merged headers are assigned by their center range.
    header_row = rows[header_idx]
    header_words = sorted(header_row, key=lambda w: w.x0)

    anchors: List[Tuple[float, Optional[str]]] = []
    for hw in header_words:
        field_name = match_header(hw.text)
        if field_name:
            anchors.append((hw.cx, field_name))

    # If the header is split over multiple lines, use the mapped header list
    # plus the first occurrence of each semantic field.
    unique_fields: Dict[str, float] = {}
    for x, f in anchors:
        if f not in unique_fields:
            unique_fields[f] = x

    # Use expected header order if there are not enough coordinate anchors.
    if len(unique_fields) < 3:
        return parse_table_lines(texts)

    ordered = sorted(unique_fields.items(), key=lambda kv: kv[1])
    centers = [x for _, x in ordered]
    fields = [f for f, _ in ordered]

    def assign(row: List[Word]) -> List[str]:
        buckets = [[] for _ in centers]
        for w in row:
            idx = min(range(len(centers)), key=lambda j: abs(w.cx - centers[j]))
            buckets[idx].append(w.text)
        return [" ".join(b).strip() for b in buckets]

    items: List[Dict[str, Any]] = []
    for row in rows[header_idx + 1:]:
        txt = row_text(row).strip()
        if not txt or is_boilerplate(txt):
            continue

        cells = assign(row)
        if not any(cells):
            continue

        item_headers = fields
        item = parse_table_cells(cells, item_headers)
        if item:
            item["source_confidence"] = min(
                0.95, float(item.get("source_confidence", 0.84)) + 0.05
            )
            items.append(item)

    # Geometry can occasionally duplicate a row or split a wrapped product.
    return dedupe_items(items)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    max_dim = max(w, h)
    if max_dim < 1800:
        scale = min(2.5, 1800 / max_dim)
        img = img.resize((int(w * scale), int(h * scale)))

    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray


def ocr_image(img: Image.Image) -> str:
    processed = preprocess_image(img)
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 4",
    ]
    results = []
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(processed, lang="eng", config=cfg)
            if txt and re.search(r"[A-Za-z0-9]", txt):
                results.append(txt)
        except Exception:
            logger.exception("OCR configuration failed")
    if not results:
        return ""
    # Choose by a quality score, not simply longest text.
    def score(t: str) -> float:
        alnum = len(re.findall(r"[A-Za-z0-9]", t))
        lines = len([x for x in t.splitlines() if x.strip()])
        weird = len(re.findall(r"[^\w\s.,:/%$€£#()\-|]", t))
        return alnum + 2 * lines - 2 * weird

    return max(results, key=score)


def extract_text_from_image(content: bytes) -> Tuple[str, str]:
    try:
        img = Image.open(io.BytesIO(content))
        return ocr_image(img), "image_ocr"
    except Exception:
        logger.exception("Image extraction failed")
        return "", "image_error"


def extract_text_from_pdf(content: bytes) -> Tuple[str, str, List[Word]]:
    words, geometry_text = pdf_words(content)

    # Prefer a normal text layer if it is meaningful.
    text_parts = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                text_parts.append("")
    except Exception:
        logger.exception("PyPDF2 extraction failed")

    text = "\n".join(text_parts).strip()
    if len(re.sub(r"\s+", "", text)) >= 30:
        return text, "pdf_text", words

    if geometry_text and len(re.sub(r"\s+", "", geometry_text)) >= 30:
        return geometry_text, "pdf_layout_text", words

    if fitz is None:
        return text, "pdf_text_empty", words

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        ocr_parts = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_parts.append(ocr_image(img))
        ocr_text = "\n".join(ocr_parts)
        return ocr_text, "pdf_ocr", words
    except Exception:
        logger.exception("PDF OCR failed")
        return text or geometry_text, "pdf_text_fallback", words


# ---------------------------------------------------------------------------
# DOCX / SPREADSHEETS / JSON
# ---------------------------------------------------------------------------

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
        logger.exception("DOCX extraction failed")
        return ""


def find_excel_header(df: pd.DataFrame) -> Tuple[Optional[int], Dict[int, str]]:
    best: Optional[Tuple[int, int, Dict[int, str]]] = None
    for i in range(min(80, len(df))):
        row = df.iloc[i]
        mapping: Dict[int, str] = {}
        for idx, val in enumerate(row):
            if pd.isna(val):
                continue
            field_name = match_header(str(val))
            if field_name and field_name not in mapping.values():
                mapping[idx] = field_name

        fields = set(mapping.values())
        if (
            len(fields) >= 3
            and any(x in fields for x in ("product", "variety", "description"))
            and any(x in fields for x in ("quantity", "boxes", "unit_price", "total", "pack_rate"))
        ):
            score = len(fields)
            if best is None or score > best[0]:
                best = (score, i, mapping)

    return (best[1], best[2]) if best else (None, {})


def extract_from_dataframe(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if df is None or df.empty:
        return [], {}

    header_row, header_map = find_excel_header(df)
    if header_row is None:
        # Treat first non-empty column as a product column only when there are
        # multiple populated rows; no invented numeric fields.
        header_row = -1
        header_map = {0: "product"}

    items = []
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]
        vals: Dict[str, Any] = {}
        for idx, field_name in header_map.items():
            if idx < len(row) and not pd.isna(row.iloc[idx]):
                vals[field_name] = row.iloc[idx]

        if not vals:
            continue

        name = None
        for field_name in ("product", "variety", "description"):
            candidate = vals.get(field_name)
            if not empty_field(candidate) and looks_like_product(str(candidate)):
                name = str(candidate).strip()
                break
        if not name:
            continue

        item = {
            "product_name": name,
            "boxes": as_number(vals.get("boxes")),
            "pack_rate": as_number(vals.get("pack_rate")),
            "quantity": as_number(vals.get("quantity")),
            "unit_price": as_number(vals.get("unit_price")),
            "total": as_number(vals.get("total")),
            "specification": {},
            "evidence": [
                {"field": f, "raw": str(v)}
                for f, v in vals.items() if not empty_field(v)
            ],
            "source_confidence": 0.94,
        }

        if not empty_field(vals.get("farm_code")):
            item["farm_code"] = str(vals["farm_code"]).strip()
        if not empty_field(vals.get("length")):
            m = re.search(r"\d+(?:\.\d+)?", str(vals["length"]))
            if m:
                item["specification"]["length"] = f"{m.group(0)}cm"

        items.append(item)

    return clean_items(items), {}


def extract_from_excel(content: bytes, ext: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        if ext == "csv":
            raw = content.decode("utf-8-sig", errors="replace")
            try:
                df = pd.read_csv(io.StringIO(raw), header=None, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(io.StringIO(raw), header=None)
        else:
            df = pd.read_excel(io.BytesIO(content), header=None)
        return extract_from_dataframe(df)
    except Exception as e:
        logger.exception("Spreadsheet read failed")
        return [], {"error": str(e)}


def extract_text_from_json(content: bytes) -> Tuple[str, str]:
    try:
        obj = json.loads(content.decode("utf-8", errors="ignore"))
        return json.dumps(obj, ensure_ascii=False, indent=2), "json"
    except Exception:
        return content.decode("utf-8", errors="ignore"), "text"


# ---------------------------------------------------------------------------
# RECONCILIATION / CONFIDENCE
# ---------------------------------------------------------------------------

def numeric_close(a: Any, b: Any, tolerance: float = 0.01) -> bool:
    x, y = parse_number(a), parse_number(b)
    if x is None or y is None:
        return False
    return abs(x - y) <= max(tolerance, abs(y) * 0.01)


def validate_item(item: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    b, p, q, u, t = [item.get(k) for k in ("boxes", "pack_rate", "quantity", "unit_price", "total")]

    if b is not None and b <= 0:
        warnings.append("boxes_not_positive")
    if p is not None and p <= 0:
        warnings.append("pack_rate_not_positive")
    if q is not None and q <= 0:
        warnings.append("quantity_not_positive")
    if u is not None and u < 0:
        warnings.append("unit_price_negative")
    if t is not None and t < 0:
        warnings.append("total_negative")

    if q is not None and u is not None and t is not None:
        if not numeric_close(float(q) * float(u), t, 0.02):
            warnings.append("quantity_x_unit_price_does_not_match_total")

    if b is not None and p is not None and q is not None:
        if not numeric_close(float(b) * float(p), q, 0.5):
            warnings.append("boxes_x_pack_rate_does_not_match_quantity")

    return warnings


def reconcile_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe arithmetic:
    - Never overwrite an extracted value.
    - Only derive a missing value when the source relation is explicit enough.
    - Every derivation is recorded.
    """
    out = dict(item)
    derived = list(out.get("derived_fields", []))

    q, b, p, u, t = [out.get(k) for k in ("quantity", "boxes", "pack_rate", "unit_price", "total")]

    if q is None and b is not None and p is not None:
        q_calc = float(b) * float(p)
        out["quantity"] = int(q_calc) if q_calc.is_integer() else q_calc
        derived.append({
            "field": "quantity",
            "formula": "boxes * pack_rate",
            "value": out["quantity"],
        })

    if t is None and out.get("quantity") is not None and u is not None:
        t_calc = float(out["quantity"]) * float(u)
        out["total"] = round(t_calc, 4)
        derived.append({
            "field": "total",
            "formula": "quantity * unit_price",
            "value": out["total"],
        })

    if derived:
        out["derived_fields"] = derived
    return out


def item_quality(item: Dict[str, Any]) -> float:
    score = float(item.get("source_confidence", 0.65))
    if item.get("product_name"):
        score += 0.08
    populated = sum(item.get(k) is not None for k in ("boxes", "pack_rate", "quantity", "unit_price", "total"))
    score += 0.04 * populated
    if item.get("raw_values"):
        score -= 0.04
    if item.get("warnings"):
        score -= min(0.22, 0.05 * len(item["warnings"]))
    if item.get("derived_fields"):
        score -= min(0.12, 0.04 * len(item["derived_fields"]))
    return round(max(0.0, min(0.99, score)), 3)


def clean_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("product_name", "")).strip()
        if not looks_like_product(name):
            continue

        out = {
            "product_name": name,
            "boxes": as_number(raw.get("boxes")),
            "pack_rate": as_number(raw.get("pack_rate")),
            "quantity": as_number(raw.get("quantity")),
            "unit_price": as_number(raw.get("unit_price")),
            "total": as_number(raw.get("total")),
        }

        for key in ("farm_code", "product_id"):
            if not empty_field(raw.get(key)):
                out[key] = raw[key]

        spec = raw.get("specification")
        if isinstance(spec, dict) and spec:
            out["specification"] = spec

        for key in ("raw_values", "evidence", "source_confidence"):
            if raw.get(key):
                out[key] = raw[key]

        out = reconcile_item(out)
        out["warnings"] = validate_item(out)
        out["confidence"] = item_quality(out)
        if not out["warnings"]:
            out.pop("warnings", None)

        cleaned.append(out)

    return cleaned


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = compact_norm(item.get("product_name", ""))
        if not key:
            continue
        old = best.get(key)
        if old is None or item_quality(item) > item_quality(old):
            best[key] = item
    return list(best.values())


# ---------------------------------------------------------------------------
# MULTI-PASS ENGINE
# ---------------------------------------------------------------------------

def analyze_bytes(content: bytes, fname: str, ext: str, deadline: float) -> Dict[str, Any]:
    if time.monotonic() > deadline:
        return {"items": [], "metadata": {}, "text_extracted": "", "extraction_method": "timed_out"}

    ext = ext.lower().lstrip(".")
    text = ""
    method = ""
    metadata: Dict[str, Any] = {}
    candidates: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "passes": [],
        "candidate_counts": {},
    }

    # Spreadsheet fast path.
    if ext in {"xlsx", "xls", "xlsm", "csv"}:
        items, metadata = extract_from_excel(content, ext)
        diagnostics["passes"].append("spreadsheet")
        diagnostics["candidate_counts"]["spreadsheet"] = len(items)
        return {
            "items": items,
            "metadata": metadata,
            "text_extracted": "",
            "extraction_method": "spreadsheet",
            "diagnostics": diagnostics,
        }

    pdf_words_list: List[Word] = []

    if ext == "pdf":
        text, method, pdf_words_list = extract_text_from_pdf(content)
        diagnostics["passes"].append(method)

        if pdf_words_list and time.monotonic() <= deadline:
            layout_items = layout_table_parse(pdf_words_list)
            diagnostics["candidate_counts"]["pdf_layout"] = len(layout_items)
            if layout_items:
                candidates.extend(layout_items)
                diagnostics["passes"].append("pdf_geometry_table")
    elif ext in {"docx", "doc"}:
        text = extract_text_from_docx(content)
        method = "docx"
        diagnostics["passes"].append("docx")
    elif ext in {"jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff"}:
        text, method = extract_text_from_image(content)
        diagnostics["passes"].append("ocr")
    elif ext == "json":
        text, method = extract_text_from_json(content)
        diagnostics["passes"].append("json")
    else:
        text = content.decode("utf-8", errors="replace")
        method = "text"
        diagnostics["passes"].append("text")

    if text:
        metadata = extract_meta_from_text(text)

    # Pass A: explicit table reconstruction.
    if text and time.monotonic() <= deadline:
        table_items = parse_table_lines(clean_ocr_text(text).splitlines())
        diagnostics["candidate_counts"]["text_table"] = len(table_items)
        if table_items:
            candidates.extend(table_items)
            diagnostics["passes"].append("text_table")

    # Pass B: free-form extraction.
    if text and time.monotonic() <= deadline:
        free_items = []
        for line in clean_ocr_text(text).splitlines():
            if time.monotonic() > deadline:
                break
            item = parse_inline_line(line)
            if item:
                free_items.append(item)
        diagnostics["candidate_counts"]["free_form"] = len(free_items)
        if free_items:
            candidates.extend(free_items)
            diagnostics["passes"].append("free_form")

    # Pass C: conservative line recovery for unstructured pasted text.
    # Unlike the previous "never give up" patch, this pass only emits a row
    # when it has a plausible product name and numeric evidence.
    if text and time.monotonic() <= deadline and not candidates:
        recovery = []
        for line in clean_ocr_text(text).splitlines():
            s = line.strip()
            if not s or is_boilerplate(s) or is_contact_or_address(s):
                continue
            if len(re.findall(r"[A-Za-z]{2,}", s)) < 2:
                continue
            if not re.search(r"\d", s):
                continue
            item = parse_inline_line(s)
            if item:
                item["source_confidence"] = min(float(item.get("source_confidence", 0.7)), 0.72)
                recovery.append(item)
        diagnostics["candidate_counts"]["conservative_recovery"] = len(recovery)
        if recovery:
            candidates.extend(recovery)
            diagnostics["passes"].append("conservative_recovery")

    # Reconcile duplicate observations from different passes.
    items = dedupe_items(clean_items(candidates))

    if not items and text:
        diagnostics["status"] = "readable_document_but_no_high_confidence_items"
        diagnostics["review_reason"] = (
            "The engine found readable content but could not identify a product/order row "
            "with enough evidence. Nothing was invented."
        )
    elif not text:
        diagnostics["status"] = "no_readable_text"
        diagnostics["review_reason"] = "No usable text was extracted from the supplied file."
    else:
        diagnostics["status"] = "ok"

    total_boxes = safe_sum(x.get("boxes") for x in items)
    total_quantity = safe_sum(x.get("quantity") for x in items)
    total_amount = safe_sum(x.get("total") for x in items)

    warnings = sorted({
        w for x in items for w in x.get("warnings", [])
    })

    return {
        "items": items,
        "metadata": metadata,
        "text_extracted": text,
        "extraction_method": method,
        "total_boxes": safe_int(total_boxes),
        "total_quantity": safe_int(total_quantity),
        "total_amount": round(total_amount, 2),
        "item_count": len(items),
        "review_required": bool(
            diagnostics.get("status") != "ok"
            or any(x.get("confidence", 0) < 0.82 or x.get("warnings") for x in items)
        ),
        "warnings": warnings,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# PRODUCT MATCHING
# ---------------------------------------------------------------------------

def normalize_product_for_match(name: str) -> str:
    n = norm(name)
    n = re.sub(r"\b\d+(?:\.\d+)?\s*cm\b", " ", n)
    n = re.sub(r"\b(?:box|boxes|bx|carton|cartons|qty|quantity|stems?|pcs?|pieces?)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def product_similarity(a: str, b: str) -> float:
    aa, bb = normalize_product_for_match(a), normalize_product_for_match(b)
    if not aa or not bb:
        return 0.0
    scores = [
        fuzz.ratio(aa, bb),
        fuzz.token_set_ratio(aa, bb),
        fuzz.WRatio(aa, bb),
    ]
    # Strong exact normalized match.
    if compact_norm(aa) == compact_norm(bb):
        return 1.0
    return max(scores) / 100.0


class MatchRequest:
    pass


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "Smart Document Intelligence Engine",
        "version": ENGINE_VERSION,
        "status": "operational",
        "design": "multi-pass, evidence-aware, conservative AI-like extraction",
        "capabilities": sorted(ALLOWED_EXTENSIONS),
        "endpoints": [
            "/api/ping",
            "/api/health",
            "/api/analyze",
            "/api/extract-text",
            "/api/match-products",
        ],
    }


@app.get("/api/ping")
async def ping():
    return {
        "ok": True,
        "status": "awake",
        "version": ENGINE_VERSION,
        "time": time.time(),
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "smart-document-engine",
        "version": ENGINE_VERSION,
        "ocr_available": bool(pytesseract),
        "scanned_pdf_ocr_available": fitz is not None,
        "spreadsheet_available": True,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


async def read_upload(file: UploadFile) -> Tuple[bytes, str, str]:
    fname = file.filename or "upload"
    ext = Path(fname).suffix.lstrip(".").lower()

    if ext not in ALLOWED_EXTENSIONS:
        # Allow an explicitly supplied content type only for text-like uploads.
        ctype = (file.content_type or "").lower()
        if ctype.startswith("text/"):
            ext = "text"
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: .{ext or 'unknown'}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_MB} MB."
        )

    return content, fname, ext


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    company_id: int = Form(0),
    file_type: Optional[str] = Form(None),
):
    started = now_ms()
    content, fname, ext = await read_upload(file)
    if file_type:
        supplied = file_type.lower().lstrip(".")
        if supplied in ALLOWED_EXTENSIONS:
            ext = supplied

    try:
        result = analyze_bytes(
            content,
            fname,
            ext,
            time.monotonic() + ANALYZE_TIMEOUT_SECONDS,
        )
        result.update({
            "success": True,
            "filename": fname,
            "file_type": ext,
            "company_id": company_id,
            "engine_version": ENGINE_VERSION,
            "processing_ms": now_ms() - started,
            # Keep the response reasonably small while retaining text for UI review.
            "text_extracted": (result.get("text_extracted") or "")[:20000],
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze failed")
        raise HTTPException(
            status_code=500,
            detail="Document analysis failed safely. Check engine logs for details.",
        ) from e


@app.post("/api/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    content, fname, ext = await read_upload(file)
    try:
        if ext == "pdf":
            text, method, _ = extract_text_from_pdf(content)
        elif ext in {"docx", "doc"}:
            text, method = extract_text_from_docx(content), "docx"
        elif ext in {"jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff"}:
            text, method = extract_text_from_image(content)
        elif ext == "json":
            text, method = extract_text_from_json(content)
        elif ext in {"xlsx", "xls", "xlsm", "csv"}:
            items, _ = extract_from_excel(content, ext)
            text = "\n".join(
                " | ".join(str(x.get(k) or "") for k in
                           ("product_name", "boxes", "pack_rate", "quantity", "unit_price", "total"))
                for x in items
            )
            method = "spreadsheet"
        else:
            text, method = content.decode("utf-8", errors="replace"), "text"

        return {
            "success": True,
            "filename": fname,
            "text": text,
            "length": len(text),
            "file_type": ext,
            "extraction_method": method,
            "engine_version": ENGINE_VERSION,
        }
    except Exception as e:
        logger.exception("extract-text failed")
        raise HTTPException(status_code=500, detail="Text extraction failed safely.") from e


@app.post("/api/match-products")
async def match_products(payload: Dict[str, Any]):
    """
    Expected:
    {
      "items": [{"product_name": "Hyndrangea pink 50cm"}],
      "company_products": [
        {"id": 1, "name": "Hydrangea Pink", "aliases": ["Hyndrangea pink"]}
      ]
    }
    """
    try:
        items = payload.get("items") or []
        company_products = payload.get("company_products") or []
        if not isinstance(items, list) or not isinstance(company_products, list):
            raise HTTPException(status_code=400, detail="items and company_products must be arrays")

        output = []
        for item in items:
            iname = str(item.get("product_name", ""))
            ranked = []

            for product in company_products:
                names = [str(product.get("name", ""))]
                aliases = product.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(str(x) for x in aliases)

                best_for_product = 0.0
                best_alias = None
                for candidate in names:
                    score = product_similarity(iname, candidate)
                    if score > best_for_product:
                        best_for_product = score
                        best_alias = candidate

                ranked.append((best_for_product, product, best_alias))

            ranked.sort(key=lambda x: x[0], reverse=True)
            top = ranked[0] if ranked else (0.0, None, None)
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            score, product, alias = top

            # Margin prevents a fuzzy near-tie from being silently selected.
            confident = (
                product is not None
                and score >= MIN_PRODUCT_MATCH
                and (score - second >= 0.05 or score >= 0.97)
            )

            if confident:
                output.append({
                    **item,
                    "product_id": product.get("id"),
                    "matched_product_name": product.get("name"),
                    "matched_alias": alias,
                    "match_confidence": round(score, 3),
                    "match_status": "matched",
                })
            else:
                output.append({
                    **item,
                    "product_id": None,
                    "matched_product_name": None,
                    "match_confidence": round(score, 3),
                    "match_status": "review_required",
                    "match_reason": "ambiguous_or_below_threshold",
                })

        return {
            "success": True,
            "items": output,
            "matched_count": sum(x["match_status"] == "matched" for x in output),
            "review_count": sum(x["match_status"] == "review_required" for x in output),
            "engine_version": ENGINE_VERSION,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Product matching failed")
        raise HTTPException(status_code=500, detail="Product matching failed safely.") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
