"""
Smart Document Intelligence Engine v6.1
For invoices, quotations, proformas, receipts, delivery notes and flower/export
orders. Confidence-aware extraction from text, OCR images, PDFs, DOCX,
Excel/CSV and JSON.

Key principles:
1. Never invent missing boxes, pack_rate, quantity or price.
2. Keep boxes, pack_rate, quantity, unit_price and total as separate fields.
3. Prefer explicit labels over positional guesses.
4. Ambiguous values are preserved in raw_values instead of misfiled.
5. Every item carries confidence + validation warnings.

ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
import os
import re
import json
import logging
import math
import time
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
    import openpyxl  # noqa: F401
except Exception:
    openpyxl = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-document-engine")

TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

ENGINE_VERSION = "11.1.0"

app = FastAPI(
    title="Smart Document Intelligence Engine",
    version=ENGINE_VERSION,
    description="Confidence-aware document and flower-order extraction engine.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MIN_MATCH_CONFIDENCE = float(os.getenv("MIN_MATCH_CONFIDENCE", "0.84"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "30"))
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "20"))


class MatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    company_products: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
#  SAFE MATH HELPERS
# ---------------------------------------------------------------------------

def safe_int(x: Any) -> Any:
    """Return int if x is a whole number; otherwise return x unchanged.
    Handles int, float, None safely (never raises)."""
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return x
        if x.is_integer():
            return int(x)
        return x
    # Try coercion from string-like
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except Exception:
        return x


def safe_sum(values) -> float:
    """Sum safely; returns 0.0 for empty input."""
    total = 0.0
    for v in values:
        if v is None:
            continue
        try:
            total += float(v)
        except Exception:
            continue
    return total


# ---------------------------------------------------------------------------
#  FIELD KNOWLEDGE
# ---------------------------------------------------------------------------

COLUMN_SYNONYMS = {
    "product": [
        "product", "product name", "product/service", "product / service",
        "product or service", "item", "item name", "service", "article",
        "articles", "commodity", "goods", "stock item", "particulars",
        "product description", "item description",
    ],
    "variety": [
        "variety", "flower", "flower name", "flower type", "species",
        "cultivar", "kind", "variety name",
    ],
    "description": [
        "description", "desc", "details", "item details", "specification",
        "specifications", "remarks", "product details",
    ],
    "farm_code": [
        "farm code", "farmcode", "farm reference", "farm ref",
        "supplier code", "grower code", "grower reference",
    ],
    "boxes": [
        "boxes", "box", "bx", "cartons", "carton", "ctn", "cases", "case",
        "bundles", "bundle", "packages", "pkg",
        "number of boxes", "no. of boxes", "no boxes", "box qty",
        "carton qty", "cartons qty",
    ],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box", "quantity per box",
        "stems per carton", "qty/carton", "quantity/carton",
        "conversion rate", "pack rate per box",
    ],
    "quantity": [
        "quantity", "qty", "qnty", "stems", "pcs", "pieces", "count",
        "total quantity", "total qty", "number of stems", "stem quantity",
        "invoice quantity", "qty invoice", "quantity invoice",
    ],
    "unit_price": [
        "price", "price per stem", "price/stem", "unit price",
        "unit price (usd)", "unit price(usd)", "cost", "price per unit",
        "per stem", "per piece", "amount per stem", "unit cost",
        "priceperstem", "rate per stem", "selling price",
        "unit selling price",
    ],
    "total": [
        "total", "total price", "total amount", "line total", "line amount",
        "sub-total", "subtotal", "extended price", "total (usd)",
        "total(usd)", "line value",
    ],
    "length": [
        "length", "length(cm)", "length (cm)", "size", "size(cm)",
        "stem length", "height", "stem size", "length cm",
    ],
    "discount": ["discount", "disc.", "rebate"],
    "tax": ["tax", "vat", "gst", "sales tax"],
}

META_LABELS = {
    "invoice_number": [
        "invoice number", "invoice no", "invoice #", "invoice no.", "inv no",
        "inv #", "quotation number", "quotation no", "quote number",
        "proforma number", "proforma no", "document number", "doc no",
        "reference", "ref no", "reference number", "document ref",
    ],
    "date": [
        "date", "date of shipment", "shipment date", "invoice date",
        "issue date", "document date", "quotation date",
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
        "order number", "purchase order no.",
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
    "purchase_order": ["purchase order", "purchase order form"],
}

CURRENCY_WORDS = {
    "usd": "USD", "us dollar": "USD", "dollar": "USD",
    "kes": "KES", "ksh": "KES", "kenya shilling": "KES",
    "eur": "EUR", "euro": "EUR", "gbp": "GBP", "pound": "GBP",
    "aed": "AED", "sar": "SAR", "qar": "QAR",
}

FIELD_PATTERNS = {
    "boxes": r"(?:no\.?\s*of\s*)?(?:boxes?|bx|cartons?|ctn|cases?|bundles?|packages?)",
    "pack_rate": r"(?:pack\s*rate|packrate|stems?\s*(?:per|/)\s*(?:box|carton)|qty\s*(?:per|/)\s*(?:box|carton)|quantity\s*(?:per|/)\s*(?:box|carton))",
    "quantity": r"(?:quantity|qty|qnty|total\s+qty|total\s+quantity|invoice\s+qty|invoice\s+quantity|stems?|pieces?|pcs|units?)",
    "unit_price": r"(?:price\s*(?:per|/)\s*(?:stem|piece|unit)|price\s*per\s*stem|price/stem|unit\s*price|unit\s*cost|rate\s*per\s*stem|selling\s*price|cost\s*per\s*unit)",
    "total": r"(?:line\s+total|total\s+amount|total\s+price|line\s+amount|extended\s+price|amount)",
    "length": r"(?:length(?:\s*\(?(?:cm|cms)\)?)?|stem\s+length|size)\b",
    "farm_code": r"(?:farm\s*code|farm\s*ref(?:erence)?|grower\s*code|supplier\s*code)",
}


# ---------------------------------------------------------------------------
#  NORMALIZATION / NUMBERS
# ---------------------------------------------------------------------------

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
    if value is None:
        return None
    if isinstance(value, bool):
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
    txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt)
    txt = txt.strip()
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


def as_number(value: Any) -> Optional[float]:
    n = parse_number(value)
    if n is None:
        return None
    return int(n) if float(n).is_integer() else n


def empty_field(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip().lower()
    return s == "" or s in {"n/a", "na", "null", "none", "-", "—"}


# ---------------------------------------------------------------------------
#  LABEL MATCHING
# ---------------------------------------------------------------------------

def match_header(label: str) -> Optional[str]:
    l = norm(label)
    l = re.sub(r"[*:.#]+$", "", l).strip()
    if not l:
        return None

    for field, names in COLUMN_SYNONYMS.items():
        if l in {norm(x) for x in names}:
            return field

    priority = [
        "pack_rate", "unit_price", "farm_code", "quantity", "boxes",
        "length", "total", "product", "variety", "description",
    ]
    for field in priority:
        for n in COLUMN_SYNONYMS[field]:
            nn = norm(n)
            if len(nn) >= 5 and (nn in l or l in nn):
                return field

    best_field, best_score = None, 0
    for field, names in COLUMN_SYNONYMS.items():
        for n in names:
            score = fuzz.token_set_ratio(l, norm(n))
            if score > best_score:
                best_score, best_field = score, field
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
            score = fuzz.token_set_ratio(key, norm(n))
            if score > best_score:
                best_score, best_meta = score, meta
    return best_meta if best_score >= 91 else None


# ---------------------------------------------------------------------------
#  DOCUMENT CLASSIFICATION / METADATA
# ---------------------------------------------------------------------------

def detect_document_type(text: str) -> Optional[str]:
    low = norm(text[:12000])
    scores = {}
    for typ, words in DOC_TYPES.items():
        scores[typ] = max(
            [fuzz.partial_ratio(low, norm(w)) for w in words] or [0]
        )
    typ, score = max(scores.items(), key=lambda x: x[1])
    return typ if score >= 70 else None


def extract_meta_from_text(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    lines = [x.strip() for x in clean_ocr_text(text).splitlines() if x.strip()]

    for i, line in enumerate(lines):
        m = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,70}?)\s*"
            r"(?:[:#]\s*|-\s+|\|\s*)(.*?)\s*$",
            line,
        )
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

    meta["document_type"] = detect_document_type(text)
    return meta


# ---------------------------------------------------------------------------
#  PRODUCT SAFETY
# ---------------------------------------------------------------------------

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

COMPANY_TERMS = re.compile(
    r"\b(limited|ltd\.?|llc|inc\.?|plc|company|enterprises?|"
    r"investment|trading|holdings?)\b", re.I
)

ADDRESS_TERMS = re.compile(
    r"\b(street|st\.|road|rd\.|avenue|ave\.|building|bldg|floor|"
    r"suite|tower|plaza|po box|postal code|p\.o\.)\b", re.I
)


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


# ---------------------------------------------------------------------------
#  FREE-FORM LINE PARSING
# ---------------------------------------------------------------------------

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

    for field, label_pat in patterns:
        m = re.search(
            rf"\b{label_pat}\b\s*(?:[:=]\s*|\s+)([^|,;]+)",
            line, re.I,
        )
        if not m:
            continue

        raw = m.group(1).strip()
        next_labels = re.search(
            r"\s+(?:pack\s*rate|packrate|qty|quantity|boxes?|cartons?|"
            r"price|unit\s*price|total|farm\s*code|length)\b",
            raw, re.I,
        )
        if next_labels:
            raw = raw[: next_labels.start()].strip()

        if field in {"boxes", "pack_rate", "quantity", "unit_price", "total"}:
            value = parse_number(raw)
            if value is not None:
                result[field] = as_number(value)
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

    s = re.sub(r"(?i)(?<=\d)\s*(?:usd|us\$|kes|ksh|eur|gbp|aed|sar|qar)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -:,;.|")
    return s


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

    item = {
        "product_name": name,
        "boxes": fields.get("boxes"),
        "pack_rate": fields.get("pack_rate"),
        "quantity": fields.get("quantity"),
        "unit_price": fields.get("unit_price"),
        "total": fields.get("total"),
        "specification": fields.get("specification", {}),
    }
    if fields.get("farm_code"):
        item["farm_code"] = fields["farm_code"]
    return item


# ---------------------------------------------------------------------------
#  TABLE PARSING
# ---------------------------------------------------------------------------

def split_columns(line: str) -> List[str]:
    """Split visible table columns without assuming a clean table layout."""
    line = str(line or "").strip()
    if not line:
        return []
    if "|" in line:
        return [x.strip() for x in re.split(r"\s*\|\s*", line) if x.strip()]
    if "\t" in line:
        return [x.strip() for x in line.split("\t") if x.strip()]
    # 2+ spaces is a strong signal for a text-extracted table.
    parts = [x.strip() for x in re.split(r"\s{2,}", line) if x.strip()]
    return parts if len(parts) >= 2 else [line]


def find_table_header(lines: List[str]) -> Tuple[Optional[int], List[Optional[str]]]:
    """Find a header using multiple adjacent lines, not just one physical line."""
    best = None
    limit = min(len(lines), 120)
    for i in range(limit):
        candidates = []
        # A PDF/OCR header can be split across several physical lines.
        for span in (1, 2, 3, 4):
            if i + span > limit:
                continue
            block = [x.strip() for x in lines[i:i+span] if x.strip()]
            if not block:
                continue
            if span == 1:
                cols = split_columns(block[0])
            else:
                # First try explicit separators; otherwise tokenize header words.
                joined = " | ".join(block)
                cols = split_columns(joined)
                if len(cols) < 3:
                    cols = re.split(r"\s{2,}|\s*\|\s*", " ".join(block))
                    cols = [x.strip() for x in cols if x.strip()]
            mapped = [match_header(c) for c in cols]
            known = [m for m in mapped if m]
            unique = set(known)
            if (len(unique) >= 3 and
                any(x in unique for x in ("product", "variety", "description")) and
                any(x in unique for x in ("quantity", "boxes", "unit_price", "total", "pack_rate"))):
                score = len(unique) * 10 + len(known)
                if best is None or score > best[0]:
                    best = (score, i, mapped, span)
    if best:
        return best[1], best[2]
    return None, []


def value_for_field(field: str, raw: str) -> Any:
    if empty_field(raw):
        return None
    if field in {"boxes", "pack_rate", "quantity", "unit_price", "total"}:
        return as_number(parse_number(raw))
    if field == "length":
        m = re.search(r"\d+(?:\.\d+)?", raw)
        return f"{m.group(0)}cm" if m else None
    return raw.strip()


def _numeric_tokens(text: str) -> List[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", text)]


def parse_table_row(line: str, headers: List[Optional[str]]) -> Optional[Dict[str, Any]]:
    cols = split_columns(line)
    if len(cols) < 2:
        # Space-flattened PDF rows: retain the entire line and let the fallback parser try it.
        return parse_inline_line(line)

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
        return parse_inline_line(line)

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
    if raw_values:
        item["raw_values"] = raw_values
    return item


def _looks_like_data_row(s: str) -> bool:
    nums = _numeric_tokens(s)
    return looks_like_product(strip_known_annotations(s)) and len(nums) >= 2


def parse_order_text(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Multi-pass parser for clean tables, broken tables and free-form text."""
    text = clean_ocr_text(text)
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    meta = extract_meta_from_text(text)
    candidates: List[Dict[str, Any]] = []

    header_idx, headers = find_table_header(lines)
    if header_idx is not None:
        # Try the header-defined table first.
        for line in lines[header_idx + 1:]:
            if is_header_or_metadata(line):
                continue
            item = parse_table_row(line, headers)
            if item:
                candidates.append(item)
        candidates = clean_items(candidates)
        if candidates:
            return candidates, meta

    # Pass 2: line-by-line labelled/free-form extraction.
    for line in lines:
        if is_header_or_metadata(line):
            continue
        item = parse_inline_line(line)
        if item:
            candidates.append(item)

    # Pass 3: OCR/PDF flattening can put a row into a single line with weak spacing.
    # Parse only lines containing a plausible product plus at least two numeric tokens.
    if not candidates:
        for line in lines:
            if _looks_like_data_row(line):
                item = parse_inline_line(line)
                if item:
                    candidates.append(item)

    return clean_items(candidates), meta


# ---------------------------------------------------------------------------
#  EXCEL / CSV
# ---------------------------------------------------------------------------

def find_excel_header(df: pd.DataFrame) -> Tuple[Optional[int], Dict[int, str]]:
    best = None
    for i in range(min(50, len(df))):
        row = df.iloc[i]
        mapping: Dict[int, str] = {}
        for idx, val in enumerate(row):
            if pd.isna(val):
                continue
            field = match_header(str(val))
            if field and field not in mapping.values():
                mapping[idx] = field

        fields = set(mapping.values())
        score = len(fields)
        if (
            score >= 2
            and any(x in fields for x in ("product", "variety", "description"))
            and any(x in fields for x in ("quantity", "boxes", "unit_price", "total", "pack_rate"))
        ):
            if best is None or score > best[0]:
                best = (score, i, mapping)

    return (best[1], best[2]) if best else (None, {})


def extract_from_dataframe(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if df is None or df.empty:
        return [], {}

    header_row, header_map = find_excel_header(df)
    if header_row is None:
        header_row = -1
        header_map = {0: "product"}

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
            candidate = vals.get(f)
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
        }

        if not empty_field(vals.get("farm_code")):
            item["farm_code"] = str(vals["farm_code"]).strip()
        if not empty_field(vals.get("length")):
            lv = str(vals["length"]).strip()
            m = re.search(r"\d+(?:\.\d+)?", lv)
            if m:
                item["specification"]["length"] = f"{m.group(0)}cm"

        items.append(item)

    return clean_items(items), {}


def extract_from_excel(content: bytes, ext: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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


# ---------------------------------------------------------------------------
#  DOCUMENT EXTRACTION
# ---------------------------------------------------------------------------

def _pdf_text_fitz(content: bytes) -> Tuple[str, int, int]:
    if fitz is None:
        return "", 0, 0
    doc = fitz.open(stream=content, filetype="pdf")
    parts = []
    text_pages = 0
    total_pages = len(doc)
    for page in doc:
        # "text" preserves line breaks better than PyPDF2 for invoices.
        txt = page.get_text("text", sort=True) or ""
        if len(re.sub(r"\s+", "", txt)) >= 20:
            text_pages += 1
        if txt.strip():
            parts.append(txt)
    doc.close()
    return "\n".join(parts).strip(), total_pages, text_pages


def _pdf_ocr_missing_pages(content: bytes, max_pages: int = MAX_OCR_PAGES) -> Tuple[str, int]:
    if fitz is None or not pytesseract:
        return "", 0
    doc = fitz.open(stream=content, filetype="pdf")
    parts = []
    processed = 0
    try:
        for idx, page in enumerate(doc):
            if idx >= min(len(doc), MAX_PDF_PAGES) or processed >= max_pages:
                break
            existing = page.get_text("text", sort=True) or ""
            # OCR only pages that do not have enough searchable text.
            if len(re.sub(r"\s+", "", existing)) >= 20:
                continue
            pix = page.get_pixmap(dpi=OCR_DPI, alpha=False, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            txt = ocr_image(img)
            if txt.strip():
                parts.append(txt)
            processed += 1
    finally:
        doc.close()
    return "\n".join(parts), processed


def extract_text_from_pdf(content: bytes) -> Tuple[str, str]:
    """Fast path first; OCR is targeted only at pages that need it."""
    # PyMuPDF is the preferred extractor because it is usually much faster and
    # retains line ordering/spacing that invoice parsers need.
    try:
        fitz_text, page_count, text_pages = _pdf_text_fitz(content)
        if fitz_text:
            # If searchable text exists, return immediately. This avoids the
            # 3-pass OCR that caused the 60s request failure on normal PDFs.
            if text_pages == page_count or len(re.sub(r"\s+", "", fitz_text)) >= 80:
                return fitz_text, "pdf_text_fitz"
    except Exception as e:
        logger.warning("PyMuPDF text extraction failed: %s", e)
        fitz_text, page_count, text_pages = "", 0, 0

    # Secondary reader for PDFs PyMuPDF cannot decode cleanly.
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts).strip()
        if len(re.sub(r"\s+", "", text)) >= 30:
            return text, "pdf_text_pypdf2"
    except Exception as e:
        logger.warning("PyPDF2 failed: %s", e)

    # Targeted OCR fallback for image/scanned pages only.
    try:
        ocr_text, processed = _pdf_ocr_missing_pages(content)
        combined = "\n".join(x for x in (fitz_text, ocr_text) if x).strip()
        if combined:
            method = "pdf_text_plus_targeted_ocr" if fitz_text else "pdf_ocr"
            return combined, method
    except Exception as e:
        logger.exception("Targeted PDF OCR failed: %s", e)

    return fitz_text or "", "pdf_text_empty"


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
    except Exception as e:
        logger.exception("DOCX failed")
        return ""


def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest < 1800:
        scale = min(2.0, 1800 / longest)
        img = img.resize((int(w * scale), int(h * scale)))
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.6)
    gray = ImageEnhance.Sharpness(gray).enhance(1.25)
    return gray


def ocr_image(img: Image.Image) -> str:
    """Fast OCR first, alternate layout mode only when the first pass is weak."""
    processed = preprocess_image(img)
    results = []
    for cfg in ("--oem 3 --psm 6", "--oem 3 --psm 11"):
        try:
            txt = pytesseract.image_to_string(processed, lang="eng", config=cfg)
            if txt and len(re.findall(r"[A-Za-z0-9]", txt)) >= 10:
                results.append(txt)
                if len(re.findall(r"[A-Za-z0-9]", txt)) >= 40:
                    break
        except Exception as e:
            logger.warning("OCR config failed: %s", e)
    return max(results, key=lambda x: len(re.findall(r"[A-Za-z0-9]", x))) if results else ""


def extract_text_from_image(content: bytes) -> Tuple[str, str]:
    try:
        img = Image.open(io.BytesIO(content))
        return ocr_image(img), "image_ocr"
    except Exception as e:
        logger.exception("Image extraction failed")
        return "", "image_error"


def extract_text_from_json(content: bytes) -> Tuple[str, str]:
    try:
        obj = json.loads(content.decode("utf-8", errors="ignore"))
        return json.dumps(obj, ensure_ascii=False, indent=2), "json"
    except Exception:
        return content.decode("utf-8", errors="ignore"), "text"


# ---------------------------------------------------------------------------
#  VALIDATION / CLEANING
# ---------------------------------------------------------------------------

def validate_item(item: Dict[str, Any]) -> List[str]:
    warnings = []
    b = item.get("boxes")
    p = item.get("pack_rate")
    q = item.get("quantity")
    u = item.get("unit_price")
    t = item.get("total")

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
        try:
            expected = float(q) * float(u)
            if abs(expected - float(t)) > max(0.02, abs(float(t)) * 0.01):
                warnings.append("quantity_x_unit_price_does_not_match_total")
        except Exception:
            pass

    if b is not None and p is not None and q is not None:
        try:
            expected_q = float(b) * float(p)
            if abs(expected_q - float(q)) > 0.5:
                warnings.append("boxes_x_pack_rate_does_not_match_quantity")
        except Exception:
            pass

    return warnings


def clean_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Missing values remain None. Nothing invented."""
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

        spec = raw.get("specification")
        if isinstance(spec, dict) and spec:
            out["specification"] = spec

        for key in ("farm_code", "product_id"):
            if not empty_field(raw.get(key)):
                out[key] = raw[key]

        if raw.get("raw_values"):
            out["raw_values"] = raw["raw_values"]

        # Safe derivation only when unambiguous
        if out["quantity"] is None and out["boxes"] is not None and out["pack_rate"] is not None:
            try:
                out["quantity"] = as_number(float(out["boxes"]) * float(out["pack_rate"]))
            except Exception:
                pass

        if out["total"] is None and out["quantity"] is not None and out["unit_price"] is not None:
            try:
                out["total"] = round(float(out["quantity"]) * float(out["unit_price"]), 4)
            except Exception:
                pass

        confidence = 0.95
        if out["product_name"]:
            confidence += 0.02
        if out["boxes"] is None:
            confidence -= 0.02
        if out["pack_rate"] is None:
            confidence -= 0.02
        if out["quantity"] is None:
            confidence -= 0.04
        if out["unit_price"] is None:
            confidence -= 0.04
        if out.get("raw_values"):
            confidence -= 0.10

        warnings = validate_item(out)
        confidence -= min(0.30, 0.08 * len(warnings))

        out["confidence"] = round(max(0.0, min(1.0, confidence)), 3)
        if warnings:
            out["warnings"] = warnings

        cleaned.append(out)

    return cleaned


# ---------------------------------------------------------------------------
#  PRODUCT MATCHING
# ---------------------------------------------------------------------------

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
                        score = max(fuzz.ratio(iname, cname), fuzz.token_set_ratio(iname, cname), fuzz.WRatio(iname, cname))
                    ranked.append((score, product, candidate))
            ranked.sort(key=lambda x: x[0], reverse=True)
            best = ranked[0] if ranked else (0, None, "")
            second = ranked[1][0] if len(ranked) > 1 else 0
            confidence = best[0] / 100.0
            # High similarity alone is not enough when two company products are close.
            margin_ok = (best[0] - second) >= 6.0 or best[0] >= 98.0
            accepted = best[1] is not None and confidence >= MIN_MATCH_CONFIDENCE and margin_ok
            record = {**item, "product_id": best[1].get("id") if accepted else None,
                      "matched_product_name": best[1].get("name") if accepted else None,
                      "match_confidence": round(confidence, 3),
                      "match_status": "matched" if accepted else "review_required"}
            if not margin_ok and best[1] is not None:
                record["match_reason"] = "top_candidates_too_close"
                record["match_alternatives"] = [
                    {"name": x[1].get("name"), "score": round(x[0] / 100, 3)}
                    for x in ranked[:3]
                ]
            out.append(record)
        return {
            "success": True,
            "items": out,
            "matched_count": sum(x.get("match_status") == "matched" for x in out),
            "review_count": sum(x.get("match_status") == "review_required" for x in out),
        }
    except Exception as e:
        logger.exception("Product matching failed")
        raise HTTPException(status_code=500, detail="Product matching failed")


# ---------------------------------------------------------------------------
#  ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/api/ping")
async def ping():
    return {"ok": True, "status": "ready", "version": ENGINE_VERSION, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


@app.get("/")
async def root():
    return {
        "service": "Smart Document Intelligence Engine",
        "version": ENGINE_VERSION,
        "status": "operational",
        "capabilities": [
            "invoice", "quotation", "proforma", "receipt", "delivery_note",
            "purchase_order", "images_ocr", "scanned_pdf_ocr",
            "pdf_text", "docx", "xlsx", "xls", "xlsm", "csv", "json", "text",
            "confidence_scoring", "validation", "provenance",
            "safe_blank_fields", "product_matching",
        ],
        "endpoints": [
            "/api/health",
            "/api/analyze (POST)",
            "/api/match-products (POST)",
            "/api/extract-text (POST)",
        ],
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "ready": True,
        "service": "smart-import-engine",
        "version": ENGINE_VERSION,
        "max_upload_mb": MAX_UPLOAD_MB,
        "ocr_available": bool(pytesseract),
        "scanned_pdf_ocr_available": fitz is not None,
    }


def _analyze_bytes(content: bytes, fname: str, ext: str, company_id: int) -> Dict[str, Any]:
    started = time.perf_counter()
    items: List[Dict[str, Any]] = []
    text_extracted = ""
    extraction_method = ""
    metadata: Dict[str, Any] = {}
    errors: List[str] = []

    if ext in ("xlsx", "xls", "xlsm", "csv"):
        items, metadata = extract_from_excel(content, ext)
        extraction_method = "spreadsheet"
    elif ext == "pdf":
        text_extracted, extraction_method = extract_text_from_pdf(content)
        items, metadata = parse_order_text(text_extracted)
    elif ext in ("docx", "doc"):
        text_extracted = extract_text_from_docx(content)
        extraction_method = "docx"
        items, metadata = parse_order_text(text_extracted)
    elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
        text_extracted, extraction_method = extract_text_from_image(content)
        items, metadata = parse_order_text(text_extracted)
    elif ext == "json":
        text_extracted, extraction_method = extract_text_from_json(content)
        items, metadata = parse_order_text(text_extracted)
    else:
        text_extracted = content.decode("utf-8", errors="ignore")
        extraction_method = "text"
        items, metadata = parse_order_text(text_extracted)

    cleaned = clean_items(items)
    if text_extracted and not metadata:
        metadata = extract_meta_from_text(text_extracted)

    # If a table parser found nothing, do one deliberately conservative second
    # pass over the extracted text. Never manufacture financial values.
    if not cleaned and text_extracted:
        retry_items, _ = parse_order_text("\n".join(text_extracted.splitlines()))
        cleaned = clean_items(retry_items)

    total_boxes_f = safe_sum(x.get("boxes") for x in cleaned)
    total_qty_f = safe_sum(x.get("quantity") for x in cleaned)
    total_amount_f = safe_sum(x.get("total") for x in cleaned)
    warnings = []
    for x in cleaned:
        warnings.extend(x.get("warnings", []))

    if not text_extracted:
        warnings.append("no_text_extracted")
    if not cleaned and text_extracted:
        warnings.append("no_items_detected_review_document")

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "success": True,
        "items": cleaned,
        "metadata": metadata,
        "document_type": metadata.get("document_type"),
        "total_boxes": safe_int(total_boxes_f),
        "total_quantity": safe_int(total_qty_f),
        "total_amount": round(float(total_amount_f), 2),
        "item_count": len(cleaned),
        "review_required": (not cleaned) or any(x.get("confidence", 0) < 0.80 or x.get("warnings") for x in cleaned),
        "warnings": sorted(set(warnings)),
        "text_extracted": text_extracted[:20000] if text_extracted else "",
        "extraction_method": extraction_method,
        "file_type": ext,
        "filename": fname,
        "engine_version": ENGINE_VERSION,
        "diagnostics": {
            "elapsed_seconds": elapsed,
            "input_bytes": len(content),
            "item_parser": "multi_pass_v11",
            "financial_values_invented": False,
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
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    fname = file.filename or "upload"
    ext = (file_type or Path(fname).suffix.lstrip(".")).lower()
    allowed = {"xlsx", "xls", "xlsm", "csv", "pdf", "docx", "doc", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "json", "txt", "text"}
    if ext not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext or 'unknown'}")

    logger.info("Processing %s (%s), company=%s, size=%s", fname, ext, company_id, len(content))
    try:
        # CPU-heavy parsing/OCR runs off the event loop. This keeps health/ping
        # responsive while another request is being processed.
        loop = __import__("asyncio").get_running_loop()
        result = await loop.run_in_executor(None, _analyze_bytes, content, fname, ext, company_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail="Document analysis failed. The document was not modified or fabricated.")


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

        return {
            "success": True,
            "text": text,
            "length": len(text),
            "file_type": ext,
            "extraction_method": method,
        }
    except Exception as e:
        logger.exception("extract-text failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
