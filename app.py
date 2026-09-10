"""
Smart Document Intelligence Engine v6.0
For invoices, quotations, proformas, receipts, delivery notes and flower/export
orders. Designed to extract structured data from text, OCR images, PDFs, DOCX,
Excel/CSV and JSON without silently inventing missing values.

Key principles:
1. Never use defaults for missing boxes, pack rate, quantity or price.
2. Keep boxes, pack_rate, quantity, unit_price and total as separate fields.
3. Prefer explicit labels over positional guesses.
4. Product names are cleaned only when a numeric token is confidently attached
   to a known field.
5. Ambiguous/unlabelled values are preserved in raw_values instead of being
   put into the wrong field.
6. Every item can carry confidence and source/provenance information.
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
import mimetypes

import pandas as pd
import PyPDF2
import docx
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
from rapidfuzz import fuzz

# Optional scanned-PDF OCR dependencies.
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

app = FastAPI(
    title="Smart Document Intelligence Engine",
    version="6.0.0",
    description="Confidence-aware document and flower-order extraction engine."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production to your frontend domain.
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MIN_MATCH_CONFIDENCE = float(os.getenv("MIN_MATCH_CONFIDENCE", "0.60"))


class MatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    company_products: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# FIELD KNOWLEDGE
# ---------------------------------------------------------------------------

COLUMN_SYNONYMS = {
    "product": [
        "product", "product name", "product/service", "product / service",
        "product or service", "item", "item name", "service", "article",
        "articles", "commodity", "goods", "stock item", "particulars",
        "product description", "item description", "name"
    ],
    "variety": [
        "variety", "flower", "flower name", "flower type", "species",
        "cultivar", "type", "kind", "variety name"
    ],
    "description": [
        "description", "desc", "details", "item details", "specification",
        "specifications", "remarks", "product details", "item details"
    ],
    "farm_code": [
        "farm code", "farmcode", "farm reference", "farm ref", "supplier code",
        "grower code", "grower reference", "code"
    ],
    "boxes": [
        "boxes", "box", "bx", "cartons", "carton", "ctn", "cases", "case",
        "packs", "pack", "bundles", "bundle", "packages", "pkg",
        "number of boxes", "no. of boxes", "no boxes", "box qty",
        "carton qty", "cartons qty"
    ],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box", "quantity per box",
        "stems per carton", "qty/carton", "quantity/carton",
        "conversion rate", "pack rate per box"
    ],
    "quantity": [
        "quantity", "qty", "qnty", "stems", "pcs", "pieces", "count",
        "total quantity", "total qty", "number of stems", "stem quantity",
        "invoice quantity", "qty invoice", "quantity invoice", "units"
    ],
    "unit_price": [
        "price", "price per stem", "price/stem", "unit price",
        "unit price (usd)", "unit price(usd)", "cost", "price per unit",
        "per stem", "per piece", "amount per stem", "unit cost",
        "priceperstem", "rate per stem", "selling price", "unit selling price"
    ],
    "total": [
        "total", "total price", "total amount", "line total", "line amount",
        "amount", "sub-total", "subtotal", "extended price", "total (usd)",
        "total(usd)", "value", "line value"
    ],
    "length": [
        "length", "length(cm)", "length (cm)", "size", "size(cm)",
        "stem length", "height", "cm", "stem size", "length cm"
    ],
    "discount": ["discount", "disc.", "disc", "rebate"],
    "tax": ["tax", "vat", "gst", "sales tax"],
}

META_LABELS = {
    "invoice_number": [
        "invoice number", "invoice no", "invoice #", "invoice no.", "inv no",
        "inv #", "invoice", "quotation number", "quotation no", "quote number",
        "proforma number", "proforma no", "document number", "doc no",
        "reference", "ref", "ref no", "reference number", "document ref"
    ],
    "date": [
        "date", "date of shipment", "shipment date", "invoice date",
        "issue date", "document date", "quotation date"
    ],
    "due_date": [
        "due date", "payment due", "valid until", "valid till", "expiry",
        "expires", "expiration"
    ],
    "currency": ["currency", "currency code", "ccy"],
    "vat_rate": ["vat rate", "vat rate (%)", "tax rate", "tax %", "vat %"],
    "country_destination": [
        "country of destination", "destination country", "destination",
        "country dest", "ship to country", "country destination"
    ],
    "point_of_entry": [
        "point of entry", "port of entry", "entry point", "port", "airport",
        "arrival port"
    ],
    "country_origin": [
        "country of origin", "origin", "origin country"
    ],
    "consignee_name": [
        "consignee name", "consignee", "bill to", "ship to", "customer name",
        "client name", "buyer name", "customer", "consignee details"
    ],
    "consignee_address": [
        "consignee address", "bill to address", "ship to address",
        "customer address", "buyer address", "delivery address"
    ],
    "seller_name": [
        "seller name", "seller", "vendor", "supplier", "exporter",
        "seller / exporter", "exporter name"
    ],
    "purchase_order_no": [
        "purchase order no", "purchase order #", "purchase order number",
        "purchase order", "po no", "po #", "po number", "customer po",
        "order number", "purchase order no."
    ],
    "payment_terms": [
        "payment terms", "payment term", "terms of payment", "payment"
    ],
    "transportation": [
        "transportation", "transport", "shipment method",
        "mode of transport", "shipping method", "transportation details"
    ],
    "awb_number": [
        "awb number", "awb no", "awb", "air waybill", "tracking number",
        "waybill", "airway bill"
    ],
    "net_weight": [
        "net weight", "net weight (kgs)", "net weight (kg)", "net kg",
        "net weight kgs"
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
    "usd": "USD", "us dollar": "USD", "dollar": "USD",
    "kes": "KES", "ksh": "KES", "kenya shilling": "KES",
    "eur": "EUR", "euro": "EUR", "gbp": "GBP", "pound": "GBP",
    "aed": "AED", "sar": "SAR", "qar": "QAR"
}

# Field-specific label patterns. Longest/more specific labels are tested first.
FIELD_PATTERNS = {
    "boxes": r"(?:no\.?\s*of\s*)?(?:boxes?|bx|cartons?|ctn|cases?|bundles?|packages?)",
    "pack_rate": r"(?:pack\s*rate|packrate|stems?\s*(?:per|/)\s*(?:box|carton)|qty\s*(?:per|/)\s*(?:box|carton)|quantity\s*(?:per|/)\s*(?:box|carton))",
    "quantity": r"(?:quantity|qty|qnty|total\s+qty|total\s+quantity|invoice\s+qty|invoice\s+quantity|stems?|pieces?|pcs|units?)",
    "unit_price": r"(?:price\s*(?:per|/)\s*(?:stem|piece|unit)|price\s*per\s*stem|price/stem|unit\s*price|unit\s*cost|rate\s*per\s*stem|selling\s*price|cost\s*per\s*unit)",
    "total": r"(?:line\s+total|total\s+amount|total\s+price|line\s+amount|extended\s+price|amount|total)",
    "length": r"(?:length(?:\s*\(?(?:cm|cms)\)?)?|stem\s+length|size)\b",
    "farm_code": r"(?:farm\s*code|farm\s*ref(?:erence)?|grower\s*code|supplier\s*code)",
}

# ---------------------------------------------------------------------------
# SAFE NORMALIZATION / NUMBERS
# ---------------------------------------------------------------------------

def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def clean_ocr_text(text: str) -> str:
    """Correct common OCR spacing mistakes without changing values."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\|", " |", text)
    text = re.sub(r"\|\s+", "| ", text)
    return text


def parse_number(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    txt = str(value).strip()
    if not txt:
        return None

    # Keep minus and decimal separators, remove currency/unit text.
    txt = re.sub(r"(?i)\b(?:usd|us\$|kes|ksh|eur|gbp|aed|sar|qar)\b", "", txt)
    txt = txt.replace("$", "").replace("€", "").replace("£", "")
    txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt)
    txt = txt.strip()

    # 1,250.50 / 1.250,50 / 1250.50 / 1250,50
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
    return value is None or str(value).strip() == "" or str(value).strip().lower() in {
        "n/a", "na", "null", "none", "-", "—"
    }


# ---------------------------------------------------------------------------
# LABEL MATCHING
# ---------------------------------------------------------------------------

def match_header(label: str) -> Optional[str]:
    l = norm(label)
    l = re.sub(r"[*:.#]+$", "", l).strip()
    if not l:
        return None

    # Exact matches first.
    for field, names in COLUMN_SYNONYMS.items():
        if l in {norm(x) for x in names}:
            return field

    # Fuzzy/partial matches, with dangerous generic labels handled carefully.
    priority = [
        "pack_rate", "unit_price", "farm_code", "quantity", "boxes",
        "length", "total", "product", "variety", "description"
    ]
    for field in priority:
        for n in COLUMN_SYNONYMS[field]:
            nn = norm(n)
            if len(nn) >= 5 and (nn in l or l in nn):
                return field

    # Fuzzy only for reasonably strong matches.
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
# DOCUMENT CLASSIFICATION / METADATA
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
        # Label: value / Label - value / Label | value
        m = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,70}?)\s*"
            r"(?:[:#]\s*|-\s+|\|\s*)(.*?)\s*$",
            line
        )
        if m:
            label, value = m.group(1), m.group(2).strip()
            field = match_meta_label(label)
            if field and value and not empty_field(value):
                meta[field] = value
                continue

        # OCR/forms frequently put label on one line and value on next line.
        field = match_meta_label(line.rstrip(":#"))
        if field and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and not match_meta_label(nxt) and not re.match(r"^[A-Za-z].*[:#]", nxt):
                if field not in meta:
                    meta[field] = nxt

    # Currency can appear as "USD - US Dollar".
    if "currency" in meta:
        c = norm(meta["currency"])
        for word, code in CURRENCY_WORDS.items():
            if word in c:
                meta["currency"] = code
                break

    meta["document_type"] = detect_document_type(text)
    return meta


# ---------------------------------------------------------------------------
# PRODUCT SAFETY
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
    "date of shipment", "currency", "purchase order", "purchase order #"
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

    # A pure numeric/code string is not a product.
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
# FIELD EXTRACTION FROM FREE-FORM LINES
# ---------------------------------------------------------------------------

def extract_labeled_fields(line: str) -> Dict[str, Any]:
    """
    Extract explicitly labelled values from a line. Crucially, a number is not
    assigned to quantity/price/boxes unless its label identifies the field.
    """
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
            line, re.I
        )
        if not m:
            continue

        raw = m.group(1).strip()
        # Stop a captured value if another known label follows it.
        next_labels = re.search(
            r"\s+(?:pack\s*rate|packrate|qty|quantity|boxes?|cartons?|"
            r"price|unit\s*price|total|farm\s*code|length)\b",
            raw, re.I
        )
        if next_labels:
            raw = raw[:next_labels.start()].strip()

        if field in {"boxes", "pack_rate", "quantity", "unit_price", "total"}:
            value = parse_number(raw)
            if value is not None:
                result[field] = as_number(value)
        elif field == "length":
            lm = re.search(r"\d+(?:\.\d+)?", raw)
            if lm:
                result["specification"] = {
                    "length": f"{lm.group(0)}cm"
                }
        elif field == "farm_code":
            result["farm_code"] = raw.split()[0]

    return result


def strip_known_annotations(name: str) -> str:
    """Remove only tokens that were explicitly recognizable as field data."""
    s = name

    # Remove explicit field phrases and their immediate values.
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

    # Remove currency symbols/standalone currency words only when adjacent to
    # an already recognized numeric token; do not destroy product names.
    s = re.sub(r"(?i)(?<=\d)\s*(?:usd|us\$|kes|ksh|eur|gbp|aed|sar|qar)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -:,;.|")
    return s


def parse_inline_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Handles examples such as:
      Roses Red 70cm | 2 boxes | pack rate 100 | qty 200 | price/stem 0.40
      Alstroemeria Pink qty 200 price 0.40
    Unlabelled numbers are not automatically promoted to financial fields.
    """
    s = line.strip()
    if not s or is_header_or_metadata(s):
        return None

    fields = extract_labeled_fields(s)

    # A bare "100 boxes" or "2 boxes" is safely recognized.
    if "boxes" not in fields:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:boxes?|bx|cartons?|ctn)\b", s, re.I)
        if m:
            fields["boxes"] = as_number(m.group(1))

    if "specification" not in fields:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*cm\b", s, re.I)
        if m:
            fields["specification"] = {"length": f"{m.group(1)}cm"}

    name = strip_known_annotations(s)

    # Remove standalone numbers only if they are attached to an explicit
    # recognizable field; never strip arbitrary numbers from a product name.
    if fields:
        name = re.sub(
            r"\b(?:boxes?|bx|cartons?|ctn|qty|quantity|qnty|pack\s*rate|"
            r"packrate|price|rate|total|amount|length)\b",
            " ", name, flags=re.I
        )
        name = re.sub(r"\s+", " ", name).strip(" -:,;.|")

    # If the line is just a collection of values, reject it.
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
# TABLE PARSING
# ---------------------------------------------------------------------------

def split_columns(line: str) -> List[str]:
    # Prefer pipe/tab separators. Then OCR-created 2+ spaces.
    if "|" in line:
        parts = [x.strip() for x in re.split(r"\s*\|\s*", line)]
    elif "\t" in line:
        parts = [x.strip() for x in line.split("\t")]
    else:
        parts = [x.strip() for x in re.split(r"\s{2,}", line.strip())]
    return [x for x in parts if x != ""]


def find_table_header(lines: List[str]) -> Tuple[Optional[int], List[Optional[str]]]:
    best = None
    for i, line in enumerate(lines):
        cols = split_columns(line)
        if len(cols) < 2:
            continue
        mapped = [match_header(c) for c in cols]
        known = [m for m in mapped if m]

        # Strong header if it has a product-like field and at least one
        # measurable/financial field.
        if (
            len(known) >= 2
            and any(x in known for x in ("product", "variety", "description"))
            and any(x in known for x in ("quantity", "boxes", "unit_price", "total", "pack_rate"))
        ):
            score = len(set(known))
            if best is None or score > best[0]:
                best = (score, i, mapped)

    return (best[1], best[2]) if best else (None, [])


def value_for_field(field: str, raw: str) -> Any:
    if empty_field(raw):
        return None
    if field in {"boxes", "pack_rate", "quantity", "unit_price", "total"}:
        return as_number(parse_number(raw))
    if field == "length":
        m = re.search(r"\d+(?:\.\d+)?", raw)
        return f"{m.group(0)}cm" if m else None
    return raw.strip()


def parse_table_row(line: str, headers: List[Optional[str]]) -> Optional[Dict[str, Any]]:
    cols = split_columns(line)
    if len(cols) < 2:
        return None

    vals: Dict[str, Any] = {}
    raw_values: Dict[str, str] = {}

    for i, cell in enumerate(cols):
        if i >= len(headers):
            # Preserve extra OCR columns instead of silently shifting data.
            raw_values[f"unmapped_{i+1}"] = cell
            continue
        field = headers[i]
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
        # Conservative fallback: only choose a cell that is clearly a
        # textual product and is not a known metadata/header field.
        for cell in cols:
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
    }

    if vals.get("length"):
        item["specification"]["length"] = vals["length"]
    if vals.get("farm_code"):
        item["farm_code"] = str(vals["farm_code"]).strip()

    if raw_values:
        item["raw_values"] = raw_values
    return item


# ---------------------------------------------------------------------------
# TEXT ENGINE
# ---------------------------------------------------------------------------

def parse_order_text(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = clean_ocr_text(text)
    lines = [x.rstrip() for x in text.splitlines()]
    meta = extract_meta_from_text(text)
    items: List[Dict[str, Any]] = []

    header_idx, headers = find_table_header(lines)
    if header_idx is not None:
        for line in lines[header_idx + 1:]:
            if is_header_or_metadata(line):
                continue
            item = parse_table_row(line, headers)
            if item:
                items.append(item)

        # If a table exists, do not mix unrelated metadata lines into items.
        if items:
            return clean_items(items), meta

    # Some pasted/OCR forms are "label value" sequences. We group lines
    # conservatively only when a line clearly contains product text.
    for line in lines:
        s = line.strip()
        if not s or len(s) < 3 or is_header_or_metadata(s):
            continue

        # Skip obvious metadata key:value lines.
        if re.match(r"^[A-Za-z][A-Za-z0-9\s./()%#*&_-]{1,70}\s*[:#]", s):
            if match_meta_label(re.split(r"[:#]", s, 1)[0]):
                continue

        item = parse_inline_line(s)
        if item:
            items.append(item)

    return clean_items(items), meta


# ---------------------------------------------------------------------------
# EXCEL / CSV
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
        # Try first row as header only if it contains known labels.
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
# DOCUMENT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_pdf(content: bytes) -> Tuple[str, str]:
    """
    Returns (text, extraction_method). First uses the embedded PDF text.
    If it is empty/too short, optionally OCRs rendered pages using PyMuPDF.
    """
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
        ocr_parts = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_parts.append(ocr_image(img))
        return "\n".join(ocr_parts), "pdf_ocr"
    except Exception as e:
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
    except Exception as e:
        logger.exception("DOCX failed")
        return ""


def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    # Upscaling substantially improves OCR for small invoice tables.
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
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 4",
    ]
    results = []
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(processed, lang="eng", config=cfg)
            if txt:
                results.append(txt)
        except Exception as e:
            logger.warning("OCR config failed: %s", e)

    if not results:
        return ""

    # Pick the OCR result with the most useful alphanumeric content.
    return max(results, key=lambda x: len(re.findall(r"[A-Za-z0-9]", x)))


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
# NORMALIZATION / VALIDATION
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

    # Only validate mathematical relationships when all required fields exist.
    if q is not None and u is not None and t is not None:
        expected = float(q) * float(u)
        if abs(expected - float(t)) > max(0.02, abs(float(t)) * 0.01):
            warnings.append("quantity_x_unit_price_does_not_match_total")

    if b is not None and p is not None and q is not None:
        expected_q = float(b) * float(p)
        if abs(expected_q - float(q)) > 0.5:
            warnings.append("boxes_x_pack_rate_does_not_match_quantity")

    return warnings


def clean_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    IMPORTANT: Missing values remain None. This intentionally replaces the
    old engine's defaults such as boxes=1, pack_rate=100 and unit_price=0.
    """
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

        # Calculate ONLY when the calculation is unambiguous.
        # Never invent a missing source value.
        if out["quantity"] is None and out["boxes"] is not None and out["pack_rate"] is not None:
            out["quantity"] = as_number(float(out["boxes"]) * float(out["pack_rate"]))

        if out["total"] is None and out["quantity"] is not None and out["unit_price"] is not None:
            out["total"] = round(float(out["quantity"]) * float(out["unit_price"]), 4)

        # Confidence starts high for explicit product identification and is
        # reduced by missing/ambiguous relationships.
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
# PRODUCT MATCHING
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
            return {"success": True, "items": req.items, "matched_count": 0}

        out = []

        for item in req.items:
            iname = normalize_product_for_match(str(item.get("product_name", "")))
            best = None
            best_score = 0

            for product in req.company_products:
                names = [str(product.get("name", ""))]
                aliases = product.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(str(x) for x in aliases)

                for candidate in names:
                    cname = normalize_product_for_match(candidate)
                    if not cname:
                        continue
                    score = max(
                        fuzz.ratio(iname, cname),
                        fuzz.token_set_ratio(iname, cname),
                        fuzz.partial_ratio(iname, cname) if len(iname) >= 5 else 0,
                    )
                    if score > best_score:
                        best_score = score
                        best = product

            # Never silently replace a weak match.
            confidence = best_score / 100.0
            if best is not None and confidence >= MIN_MATCH_CONFIDENCE:
                out.append({
                    **item,
                    "product_id": best.get("id"),
                    "matched_product_name": best.get("name"),
                    "match_confidence": round(confidence, 3),
                    "match_status": "matched"
                })
            else:
                out.append({
                    **item,
                    "product_id": None,
                    "match_confidence": round(confidence, 3),
                    "match_status": "review_required"
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


# ---------------------------------------------------------------------------
# MAIN ANALYSIS ENDPOINT
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "Smart Document Intelligence Engine",
        "version": "6.0.0",
        "status": "operational",
        "capabilities": [
            "invoice", "quotation", "proforma", "receipt", "delivery_note",
            "purchase_order", "images_ocr", "scanned_pdf_ocr",
            "pdf_text", "docx", "xlsx", "xls", "xlsm", "csv", "json", "text",
            "confidence_scoring", "validation", "provenance",
            "safe_blank_fields", "product_matching"
        ],
        "endpoints": [
            "/api/health",
            "/api/analyze (POST)",
            "/api/match-products (POST)",
            "/api/extract-text (POST)"
        ]
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "smart-import-engine",
        "version": "6.0.0",
        "ocr_available": bool(pytesseract),
        "scanned_pdf_ocr_available": fitz is not None,
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    company_id: int = Form(0),
    file_type: Optional[str] = Form(None),
):
    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than {MAX_UPLOAD_MB} MB."
            )

        fname = file.filename or "upload"
        ext = (file_type or Path(fname).suffix.lstrip(".")).lower()

        logger.info(
            "Processing %s (%s), company=%s, size=%s",
            fname, ext, company_id, len(content)
        )

        items: List[Dict[str, Any]] = []
        text_extracted = ""
        extraction_method = ""
        metadata: Dict[str, Any] = {}

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

        # Metadata can also come from spreadsheet labels when the sheet is
        # actually a form; parse text if there is no table result.
        if text_extracted and not metadata:
            metadata = extract_meta_from_text(text_extracted)

        total_boxes = sum(float(x["boxes"]) for x in cleaned if x["boxes"] is not None)
        total_quantity = sum(float(x["quantity"]) for x in cleaned if x["quantity"] is not None)
        total_amount = sum(float(x["total"]) for x in cleaned if x["total"] is not None)

        # Convert integer-looking aggregate numbers back to integers.
        total_boxes = int(total_boxes) if total_boxes.is_integer() else total_boxes
        total_quantity = int(total_quantity) if total_quantity.is_integer() else total_quantity

        warnings = []
        for x in cleaned:
            warnings.extend(x.get("warnings", []))

        return {
            "success": True,
            "items": cleaned,
            "metadata": metadata,
            "document_type": metadata.get("document_type"),
            "total_boxes": total_boxes,
            "total_quantity": total_quantity,
            "total_amount": round(total_amount, 2),
            "item_count": len(cleaned),
            "review_required": any(
                x.get("confidence", 0) < 0.80 or x.get("warnings")
                for x in cleaned
            ),
            "warnings": sorted(set(warnings)),
            "text_extracted": text_extracted[:12000] if text_extracted else "",
            "extraction_method": extraction_method,
            "file_type": ext,
            "filename": fname,
            "engine_version": "6.0.0",
        }

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
