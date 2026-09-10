"""
Smart Document Intelligence Engine v4.0
Understands ALL document formats: invoices, quotations, proformas, receipts,
Word docs, images (OCR), Excel, CSV, plain text.

Handles multi-column headers:
  Product/Service* | Description | Qty | Unit Price | Total
  Boxes | Packrate | Variety | Farm Code | Length(cm) | Quantity | Price/Stem | Total
  Consignee Name | Purchase Order# | Consignee Address
  Date of Shipment | Currency | Country of Origin | Payment Terms
  ... and every variant

ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io, os, re, logging
from typing import List, Dict, Any, Optional
import PyPDF2, docx
from PIL import Image
import pytesseract
from rapidfuzz import fuzz
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Tesseract ----------
TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

app = FastAPI(title="Flower Smart Import Engine", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    items: List[Dict]
    company_products: List[Dict]


# ============================================================
#  KNOWLEDGE BASE — Header labels & their meanings
# ============================================================

# Column header labels -> our internal field
COLUMN_SYNONYMS = {
    "product": [
        "product", "product name", "product/service", "product / service", "item",
        "item name", "item description", "service", "article", "articles",
        "commodity", "goods", "stock item", "particulars",
    ],
    "variety": [
        "variety", "flower", "flower name", "flower type", "species",
        "cultivar", "type", "kind",
    ],
    "description": [
        "description", "desc", "details", "item details", "specification",
        "specifications", "remarks", "notes",
    ],
    "farm_code": [
        "farm code", "farmcode", "code", "fookrate code", "fookrate",
        "farm reference", "supplier code",
    ],
    "boxes": [
        "boxes", "box", "bx", "cartons", "carton", "ctn", "cases", "case",
        "packs", "pack", "units", "unit", "bundles", "bunch", "bunches",
        "packages", "pkg",
    ],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box", "quantity per box",
        "unit rate", "conversion rate",
    ],
    "quantity": [
        "quantity", "qty", "qnty", "amount", "units", "stems", "pcs", "pieces",
        "number", "no.", "count", "total quantity", "total qty",
    ],
    "unit_price": [
        "price", "price per stem", "price/stem", "unit price", "unit price(usd)",
        "rate", "cost", "price per unit", "per stem", "per piece", "amount per stem",
        "unit cost", "unit rate", "priceperstem",
    ],
    "total": [
        "total", "total price", "total amount", "line total", "amount",
        "sub-total", "subtotal", "line amount", "extended price", "total(usd)",
    ],
    "length": [
        "length", "length(cm)", "length (cm)", "size", "size(cm)", "stem length",
        "cm", "height",
    ],
    "discount": ["discount", "disc.", "disc", "rebate"],
    "tax": ["tax", "vat", "gst", "sales tax"],
    "notes": ["notes", "note", "comments", "comment", "remarks"],
}

# Document-level metadata labels -> our internal field
META_LABELS = {
    "invoice_number": [
        "invoice number", "invoice no", "invoice #", "invoice no.", "invoice no:",
        "inv no", "inv #", "invoice", "quotation number", "quotation no",
        "proforma number", "proforma no", "document number", "doc no", "reference",
        "ref", "ref no", "reference number",
    ],
    "date": [
        "date", "date of shipment", "shipment date", "invoice date", "issue date",
        "document date", "quotation date",
    ],
    "due_date": [
        "due date", "due", "payment due", "valid until", "valid till", "expiry",
        "expires", "expiration",
    ],
    "currency": ["currency", "currency code", "ccy"],
    "vat_rate": ["vat rate", "vat rate (%)", "tax rate", "tax %", "vat %"],
    "country_destination": [
        "country of destination", "destination country", "destination", "country dest",
        "ship to country", "country",
    ],
    "point_of_entry": [
        "point of entry", "port of entry", "entry point", "port", "airport",
        "arrival port",
    ],
    "country_origin": ["country of origin", "origin", "origin country"],
    "consignee_name": [
        "consignee name", "consignee", "bill to", "ship to", "customer name",
        "client name", "buyer name", "customer",
    ],
    "consignee_address": [
        "consignee address", "address", "bill to address", "ship to address",
        "customer address", "buyer address",
    ],
    "seller_name": ["seller name", "seller", "vendor", "supplier", "from", "exporter"],
    "purchase_order_no": [
        "purchase order no", "purchase order #", "purchase order number",
        "purchase order", "po no", "po #", "po number", "customer po", "order number",
    ],
    "payment_terms": ["payment terms", "payment term", "terms of payment", "payment"],
    "transportation": ["transportation", "transport", "shipment method", "mode of transport", "shipping method"],
    "awb_number": ["awb number", "awb no", "awb", "air waybill", "tracking number", "waybill"],
    "net_weight": ["net weight", "weight", "net weight (kgs)", "net weight (kg)", "net kg", "gross weight"],
    "notes": ["notes", "note", "comments", "comment"],
}

# Doc type keywords
DOC_TYPES = {
    "invoice": ["invoice", "tax invoice", "commercial invoice"],
    "quotation": ["quotation", "quote"],
    "proforma": ["proforma", "pro forma", "pro-forma", "proforma invoice"],
    "receipt": ["receipt", "payment receipt"],
    "delivery_note": ["delivery note", "delivery"],
    "credit_note": ["credit note", "credit memo"],
}

# Lines that are DEFINITELY not products
HARD_REJECT_PATTERNS = [
    # Document meta
    r'^\s*(invoice|quotation|proforma|pro\s*forma|receipt|delivery\s*note|credit\s*note)\s*(number|no\.?|#)?\s*[:.]?\s*[\w/\-]*\s*$',
    r'^\s*(inv|qtn|pfr|inv-|qtn-|pfr-|flr|pm[y]?)[/\-\s]?[\w\-]*\s*$',
    r'^\s*#?\s*\w{2,5}\d{4,}[\/\-]?\d*\s*$',
    # Section headers
    r'^\s*(bill\s+to|ship\s+to|sold\s+to|buyer|seller|customer|consignee|consignor)\s*:?\s*$',
    r'^\s*(invoice\s+details|document\s+details|consignee\s+details|seller\s+details|order\s+details|shipment\s+details|items?|products?|order\s+items)\s*:?\s*$',
    r'^\s*#?\s*(product|item|variety|description|qty|quantity|unit\s+price|price|total|amount|boxes|pack\s*rate|packrate|farm\s*code|length)\b.*\b(qty|quantity|price|total|amount|rate)\b',
    r'^\s*product\s*[\/|]?\s*service\s*\*?\s*description\b',
    # Metadata key labels (just the label, no product)
    r'^\s*(date\s+of\s+shipment|invoice\s+date|document\s+date|due\s+date|valid\s+until|valid\s+till|expiry)\s*[:.]?\s*$',
    r'^\s*(currency|vat\s*rate|tax\s*rate|country\s+of\s+destination|country\s+of\s+origin|point\s+of\s+entry|payment\s+terms|transportation|awb\s+number|net\s+weight|consignee\s+name|consignee\s+address|seller\s+name|purchase\s+order\s*(no|number|#)?)\s*[:.#]?\s*$',
    # Contacts
    r'^\s*(phone|tel|telephone|mobile|fax|email|e-mail|web|website|url)\s*[:.]',
    r'^\s*\+?\d[\d\s\-()]{7,}\s*$',
    r'^\s*[\w.\-]+@[\w.\-]+\.\w+\s*$',
    r'^\s*https?://\S+\s*$',
    # Dates alone
    r'^\s*(date|due\s+date|valid\s+until|issued|date\s+of\s+shipment)\s*[:.]',
    r'^\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s*$',
    r'^\s*\d{4}-\d{2}-\d{2}([T\s]\d{2}:\d{2})?\s*$',
    r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\s*$',
    # Totals
    r'^\s*(sub[\s\-]?total|grand\s+total|total\s+amount|amount\s+due|balance\s+due|balance)\b',
    r'^\s*(tax|vat|gst)\s*\(?\d*%?\)?\s*[:.]',
    r'^\s*(shipping|delivery|freight|discount|rebate)\s*[:.]',
    # Notes / terms
    r'^\s*(notes?|terms?|conditions?|terms?\s*(and|&)\s*conditions?)\s*[:.]?\s*$',
    r'^\s*thank\s+you',
    r'^\s*this\s+is\s+a\s+computer[\s\-]?generated',
    r'^\s*generated\s+on\s*[:.]',
    r'^\s*\d+\.\s*(prices?|all\s+prices|delivery|payment|total\s+price|goods|the\s+)',
    r'^\s*page\s+\d+',
    # Addresses
    r'^\s*(p\.?\s*o\.?\s*box|po\s+box|postal\s+code)\s',
    r'^\s*(building|street|road|avenue|floor|suite|tower|plaza|centre|center)\b',
    r'^\s*\d+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*(street|st\.?|road|rd\.?|avenue|ave\.?|building|bldg|tower|centre|center|plaza)\b',
    # Countries/cities alone
    r'^\s*(kenya|tanzania|uganda|uae|united\s+arab\s+emirates|saudi\s+arabia|qatar|netherlands|ethiopia|egypt|rwanda|burundi|south\s+africa|italy|kazakhstan|nairobi|mombasa|kisumu|kampala|dubai|abu\s+dhabi|sharjah|riyadh|jeddah|dammam|doha|amman|rome|almaty|johannesburg|cape\s+town)\s*$',
    # Just symbols/numbers
    r'^\s*[\d\s.,:;/\-()%#KESkes]*\s*$',
]

def is_metadata_or_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 3:
        return True

    for pat in HARD_REJECT_PATTERNS:
        if re.search(pat, s, re.IGNORECASE):
            return True

    # No letters at all
    if not re.search(r'[A-Za-z]{2,}', s):
        return True

    # Company-suffix-only lines
    if re.match(r'^\s*[\w&.\s]+(limited|ltd\.?|llc|inc\.?|plc|company|enterprises?|establishment|investment|trading|holdings?)\s*\.?\s*$',
                s, re.IGNORECASE):
        return True

    return False


def looks_like_product(name: str) -> bool:
    """Final check: does this name look like an actual product?"""
    n = name.strip()
    if len(n) < 3:
        return False
    if not re.search(r'[A-Za-z]{2,}', n):
        return False
    # Reject obvious non-products
    if re.search(r'@|https?://|www\.', n):
        return False
    if re.search(r'\b(limited|ltd|llc|inc|plc|establishment|investment|trading|holdings?|company)\b', n, re.IGNORECASE):
        # Exception: if "company" is followed by product words like flower/rose etc.
        if not re.search(r'\b(flower|rose|plant|goods|supplies|item|product)\b', n, re.IGNORECASE):
            return False
    if re.search(r'\b(street|road|avenue|building|floor|suite|tower|plaza|box|postal)\b', n, re.IGNORECASE):
        return False
    if re.search(r'\+?\d[\d\s\-()]{7,}', n):
        return False
    # Must contain at least 3 alpha chars that are contiguous words
    if not re.search(r'\b[A-Za-z]{3,}', n):
        return False
    return True


# ============================================================
#  PARSE HELPERS
# ============================================================
NUMBER_RE = re.compile(r'-?\d{1,3}(?:[,.\s]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?')

def parse_number(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        try:
            f = float(s)
            if pd.isna(f):
                return None
            return f
        except Exception:
            return None
    txt = str(s).strip()
    if not txt:
        return None
    # Remove currency codes and symbols
    txt = re.sub(r'[A-Za-z$€£¥₹]', '', txt).strip()
    # Remove thousands separators (comma or space) but keep decimal point
    # Detect decimal separator: if last separator is followed by 1-2 digits -> decimal
    if re.search(r'[,.]\d{1,2}$', txt) and not re.search(r'\d{3}$', txt):
        # Decimal point
        head, sep, tail = re.match(r'^(.*?)([,.])(\d{1,2})$', txt).groups()
        head = re.sub(r'[,.\s]', '', head)
        txt = f"{head}.{tail}"
    else:
        txt = re.sub(r'[,.\s]', '', txt)
    try:
        return float(txt)
    except Exception:
        return None


def match_header(label: str) -> Optional[str]:
    """Given a header cell, return our internal field name."""
    l = label.strip().lower()
    l = re.sub(r'\s*[\*:.#]+\s*$', '', l)   # strip trailing markers
    l = re.sub(r'^\s*#\s*', '', l)           # strip leading #
    l = re.sub(r'\s+', ' ', l)

    # Direct synonyms first
    for field, names in COLUMN_SYNONYMS.items():
        for n in names:
            if l == n:
                return field
    # Then fuzzy / partial
    for field, names in COLUMN_SYNONYMS.items():
        for n in names:
            if n in l or l in n:
                return field
    return None


def match_meta_label(line: str) -> Optional[str]:
    """If a line looks like `Key: Value`, return which meta key it is."""
    m = re.match(r'^\s*([A-Za-z][A-Za-z\s./()%#\-]{1,60}?)\s*[:#]\s*(.*)$', line)
    if not m:
        return None
    key = re.sub(r'\s+', ' ', m.group(1).strip().lower())
    key = re.sub(r'[\*\s]+$', '', key)
    for meta, names in META_LABELS.items():
        for n in names:
            if key == n or key.startswith(n) or n.startswith(key):
                return meta
    return None


# ============================================================
#  EXCEL PARSER (Robust)
# ============================================================
def extract_from_excel(content: bytes) -> List[Dict]:
    try:
        df = pd.read_excel(io.BytesIO(content), header=None)
    except Exception as e:
        logger.error(f"Excel read: {e}")
        return []

    if df.empty:
        return []

    # ---- Find the header row ----
    header_row = None
    header_map: Dict[int, str] = {}
    for i in range(min(30, len(df))):
        row = df.iloc[i]
        row_str = ' '.join(str(v).lower() for v in row if pd.notna(v))
        # Look for at least two header keywords
        hits = 0
        for kw in ('product','item','variety','description','qty','quantity','price','total','amount','box','pack'):
            if kw in row_str:
                hits += 1
        if hits >= 2:
            # Try to map each cell
            temp_map = {}
            for idx, val in enumerate(row):
                if pd.isna(val):
                    continue
                field = match_header(str(val))
                if field and field not in temp_map.values():
                    temp_map[idx] = field
            # Need at least product/description + (qty or price)
            fields = set(temp_map.values())
            if ('product' in fields or 'variety' in fields or 'description' in fields) and \
               ('quantity' in fields or 'unit_price' in fields or 'boxes' in fields):
                header_row = i
                header_map = temp_map
                break

    if header_row is None:
        # No clear header: fall back to first column as product
        header_row = -1
        header_map = {0: 'product'}

    items = []
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]
        # Gather cell values by field
        vals: Dict[str, Any] = {}
        for idx, field in header_map.items():
            if idx < len(row) and pd.notna(row.iloc[idx]):
                vals[field] = row.iloc[idx]

        if not vals:
            continue

        # Product name from most specific field first
        name = None
        for f in ('product', 'variety', 'description'):
            if f in vals and str(vals[f]).strip():
                candidate = str(vals[f]).strip()
                if not is_metadata_or_header(candidate):
                    name = candidate
                    break
        if not name:
            # Try any text cell in the row
            for v in vals.values():
                s = str(v).strip()
                if s and not is_metadata_or_header(s) and looks_like_product(s):
                    name = s
                    break
        if not name or not looks_like_product(name):
            continue

        item: Dict[str, Any] = {'product_name': name}

        if 'specification' not in item:
            item['specification'] = {}
        if 'length' in vals:
            item['specification']['length'] = str(vals['length']).replace('.0', '') + 'cm' \
                if not str(vals['length']).lower().endswith('cm') else str(vals['length'])

        # Numeric fields
        for src, dest in [
            ('boxes', 'boxes'), ('pack_rate', 'pack_rate'),
            ('quantity', 'quantity'), ('unit_price', 'unit_price'),
            ('total', 'total'), ('discount', 'discount'), ('tax', 'tax'),
        ]:
            if src in vals:
                n = parse_number(vals[src])
                if n is not None:
                    item[dest] = n

        # Farm code
        if 'farm_code' in vals and str(vals['farm_code']).strip():
            item['farm_code'] = str(vals['farm_code']).strip()

        # Defaults
        item.setdefault('boxes', 1)
        item.setdefault('pack_rate', 100)
        item.setdefault('unit_price', 0.0)

        # If we have a line total + unit price but no qty, infer qty
        if item.get('total') and item.get('unit_price') and not item.get('quantity'):
            try:
                item['quantity'] = int(round(item['total'] / item['unit_price']))
            except Exception:
                pass

        if not item.get('quantity'):
            item['quantity'] = int(item['boxes']) * int(item['pack_rate'])

        item['total'] = item.get('total') or float(item['quantity']) * float(item['unit_price'])
        items.append(item)

    return items


# ============================================================
#  TEXT EXTRACTORS
# ============================================================
def extract_text_from_pdf(content: bytes) -> str:
    try:
        r = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join((p.extract_text() or '') for p in r.pages)
    except Exception as e:
        logger.error(f"PDF: {e}")
        return ""


def extract_text_from_docx(content: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(content))
        parts = []
        for p in d.paragraphs:
            if p.text.strip():
                parts.append(p.text)
        # Also extract tables
        for t in d.tables:
            for row in t.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(' | '.join(cells))
        return "\n".join(parts)
    except Exception as e:
        logger.error(f"DOCX: {e}")
        return ""


def extract_text_from_image(content: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        try:
            # OCR: use --psm 6 (uniform block of text) for invoices
            return pytesseract.image_to_string(img, lang='eng', config='--psm 6')
        except Exception as e:
            logger.warning(f"OCR psm6 failed, retry default: {e}")
            return pytesseract.image_to_string(img, lang='eng')
    except Exception as e:
        logger.error(f"Image: {e}")
        return ""


# ============================================================
#  TEXT PARSER — Handles "Key: Value" AND tabular data
# ============================================================
def parse_order_text(text: str) -> List[Dict]:
    """Parse text, extracting both metadata (ignored) and product rows."""
    lines = [l.rstrip() for l in text.split('\n')]
    items: List[Dict] = []

    # 1) Try to find a table header line and split rows below it
    header_idx, header_cols = find_table_header(lines)
    if header_idx is not None and len(header_cols) >= 2:
        for line in lines[header_idx + 1:]:
            item = parse_table_row(line, header_cols)
            if item:
                items.append(item)
        if items:
            return items

    # 2) Fall back to per-line heuristic
    for raw in lines:
        line = raw.strip()
        if not line or len(line) < 5:
            continue
        if is_metadata_or_header(line):
            continue
        # Skip lines that are just `Key: Value` metadata
        if match_meta_label(line):
            continue
        item = parse_inline_line(line)
        if item:
            items.append(item)
    return items


def find_table_header(lines: List[str]):
    """Return (index, [field names]) of the table header row."""
    for i, line in enumerate(lines):
        # Split by multiple spaces, tabs, or pipe
        cols = re.split(r'\s{2,}|\t+|\s*\|\s*', line.strip())
        cols = [c for c in cols if c.strip()]
        if len(cols) < 2:
            continue
        mapped = [match_header(c) for c in cols]
        known = [m for m in mapped if m]
        # Need at least 2 known columns including qty/price
        if len(known) >= 2 and ('quantity' in known or 'unit_price' in known or 'total' in known or 'boxes' in known):
            return i, mapped
    return None, []


def parse_table_row(line: str, headers: List[Optional[str]]) -> Optional[Dict]:
    cols = re.split(r'\s{2,}|\t+|\s*\|\s*', line.strip())
    cols = [c for c in cols if c.strip()]
    if len(cols) < 2:
        return None

    vals: Dict[str, Any] = {}
    for i, cell in enumerate(cols):
        if i >= len(headers):
            break
        field = headers[i]
        if not field:
            continue
        vals[field] = cell.strip()

    if not vals:
        return None

    name = None
    for f in ('product', 'variety', 'description'):
        if f in vals and vals[f]:
            candidate = str(vals[f]).strip()
            if not is_metadata_or_header(candidate):
                name = candidate
                break
    if not name:
        # Use the first text-only cell
        for cell in cols:
            if looks_like_product(cell):
                name = cell.strip()
                break
    if not name or not looks_like_product(name):
        return None

    item: Dict[str, Any] = {'product_name': name, 'specification': {}}
    for src, dest in [('boxes','boxes'),('pack_rate','pack_rate'),('quantity','quantity'),
                      ('unit_price','unit_price'),('total','total'),
                      ('discount','discount'),('tax','tax')]:
        if src in vals:
            n = parse_number(vals[src])
            if n is not None:
                item[dest] = n
    if 'farm_code' in vals and vals['farm_code']:
        item['farm_code'] = vals['farm_code']
    if 'length' in vals and vals['length']:
        lv = str(vals['length']).replace('.0','').strip()
        item['specification']['length'] = lv if lv.lower().endswith('cm') else lv + 'cm'

    item.setdefault('boxes', 1)
    item.setdefault('pack_rate', 100)
    item.setdefault('unit_price', 0.0)
    if not item.get('quantity'):
        if item.get('total') and item.get('unit_price'):
            try: item['quantity'] = int(round(item['total'] / item['unit_price']))
            except Exception: pass
    if not item.get('quantity'):
        item['quantity'] = int(item['boxes']) * int(item['pack_rate'])
    item['total'] = item.get('total') or float(item['quantity']) * float(item['unit_price'])
    return item


def parse_inline_line(line: str) -> Optional[Dict]:
    """Parse a free-form line like `Hydrangea Pink 50cm packrate 60 3bx price 1.65`."""
    if not looks_like_product(line):
        return None

    item: Dict[str, Any] = {}

    m = re.search(r'(\d+)\s*cm', line, re.IGNORECASE)
    if m: item['specification'] = {'length': m.group(1) + 'cm'}

    m = re.search(r'pack\s*rate?\s*[:=]?\s*(\d+)', line, re.IGNORECASE)
    if m: item['pack_rate'] = int(m.group(1))

    m = re.search(r'(\d+)\s*(?:bx|box|boxes|carton|cartons|ctn)', line, re.IGNORECASE)
    if m: item['boxes'] = int(m.group(1))

    m = re.search(r'qty\s*[:=]?\s*(\d+)', line, re.IGNORECASE)
    if m: item['quantity'] = int(m.group(1))

    # Price
    m = re.search(r'(?:price|rate|@|usd|kes|eur|gbp)\s*[:=]?\s*([\d,.]+)', line, re.IGNORECASE)
    if m:
        n = parse_number(m.group(1))
        if n is not None:
            item['unit_price'] = n

    # If line ends with two numbers (qty + price)
    tail = re.findall(r'(\d[\d,.]*)\s*$', line)
    if tail and 'unit_price' not in item:
        n = parse_number(tail[-1])
        if n is not None and n > 0:
            item['unit_price'] = n

    # Product name: strip all numeric markers
    name = line
    for pat in [r'pack\s*rate?\s*[:=]?\s*\d+', r'\d+\s*(?:bx|box|boxes|carton|cartons|ctn)',
                r'(?:price|rate|@)\s*[:=]?\s*[\d,.]+', r'\d+\s*cm', r'per\s*stem',
                r'qty\s*[:=]?\s*\d+', r'\b(usd|kes|eur|gbp)\b',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b']:
        name = re.sub(pat, ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip(' -:,;.|')
    if not name or len(name) < 3 or not looks_like_product(name):
        return None

    item['product_name'] = name
    item.setdefault('boxes', 1)
    item.setdefault('pack_rate', 100)
    item.setdefault('unit_price', 0.0)
    item.setdefault('quantity', item['boxes'] * item['pack_rate'])
    return item


# ============================================================
#  ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    return {"service": "Flower Smart Import Engine", "version": "4.0.0", "status": "operational"}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "smart-import-engine", "version": "4.0.0",
            "timestamp": pd.Timestamp.now().isoformat()}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), company_id: int = Form(0),
                  file_type: Optional[str] = Form(None)):
    try:
        content = await file.read()
        fname = file.filename or "upload"
        if not file_type:
            file_type = fname.split('.')[-1].lower() if '.' in fname else 'txt'
        logger.info(f"Processing {fname} ({file_type}) company={company_id} size={len(content)}")

        items, text_extracted = [], ""

        if file_type in ('xlsx', 'xls', 'xlsm'):
            items = extract_from_excel(content)
        elif file_type == 'pdf':
            text_extracted = extract_text_from_pdf(content)
            items = parse_order_text(text_extracted)
        elif file_type in ('docx', 'doc'):
            text_extracted = extract_text_from_docx(content)
            items = parse_order_text(text_extracted)
        elif file_type in ('jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'):
            text_extracted = extract_text_from_image(content)
            items = parse_order_text(text_extracted)
        elif file_type == 'csv':
            try:
                items = extract_from_excel(content)
                if not items:
                    text_extracted = content.decode('utf-8', errors='ignore')
                    items = parse_order_text(text_extracted)
            except Exception:
                text_extracted = content.decode('utf-8', errors='ignore')
                items = parse_order_text(text_extracted)
        else:
            text_extracted = content.decode('utf-8', errors='ignore')
            items = parse_order_text(text_extracted)

        # Final pass — clean and total
        cleaned = []
        tb = tq = 0
        ta = 0.0
        for it in items:
            name = str(it.get('product_name', '')).strip()
            if not looks_like_product(name):
                continue
            if not it.get('quantity'):
                it['quantity'] = int(it.get('boxes', 1)) * int(it.get('pack_rate', 100))
            it['total'] = float(it.get('total') or (float(it.get('quantity', 0)) * float(it.get('unit_price', 0))))
            tb += int(it.get('boxes', 0))
            tq += int(it.get('quantity', 0))
            ta += float(it.get('total', 0))
            cleaned.append(it)

        logger.info(f"Extracted {len(cleaned)} real products from {fname}")
        return {
            'success': True,
            'items': cleaned,
            'total_boxes': tb,
            'total_quantity': tq,
            'total_amount': ta,
            'text_extracted': text_extracted[:1500] if text_extracted else '',
            'item_count': len(cleaned),
            'file_type': file_type,
            'filename': fname,
        }
    except Exception as e:
        logger.error(f"analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match-products")
async def match_products_endpoint(req: MatchRequest):
    try:
        if not req.items or not req.company_products:
            return {'success': True, 'items': req.items}
        aliases_map = {}
        for p in req.company_products:
            al = [str(p.get('name', '')).lower()]
            if isinstance(p.get('aliases'), list):
                al.extend(str(a).lower() for a in p['aliases'])
            aliases_map[p['id']] = al
        out = []
        for item in req.items:
            best, score = None, 0
            iname = str(item.get('product_name', '')).lower()
            for p in req.company_products:
                s = fuzz.ratio(iname, str(p.get('name', '')).lower())
                for a in aliases_map.get(p['id'], []):
                    s = max(s, fuzz.ratio(iname, a))
                if s < 70:
                    pn = str(p.get('name', '')).lower()
                    if pn and (pn in iname or iname in pn):
                        s = 75
                if s > score and s >= 60:
                    score, best = s, p
            if best:
                out.append({**item, 'product_id': best['id'], 'product_name': best['name'],
                            'match_confidence': score / 100})
            else:
                out.append({**item, 'product_id': None, 'match_confidence': 0})
        return {'success': True, 'items': out,
                'matched_count': sum(1 for i in out if i.get('product_id'))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
