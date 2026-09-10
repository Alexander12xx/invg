"""
Smart Document Intelligence Engine
Flower Business SaaS
ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import json
import re
import os
import logging
from typing import List, Dict, Any, Optional
import PyPDF2
import docx
from PIL import Image
import pytesseract
from rapidfuzz import fuzz
import uvicorn

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- Tesseract path ----------
TESS_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if os.path.exists(TESS_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD
    logger.info(f"Tesseract command set to: {TESS_CMD}")
else:
    logger.warning(f"Tesseract not found at {TESS_CMD}, OCR will be skipped")

# ---------- FastAPI ----------
app = FastAPI(
    title="Flower Business Smart Import Engine",
    description="AI-powered document intelligence for flower businesses",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class OrderItem(BaseModel):
    product_name: str = ""
    specification: Dict[str, Any] = {}
    boxes: int = 1
    pack_rate: int = 100
    quantity: int = 0
    unit_price: float = 0.0
    total: float = 0.0

class MatchRequest(BaseModel):
    items: List[Dict]
    company_products: List[Dict]


# ---------- Extractors ----------
def extract_text_from_excel(content: bytes) -> List[Dict]:
    try:
        df = pd.read_excel(io.BytesIO(content))
        items = []
        col_map = {}
        for col in df.columns:
            c = str(col).lower()
            if any(w in c for w in ['product', 'variety', 'flower', 'name', 'item']):
                col_map.setdefault('product', col)
            elif any(w in c for w in ['box', 'carton', 'case']):
                col_map.setdefault('boxes', col)
            elif any(w in c for w in ['pack', 'rate', 'bunch']):
                col_map.setdefault('pack_rate', col)
            elif any(w in c for w in ['price', 'cost']) or c == 'rate':
                col_map.setdefault('price', col)
            elif any(w in c for w in ['quantity', 'qty', 'stem']):
                col_map.setdefault('quantity', col)
            elif any(w in c for w in ['length', 'size', 'cm']):
                col_map.setdefault('length', col)

        for _, row in df.iterrows():
            item = {}
            if 'product' in col_map and pd.notna(row[col_map['product']]):
                item['product_name'] = str(row[col_map['product']]).strip()
            else:
                first = df.columns[0]
                if pd.notna(row[first]):
                    item['product_name'] = str(row[first]).strip()

            if 'boxes' in col_map and pd.notna(row[col_map['boxes']]):
                try: item['boxes'] = int(float(row[col_map['boxes']]))
                except: item['boxes'] = 1
            else:
                item['boxes'] = 1

            if 'pack_rate' in col_map and pd.notna(row[col_map['pack_rate']]):
                try: item['pack_rate'] = int(float(row[col_map['pack_rate']]))
                except: item['pack_rate'] = 100
            else:
                item['pack_rate'] = 100

            if 'price' in col_map and pd.notna(row[col_map['price']]):
                try: item['unit_price'] = float(str(row[col_map['price']]).replace(',', ''))
                except: item['unit_price'] = 0.0
            else:
                item['unit_price'] = 0.0

            if 'quantity' in col_map and pd.notna(row[col_map['quantity']]):
                try: item['quantity'] = int(float(row[col_map['quantity']]))
                except: item['quantity'] = item['boxes'] * item['pack_rate']
            else:
                item['quantity'] = item['boxes'] * item['pack_rate']

            if 'length' in col_map and pd.notna(row[col_map['length']]):
                item['specification'] = {'length': str(row[col_map['length']]).replace('.0','') + 'cm'}

            if item.get('product_name'):
                items.append(item)
        return items
    except Exception as e:
        logger.error(f"Excel parsing error: {e}")
        return []


def extract_text_from_pdf(content: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        return ""


def extract_text_from_docx(content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception as e:
        logger.error(f"DOCX parsing error: {e}")
        return ""


def extract_text_from_image(content: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            if text and text.strip():
                return text
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        return ""
    except Exception as e:
        logger.error(f"Image OCR error: {e}")
        return ""


def parse_order_text(text: str) -> List[Dict]:
    """Parse order text into structured items"""
    items = []
    for raw in text.split('\n'):
        line = raw.strip()
        if not line or len(line) < 5:
            continue
        if any(w in line.lower() for w in ['product', 'item', 'description', 'subtotal']) and len(line.split()) < 3:
            continue

        item = {}

        m = re.search(r'(\d+)\s*cm', line, re.IGNORECASE)
        if m: item['specification'] = {'length': m.group(1) + 'cm'}

        m = re.search(r'pack\s*rate?\s*[:=]?\s*(\d+)', line, re.IGNORECASE)
        if m: item['pack_rate'] = int(m.group(1))

        m = re.search(r'(\d+)\s*(?:bx|box|boxes|carton)', line, re.IGNORECASE)
        if m: item['boxes'] = int(m.group(1))

        m = re.search(r'(?:price|rate|@)\s*[:=]?\s*([\d.]+)', line, re.IGNORECASE)
        if m:
            try: item['unit_price'] = float(m.group(1))
            except: pass

        # Extract product name
        name = line
        for pat in [r'pack\s*rate?\s*[:=]?\s*\d+', r'\d+\s*(?:bx|box|boxes|carton)',
                    r'(?:price|rate|@)\s*[:=]?\s*[\d.]+', r'\d+\s*cm', r'per\s*stem',
                    r'qty\s*[:=]?\s*\d+']:
            name = re.sub(pat, '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name).strip(' -:,;')
        if name and len(name) > 2:
            item['product_name'] = name

        if item.get('product_name'):
            item.setdefault('boxes', 1)
            item.setdefault('pack_rate', 100)
            item.setdefault('unit_price', 0.0)
            item['quantity'] = item.get('boxes', 1) * item.get('pack_rate', 100)
            items.append(item)
    return items


def match_products_fn(items: List[Dict], company_products: List[Dict]) -> List[Dict]:
    if not company_products:
        return items
    product_aliases = {}
    for p in company_products:
        aliases = [str(p.get('name', '')).lower()]
        if 'aliases' in p and isinstance(p['aliases'], list):
            aliases.extend([str(a).lower() for a in p['aliases']])
        product_aliases[p['id']] = aliases

    out = []
    for item in items:
        best, best_score = None, 0
        iname = str(item.get('product_name', '')).lower()
        for p in company_products:
            score = fuzz.ratio(iname, str(p.get('name','')).lower())
            for a in product_aliases.get(p['id'], []):
                score = max(score, fuzz.ratio(iname, a))
            if score < 70:
                pn = str(p.get('name','')).lower()
                if pn and (pn in iname or iname in pn):
                    score = 75
            if score > best_score and score >= 60:
                best_score, best = score, p
        if best:
            out.append({**item, 'product_id': best['id'],
                        'product_name': best['name'],
                        'match_confidence': best_score / 100})
        else:
            out.append({**item, 'product_id': None, 'match_confidence': 0})
    return out


# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {
        "service": "Flower Business Smart Import Engine",
        "version": "2.1.0",
        "status": "operational",
        "endpoints": ["/api/health", "/api/analyze (POST)", "/api/match-products (POST)"],
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "smart-import-engine",
        "version": "2.1.0",
        "timestamp": pd.Timestamp.now().isoformat(),
    }


@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    company_id: int = Form(0),
    file_type: Optional[str] = Form(None),
):
    try:
        content = await file.read()
        fname = file.filename or "upload"
        if not file_type:
            file_type = fname.split('.')[-1].lower() if '.' in fname else 'txt'
        logger.info(f"Processing {fname} ({file_type}) for company {company_id} ({len(content)} bytes)")

        items, text_extracted = [], ""

        if file_type in ('xlsx', 'xls'):
            items = extract_text_from_excel(content)
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
                items = extract_text_from_excel(content)
            except Exception:
                text_extracted = content.decode('utf-8', errors='ignore')
                items = parse_order_text(text_extracted)
        else:
            text_extracted = content.decode('utf-8', errors='ignore')
            items = parse_order_text(text_extracted)

        # Totals
        total_boxes = total_qty = 0
        total_amount = 0.0
        for it in items:
            if not it.get('quantity'):
                it['quantity'] = it.get('boxes', 1) * it.get('pack_rate', 100)
            it['total'] = it.get('quantity', 0) * it.get('unit_price', 0)
            total_boxes += it.get('boxes', 0)
            total_qty += it.get('quantity', 0)
            total_amount += it.get('total', 0)

        logger.info(f"Extracted {len(items)} items from {fname}")
        return {
            'success': True,
            'items': items,
            'total_boxes': total_boxes,
            'total_quantity': total_qty,
            'total_amount': total_amount,
            'text_extracted': text_extracted[:500] if text_extracted else '',
            'item_count': len(items),
            'file_type': file_type,
            'filename': fname,
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match-products")
async def match_products_endpoint(req: MatchRequest):
    try:
        if not req.items:
            return {'success': True, 'items': []}
        if not req.company_products:
            return {'success': True, 'items': req.items}
        matched = match_products_fn(req.items, req.company_products)
        return {'success': True, 'items': matched,
                'matched_count': sum(1 for i in matched if i.get('product_id'))}
    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        fname = file.filename or ""
        ext = fname.split('.')[-1].lower() if '.' in fname else ""
        if ext == 'pdf':
            text = extract_text_from_pdf(content)
        elif ext in ('docx', 'doc'):
            text = extract_text_from_docx(content)
        elif ext in ('jpg', 'jpeg', 'png', 'gif'):
            text = extract_text_from_image(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        return {'success': True, 'text': text, 'length': len(text), 'file_type': ext}
    except Exception as e:
        logger.error(f"Extract error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
