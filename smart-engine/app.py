from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import json
import re
from typing import List, Dict, Any
import PyPDF2
import docx
from PIL import Image
import pytesseract
from rapidfuzz import fuzz, process

app = FastAPI(title="Smart Document Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrderItem(BaseModel):
    product_name: str
    specification: Dict[str, Any] = {}
    boxes: int = 1
    pack_rate: int = 100
    quantity: int = 0
    unit_price: float = 0.0
    total: float = 0.0

class OrderResult(BaseModel):
    items: List[OrderItem]
    total_boxes: int = 0
    total_quantity: int = 0
    total_amount: float = 0.0
    confidence: float = 0.0

# Product matching functions
def extract_text_from_excel(file_content: bytes) -> List[Dict]:
    df = pd.read_excel(io.BytesIO(file_content))
    items = []
    for _, row in df.iterrows():
        item = {}
        # Try to detect columns
        for col in df.columns:
            col_lower = str(col).lower()
            if any(word in col_lower for word in ['product', 'variety', 'flower', 'name']):
                item['product_name'] = str(row[col]) if pd.notna(row[col]) else ''
            elif any(word in col_lower for word in ['box', 'carton']):
                item['boxes'] = int(row[col]) if pd.notna(row[col]) else 1
            elif any(word in col_lower for word in ['pack', 'rate']):
                item['pack_rate'] = int(row[col]) if pd.notna(row[col]) else 100
            elif any(word in col_lower for word in ['price', 'rate']):
                item['unit_price'] = float(row[col]) if pd.notna(row[col]) else 0
            elif any(word in col_lower for word in ['quantity', 'qty', 'stem']):
                item['quantity'] = int(row[col]) if pd.notna(row[col]) else 0
        if item.get('product_name'):
            items.append(item)
    return items

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_content))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_image(file_content: bytes) -> str:
    image = Image.open(io.BytesIO(file_content))
    text = pytesseract.image_to_string(image)
    return text

def parse_order_text(text: str) -> List[Dict]:
    lines = text.strip().split('\n')
    items = []
    current_item = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to parse line
        # Pattern: Product name, maybe with specification, pack rate, boxes, price
        # Example: "Hyndrangea Pink 50cm packrate 60 3bxs price 1.65"
        
        # Extract product name (up to numbers or keywords)
        name_match = re.match(r'^([a-zA-Z\s\-]+?)(?:\s+\d+cm|\s+packrate|\s+\d+bx|\s+price)', line)
        if name_match:
            current_item['product_name'] = name_match.group(1).strip()
        
        # Extract pack rate
        pack_match = re.search(r'packrate\s+(\d+)', line, re.IGNORECASE)
        if pack_match:
            current_item['pack_rate'] = int(pack_match.group(1))
        
        # Extract boxes
        boxes_match = re.search(r'(\d+)\s*bx', line, re.IGNORECASE)
        if boxes_match:
            current_item['boxes'] = int(boxes_match.group(1))
        
        # Extract price
        price_match = re.search(r'price\s+([\d.]+)', line, re.IGNORECASE)
        if price_match:
            current_item['unit_price'] = float(price_match.group(1))
        
        # Extract length
        length_match = re.search(r'(\d+)\s*cm', line, re.IGNORECASE)
        if length_match:
            current_item['specification'] = {'length': length_match.group(1) + 'cm'}
        
        # If we have a product name and price, save it
        if current_item.get('product_name') and current_item.get('unit_price'):
            items.append(current_item.copy())
            current_item = {}
    
    return items

def match_products(items: List[Dict], company_products: List[Dict]) -> List[Dict]:
    matched_items = []
    product_names = [p['name'].lower() for p in company_products]
    
    for item in items:
        best_match = None
        best_score = 0
        
        for product in company_products:
            score = fuzz.ratio(item['product_name'].lower(), product['name'].lower())
            # Check aliases
            for alias in product.get('aliases', []):
                alias_score = fuzz.ratio(item['product_name'].lower(), alias.lower())
                if alias_score > score:
                    score = alias_score
            
            if score > best_score and score > 70:
                best_score = score
                best_match = product
        
        if best_match:
            matched_items.append({
                **item,
                'product_id': best_match['id'],
                'product_name': best_match['name'],
                'match_confidence': best_score / 100
            })
        else:
            matched_items.append({
                **item,
                'product_id': None,
                'match_confidence': 0
            })
    
    return matched_items

@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    company_id: int = 0,
    file_type: str = None
):
    try:
        content = await file.read()
        text = ""
        items = []
        
        # Determine file type
        if file_type:
            file_type = file_type.lower()
        else:
            file_type = file.filename.split('.')[-1].lower()
        
        if file_type in ['xlsx', 'xls']:
            items = extract_text_from_excel(content)
        elif file_type == 'pdf':
            text = extract_text_from_pdf(content)
            items = parse_order_text(text)
        elif file_type in ['docx', 'doc']:
            text = extract_text_from_docx(content)
            items = parse_order_text(text)
        elif file_type in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            text = extract_text_from_image(content)
            items = parse_order_text(text)
        elif file_type == 'csv':
            # Parse CSV
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            for _, row in df.iterrows():
                item = {}
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(word in col_lower for word in ['product', 'variety', 'name']):
                        item['product_name'] = str(row[col]) if pd.notna(row[col]) else ''
                    elif any(word in col_lower for word in ['price', 'rate']):
                        item['unit_price'] = float(row[col]) if pd.notna(row[col]) else 0
                    elif any(word in col_lower for word in ['quantity', 'qty']):
                        item['quantity'] = int(row[col]) if pd.notna(row[col]) else 0
                if item.get('product_name'):
                    items.append(item)
        else:
            # Try as plain text
            text = content.decode('utf-8')
            items = parse_order_text(text)
        
        # Calculate totals
        total_boxes = 0
        total_quantity = 0
        total_amount = 0
        
        for item in items:
            item['quantity'] = item.get('quantity', item.get('boxes', 1) * item.get('pack_rate', 100))
            item['total'] = item.get('quantity', 0) * item.get('unit_price', 0)
            total_boxes += item.get('boxes', 0)
            total_quantity += item.get('quantity', 0)
            total_amount += item.get('total', 0)
        
        return {
            'success': True,
            'items': items,
            'total_boxes': total_boxes,
            'total_quantity': total_quantity,
            'total_amount': total_amount,
            'text_extracted': text[:500] if text else '',
            'item_count': len(items)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match-products")
async def match_products_endpoint(items: List[Dict], company_products: List[Dict]):
    try:
        matched = match_products(items, company_products)
        return {
            'success': True,
            'items': matched
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {'status': 'healthy', 'service': 'Smart Import Engine'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
