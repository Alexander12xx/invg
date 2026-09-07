"""
Smart Document Intelligence Engine
For Flower Business SaaS
ALTECH SOFTWARE DEVELOPERS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Flower Business Smart Import Engine",
    description="AI-powered document intelligence for flower businesses",
    version="2.0.0"
)

# CORS Configuration - Allow InfinityFree to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://altechtools.ct.ws",
        "https://*.ct.ws",
        "https://*.infinityfree.com",
        "http://localhost:*",
        "*"  # Remove this in production, use specific domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
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
    text_extracted: str = ""

class MatchRequest(BaseModel):
    items: List[Dict]
    company_products: List[Dict]

# Helper Functions
def extract_text_from_excel(file_content: bytes) -> List[Dict]:
    """Extract order data from Excel file"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        items = []
        
        # Try to detect column names
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if any(word in col_lower for word in ['product', 'variety', 'flower', 'name', 'item']):
                col_map['product'] = col
            elif any(word in col_lower for word in ['box', 'carton', 'case']):
                col_map['boxes'] = col
            elif any(word in col_lower for word in ['pack', 'rate', 'bunch']):
                col_map['pack_rate'] = col
            elif any(word in col_lower for word in ['price', 'rate', 'cost']):
                col_map['price'] = col
            elif any(word in col_lower for word in ['quantity', 'qty', 'stem']):
                col_map['quantity'] = col
            elif any(word in col_lower for word in ['length', 'size', 'cm']):
                col_map['length'] = col
        
        for _, row in df.iterrows():
            item = {}
            if 'product' in col_map and pd.notna(row[col_map['product']]):
                item['product_name'] = str(row[col_map['product']]).strip()
            else:
                # Try to use first column as product name
                first_col = df.columns[0]
                if pd.notna(row[first_col]):
                    item['product_name'] = str(row[first_col]).strip()
            
            if 'boxes' in col_map and pd.notna(row[col_map['boxes']]):
                item['boxes'] = int(row[col_map['boxes']]) if str(row[col_map['boxes']]).isdigit() else 1
            else:
                item['boxes'] = 1
                
            if 'pack_rate' in col_map and pd.notna(row[col_map['pack_rate']]):
                item['pack_rate'] = int(row[col_map['pack_rate']]) if str(row[col_map['pack_rate']]).isdigit() else 100
            
            if 'price' in col_map and pd.notna(row[col_map['price']]):
                try:
                    item['unit_price'] = float(str(row[col_map['price']]).replace(',', ''))
                except:
                    item['unit_price'] = 0
            
            if 'quantity' in col_map and pd.notna(row[col_map['quantity']]):
                try:
                    item['quantity'] = int(row[col_map['quantity']])
                except:
                    item['quantity'] = item.get('boxes', 1) * item.get('pack_rate', 100)
            
            if 'length' in col_map and pd.notna(row[col_map['length']]):
                item['specification'] = {'length': str(row[col_map['length']]) + 'cm'}
            
            if item.get('product_name'):
                # Calculate quantity if not provided
                if 'quantity' not in item:
                    item['quantity'] = item.get('boxes', 1) * item.get('pack_rate', 100)
                items.append(item)
        
        return items
    except Exception as e:
        logger.error(f"Excel parsing error: {e}")
        return []

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from Word document"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for para in doc.paragraphs:
            if para.text:
                text += para.text + "\n"
        return text
    except Exception as e:
        logger.error(f"DOCX parsing error: {e}")
        return ""

def extract_text_from_image(file_content: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(io.BytesIO(file_content))
        # Preprocess image for better OCR
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Try Tesseract OCR
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            if text.strip():
                return text
        except:
            pass
        
        # Fallback: return empty
        return ""
    except Exception as e:
        logger.error(f"Image OCR error: {e}")
        return ""

def parse_order_text(text: str) -> List[Dict]:
    """Parse order text into structured items"""
    lines = text.strip().split('\n')
    items = []
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
            
        # Skip header lines
        if any(word in line.lower() for word in ['product', 'item', 'description', 'qty', 'price', 'total', 'subtotal', 'invoice', 'date']):
            if len(line.split()) < 3:
                continue
        
        item = {}
        
        # Try to extract product name (words before numbers or keywords)
        # Pattern: Product name, maybe with specification, pack rate, boxes, price
        # Example: "Hydrangea Pink 50cm packrate 60 3bxs price 1.65"
        
        # Extract length/size
        length_match = re.search(r'(\d+)\s*cm', line, re.IGNORECASE)
        if length_match:
            item['specification'] = {'length': length_match.group(1) + 'cm'}
        
        # Extract pack rate
        pack_match = re.search(r'packrate?\s*(\d+)', line, re.IGNORECASE)
        if pack_match:
            item['pack_rate'] = int(pack_match.group(1))
        
        # Extract boxes
        boxes_match = re.search(r'(\d+)\s*bx', line, re.IGNORECASE)
        if not boxes_match:
            boxes_match = re.search(r'(\d+)\s*box', line, re.IGNORECASE)
        if boxes_match:
            item['boxes'] = int(boxes_match.group(1))
        
        # Extract price
        price_match = re.search(r'price\s*([\d.]+)', line, re.IGNORECASE)
        if not price_match:
            price_match = re.search(r'@\s*([\d.]+)', line)
        if not price_match:
            price_match = re.search(r'([\d.]+)\s*per\s*stem', line, re.IGNORECASE)
        if price_match:
            item['unit_price'] = float(price_match.group(1))
        
        # Extract product name (remove all the extracted parts)
        product_name = line
        for pattern in [r'packrate?\s*\d+', r'\d+\s*bx', r'price\s*[\d.]+', r'@\s*[\d.]+', r'\d+\s*cm', r'per\s*stem']:
            product_name = re.sub(pattern, '', product_name, flags=re.IGNORECASE)
        
        # Clean up product name
        product_name = re.sub(r'\s+', ' ', product_name).strip()
        if product_name and len(product_name) > 2:
            item['product_name'] = product_name
        
        # If we have a product name, save it
        if item.get('product_name'):
            # Set defaults
            item['boxes'] = item.get('boxes', 1)
            item['pack_rate'] = item.get('pack_rate', 100)
            item['unit_price'] = item.get('unit_price', 0)
            item['quantity'] = item.get('boxes', 1) * item.get('pack_rate', 100)
            items.append(item)
    
    return items

def match_products(items: List[Dict], company_products: List[Dict]) -> List[Dict]:
    """Match extracted items to company products using fuzzy matching"""
    matched_items = []
    
    if not company_products:
        return items
    
    product_names = [p.get('name', '').lower() for p in company_products]
    product_aliases = {}
    
    for product in company_products:
        aliases = [product.get('name', '').lower()]
        if 'aliases' in product:
            aliases.extend([a.lower() for a in product['aliases']])
        product_aliases[product['id']] = aliases
    
    for item in items:
        best_match = None
        best_score = 0
        item_name = item.get('product_name', '').lower()
        
        for product in company_products:
            # Check product name
            score = fuzz.ratio(item_name, product.get('name', '').lower())
            
            # Check aliases
            for alias in product_aliases.get(product['id'], []):
                alias_score = fuzz.ratio(item_name, alias)
                if alias_score > score:
                    score = alias_score
            
            # Check partial matching
            if score < 70:
                # Check if product name is contained in item name
                if product.get('name', '').lower() in item_name:
                    score = 75
                # Check if item name is contained in product name
                elif item_name in product.get('name', '').lower():
                    score = 75
            
            if score > best_score and score >= 60:
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

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Flower Business Smart Import Engine",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": [
            "/api/health",
            "/api/analyze (POST)",
            "/api/match-products (POST)"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "smart-import-engine",
        "version": "2.0.0",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    company_id: int = Form(0),
    file_type: Optional[str] = Form(None)
):
    """
    Analyze uploaded document and extract order data
    Supports: Excel, PDF, Word, Images, CSV, Text
    """
    try:
        # Read file
        content = await file.read()
        
        # Determine file type
        if not file_type:
            file_type = file.filename.split('.')[-1].lower()
        
        logger.info(f"Processing file: {file.filename} ({file_type}) for company {company_id}")
        
        items = []
        text_extracted = ""
        
        # Process based on file type
        if file_type in ['xlsx', 'xls']:
            items = extract_text_from_excel(content)
            logger.info(f"Extracted {len(items)} items from Excel")
            
        elif file_type == 'pdf':
            text_extracted = extract_text_from_pdf(content)
            items = parse_order_text(text_extracted)
            logger.info(f"Extracted {len(items)} items from PDF")
            
        elif file_type in ['docx', 'doc']:
            text_extracted = extract_text_from_docx(content)
            items = parse_order_text(text_extracted)
            logger.info(f"Extracted {len(items)} items from DOCX")
            
        elif file_type in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            text_extracted = extract_text_from_image(content)
            items = parse_order_text(text_extracted)
            logger.info(f"Extracted {len(items)} items from Image")
            
        elif file_type == 'csv':
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                # Use Excel parser logic
                items = extract_text_from_excel(content)
            except:
                # Try as text
                text_extracted = content.decode('utf-8')
                items = parse_order_text(text_extracted)
                
        elif file_type == 'txt':
            text_extracted = content.decode('utf-8')
            items = parse_order_text(text_extracted)
            
        else:
            # Try as plain text
            try:
                text_extracted = content.decode('utf-8')
                items = parse_order_text(text_extracted)
            except:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Calculate totals
        total_boxes = 0
        total_quantity = 0
        total_amount = 0
        
        for item in items:
            # Ensure quantity is set
            if 'quantity' not in item or not item['quantity']:
                item['quantity'] = item.get('boxes', 1) * item.get('pack_rate', 100)
            
            # Calculate total
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
            'text_extracted': text_extracted[:500] if text_extracted else '',
            'item_count': len(items),
            'file_type': file_type,
            'filename': file.filename
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match-products")
async def match_products_endpoint(request: MatchRequest):
    """
    Match extracted items to company products
    """
    try:
        items = request.items
        company_products = request.company_products
        
        if not items:
            return {'success': True, 'items': []}
        
        if not company_products:
            return {'success': True, 'items': items}
        
        matched = match_products(items, company_products)
        
        return {
            'success': True,
            'items': matched,
            'matched_count': sum(1 for i in matched if i.get('product_id'))
        }
        
    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract-text")
async def extract_text(
    file: UploadFile = File(...)
):
    """
    Extract raw text from document without parsing
    """
    try:
        content = await file.read()
        file_type = file.filename.split('.')[-1].lower()
        
        text = ""
        
        if file_type in ['pdf']:
            text = extract_text_from_pdf(content)
        elif file_type in ['docx', 'doc']:
            text = extract_text_from_docx(content)
        elif file_type in ['jpg', 'jpeg', 'png', 'gif']:
            text = extract_text_from_image(content)
        elif file_type in ['txt']:
            text = content.decode('utf-8')
        else:
            text = content.decode('utf-8', errors='ignore')
        
        return {
            'success': True,
            'text': text,
            'length': len(text),
            'file_type': file_type
        }
        
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
