# Smart Document Intelligence Engine v10

A free/open-source, AI-like document extraction engine designed to replace the brittle v6/v9 parser.

## What changed

The old engine was mostly a sequence of regex/table rules. The new engine uses an ensemble:

1. File-type routing
2. PDF text extraction
3. PDF coordinate/layout extraction
4. OCR fallback for scanned PDFs/images
5. DOCX paragraph + table extraction
6. Spreadsheet header discovery
7. Semantic field-label matching
8. Table reconstruction
9. Free-form line parsing
10. Conservative recovery pass
11. Candidate reconciliation/deduplication
12. Arithmetic validation
13. Confidence scoring + provenance
14. Product alias matching with ambiguity margin

### Important safety improvement

Do **not** force the engine to return a fake item just to avoid `item_count = 0`.

If a document is readable but no row is supported by enough evidence, v10 returns zero items plus:

- `review_required: true`
- `diagnostics.status: readable_document_but_no_high_confidence_items`
- `diagnostics.review_reason`

This is safer than inventing a product or assigning a number to the wrong field.

## Endpoints

- `GET /api/ping`
- `GET /api/health`
- `POST /api/analyze`
- `POST /api/extract-text`
- `POST /api/match-products`

## Run locally

Install Tesseract OCR separately on the machine/container, then:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

For Render, use the included `render.yaml`.

## Tesseract on Linux

On Debian/Ubuntu based images:

```bash
apt-get update && apt-get install -y tesseract-ocr
```

If your Render environment does not include Tesseract automatically, use a Docker deployment with a system package layer.

## Example pasted text

The engine can understand lines such as:

Hyndrangea pink 50cm packrate 60 3bxs price 1.65
Hyndrangea julita pink 50cm packrate 60 2bx price 1.65

It can also work with labels:

Flower | Length | Pack Rate | Boxes | Total Stems | Unit Price | Amount
Hydrangea Pink | 50cm | 60 | 3 | 180 | 1.65 | 297

## Product matching

Keep product aliases in your tenant database. Never treat every fuzzy similarity as a match. v10 requires both a score threshold and a separation margin from the second-best product.

This is important for names that are intentionally different, such as Celocia vs Celosia.

## Multi-tenant security note

The intelligence engine does not own tenant data. The PHP/MySQL application should pass the authenticated company/tenant context and only send that tenant's product catalog to `/api/match-products`.

Do not trust `company_id` from an unauthenticated browser as an authorization mechanism. Tenant authorization belongs in the main PHP application/API layer.
