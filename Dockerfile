# ============================================================================
#  Altech Smart Document Intelligence Engine  —  Dockerfile
#  Version: 27.1  (adds open-source document tools: LibreOffice, Poppler, …)
# ============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    # LibreOffice headless (used by office_convert.py)
    HOME=/tmp \
    SOFFICE_BIN=soffice \
    SOFFICE_TIMEOUT=120 \
    # Tesseract binary location
    TESSERACT_CMD=/usr/bin/tesseract \
    # Tools engine cap
    TOOLS_MAX_FILE_MB=15

# ---------------------------------------------------------------------------
#  System packages
#  - tesseract-ocr      : OCR for scanned PDFs and images
#  - tesseract-ocr-eng  : English language pack (add more if needed)
#  - poppler-utils      : pdf2image backend (used by some OCR flows)
#  - libreoffice-*      : Word / Excel / PowerPoint  →  PDF conversion
#  - fonts-*            : Proper font rendering inside LibreOffice output
#  - libgl1 / libglib2.0-0 : Pillow & PyMuPDF runtime libraries
#  - ghostscript        : PDF compression helpers (PyMuPDF falls back to it)
#  - curl               : Health-check helpers (optional)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libreoffice \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    fonts-liberation \
    fonts-dejavu-core \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
#  Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
#  Application code
#  - Root-level .py files (app.py, chat_engine.py, detector.py, pipeline.py,
#    unstructured_helper.py, tools_engine.py, pdf_ops.py, pdf_to_word.py,
#    office_convert.py, image_tools.py, ocr_engine.py, security_tools.py, …)
#  - document_engine/ package
# ---------------------------------------------------------------------------
COPY *.py ./
COPY document_engine/ ./document_engine/

# Ensure a writable HOME for LibreOffice / Tesseract on Render
RUN mkdir -p /tmp && chmod 777 /tmp

EXPOSE 8000

# Single worker (free-tier friendly), long keep-alive for cold starts
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]
