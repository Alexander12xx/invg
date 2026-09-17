FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies for Unstructured (all file types) [citation:3][citation:8]
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core
    libmagic1 \
    # PDF + OCR
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    # OpenCV support
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Office documents
    libreoffice \
    pandoc \
    # Build tools
    gcc g++ musl-dev libffi-dev gfortran libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download Unstructured models (layout detection, table detection)
RUN python -c "from unstructured.partition.model_init import initialize; initialize()"
RUN python -c "from unstructured_inference.models.tables import UnstructuredTableTransformerModel; model = UnstructuredTableTransformerModel(); model.initialize('microsoft/table-transformer-structure-recognition')"

COPY *.py .
EXPOSE 8000

CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]
