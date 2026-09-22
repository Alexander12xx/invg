"""
smart-engine/tools_engine.py — Altech Document Tools Engine
Open-source alternative to iLovePDF / Smallpdf / PDF24.
Mounts all tool endpoints onto the main FastAPI app.

Usage (in existing app.py):
    from tools_engine import mount_tools
    mount_tools(app)
"""

from __future__ import annotations
import io
import os
import tempfile
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# --- Feature modules (new files, all open-source) ---
from pdf_ops import (
    merge_pdfs, split_pdf, compress_pdf, rotate_pdf,
    add_watermark, get_pdf_info,
)
from pdf_to_word import convert_pdf_to_word
from office_convert import convert_office_to_pdf
from image_tools import images_to_pdf, optimize_image
from ocr_engine import ocr_pdf, ocr_image
from security_tools import encrypt_pdf, decrypt_pdf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_MB       = int(os.getenv("TOOLS_MAX_FILE_MB", "15"))
MAX_FILE_BYTES    = MAX_FILE_MB * 1024 * 1024
MAX_FILES_BATCH   = 20
STREAM_CHUNK      = 1024 * 256  # 256 KB

DISPOSITION = 'attachment; filename="{name}"'


def _safe_name(name: str, default: str = "file") -> str:
    name = (name or default).strip().replace("\\", "/").split("/")[-1]
    name = "".join(c for c in name if c.isprintable() and c not in '"<>|')
    return name or default


def _too_large(f: UploadFile) -> bool:
    # Starlette exposes size after read; but we also check header
    try:
        pos = f.file.tell()
        f.file.seek(0, 2)
        size = f.file.tell()
        f.file.seek(pos)
        return size > MAX_FILE_BYTES
    except Exception:
        return False


def _stream(data: bytes, media_type: str, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={
            "Content-Disposition": DISPOSITION.format(name=filename),
            "X-Engine": "Altech-Tools/1.0",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Mount all endpoints
# ---------------------------------------------------------------------------
def mount_tools(app: FastAPI) -> None:
    """Attach every tool endpoint under /tools/*"""

    PREFIX = "/tools"

    # -------- health --------
    @app.get(f"{PREFIX}/health")
    async def tools_health():
        return {
            "ok": True,
            "engine": "tools",
            "version": "1.0.0",
            "max_file_mb": MAX_FILE_MB,
            "max_files": MAX_FILES_BATCH,
            "features": [
                "merge", "split", "compress", "rotate", "watermark",
                "pdf-to-word", "office-to-pdf", "image-to-pdf",
                "ocr", "protect", "unlock",
            ],
        }

    # -------- 1. MERGE --------
    @app.post(f"{PREFIX}/merge")
    async def ep_merge(files: List[UploadFile] = File(...)):
        if len(files) < 2:
            raise HTTPException(400, "At least 2 PDF files required")
        if len(files) > MAX_FILES_BATCH:
            raise HTTPException(400, f"Maximum {MAX_FILES_BATCH} files per merge")

        blobs: list[bytes] = []
        for f in files:
            if _too_large(f):
                raise HTTPException(413, f"{f.filename} exceeds {MAX_FILE_MB} MB")
            blobs.append(await f.read())

        try:
            merged = merge_pdfs(blobs)
        except Exception as e:
            raise HTTPException(500, f"Merge failed: {e}")

        return _stream(merged, "application/pdf", "merged.pdf")

    # -------- 2. SPLIT --------
    @app.post(f"{PREFIX}/split")
    async def ep_split(
        file: UploadFile = File(...),
        mode: str = Form("ranges"),           # ranges | every | extract
        ranges: str = Form(""),               # e.g. "1-3,5,7-9"
        every: int = Form(0),                 # split every N pages
    ):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        try:
            zip_bytes = split_pdf(
                data,
                mode=mode,
                ranges=ranges,
                every=every,
            )
        except Exception as e:
            raise HTTPException(500, f"Split failed: {e}")

        return _stream(zip_bytes, "application/zip", "split-pages.zip")

    # -------- 3. COMPRESS --------
    @app.post(f"{PREFIX}/compress")
    async def ep_compress(
        file: UploadFile = File(...),
        level: str = Form("medium"),          # low | medium | high
    ):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        try:
            out = compress_pdf(data, level=level)
        except Exception as e:
            raise HTTPException(500, f"Compress failed: {e}")

        return _stream(out, "application/pdf", "compressed.pdf")

    # -------- 4. ROTATE --------
    @app.post(f"{PREFIX}/rotate")
    async def ep_rotate(
        file: UploadFile = File(...),
        angle: int = Form(90),                # 90 | 180 | 270
        pages: str = Form("all"),             # "all" or "1,3,5"
    ):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        try:
            out = rotate_pdf(data, angle=angle, pages=pages)
        except Exception as e:
            raise HTTPException(500, f"Rotate failed: {e}")

        return _stream(out, "application/pdf", "rotated.pdf")

    # -------- 5. WATERMARK --------
    @app.post(f"{PREFIX}/watermark")
    async def ep_watermark(
        file: UploadFile = File(...),
        text: str = Form(...),
        opacity: float = Form(0.25),
        angle: int = Form(45),
        font_size: int = Form(48),
        color: str = Form("#999999"),
    ):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        try:
            out = add_watermark(
                data, text=text, opacity=opacity,
                angle=angle, font_size=font_size, color=color,
            )
        except Exception as e:
            raise HTTPException(500, f"Watermark failed: {e}")

        return _stream(out, "application/pdf", "watermarked.pdf")

    # -------- 6. PDF → WORD --------
    @app.post(f"{PREFIX}/pdf-to-word")
    async def ep_pdf_to_word(file: UploadFile = File(...)):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        try:
            docx_bytes = convert_pdf_to_word(data)
        except Exception as e:
            raise HTTPException(500, f"PDF→Word failed: {e}")

        return _stream(
            docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "converted.docx",
        )

    # -------- 7. OFFICE → PDF --------
    @app.post(f"{PREFIX}/office-to-pdf")
    async def ep_office_to_pdf(file: UploadFile = File(...)):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        try:
            pdf_bytes = convert_office_to_pdf(data, filename=_safe_name(file.filename))
        except Exception as e:
            raise HTTPException(500, f"Office→PDF failed: {e}")

        return _stream(pdf_bytes, "application/pdf", "converted.pdf")

    # -------- 8. IMAGE → PDF --------
    @app.post(f"{PREFIX}/image-to-pdf")
    async def ep_image_to_pdf(
        files: List[UploadFile] = File(...),
        page_size: str = Form("A4"),
        orientation: str = Form("portrait"),
        margin: int = Form(20),
    ):
        if not files:
            raise HTTPException(400, "No images uploaded")
        if len(files) > MAX_FILES_BATCH:
            raise HTTPException(400, f"Max {MAX_FILES_BATCH} images")

        blobs: list[bytes] = []
        for f in files:
            if _too_large(f):
                raise HTTPException(413, f"{f.filename} exceeds {MAX_FILE_MB} MB")
            blobs.append(await f.read())

        try:
            pdf_bytes = images_to_pdf(
                blobs,
                page_size=page_size,
                orientation=orientation,
                margin=margin,
            )
        except Exception as e:
            raise HTTPException(500, f"Image→PDF failed: {e}")

        return _stream(pdf_bytes, "application/pdf", "images.pdf")

    # -------- 9. OCR --------
    @app.post(f"{PREFIX}/ocr")
    async def ep_ocr(
        file: UploadFile = File(...),
        lang: str = Form("eng"),
        output: str = Form("txt"),            # txt | searchable-pdf
    ):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        fname = _safe_name(file.filename, "scan.pdf")
        is_pdf = fname.lower().endswith(".pdf")

        try:
            if is_pdf:
                result = ocr_pdf(data, lang=lang, output=output)
            else:
                result = ocr_image(data, lang=lang)

            if isinstance(result, bytes):
                if output == "searchable-pdf":
                    return _stream(result, "application/pdf", "searchable.pdf")
                return _stream(result, "text/plain", "extracted.txt")
            return JSONResponse({"text": result})
        except Exception as e:
            raise HTTPException(500, f"OCR failed: {e}")

    # -------- 10. PROTECT (Encrypt) --------
    @app.post(f"{PREFIX}/protect")
    async def ep_protect(
        file: UploadFile = File(...),
        password: str = Form(...),
        allow_print: bool = Form(True),
        allow_copy: bool = Form(False),
    ):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")
        if len(password) < 4:
            raise HTTPException(400, "Password must be at least 4 characters")

        data = await file.read()
        try:
            out = encrypt_pdf(
                data, password=password,
                allow_print=allow_print, allow_copy=allow_copy,
            )
        except Exception as e:
            raise HTTPException(500, f"Encrypt failed: {e}")

        return _stream(out, "application/pdf", "protected.pdf")

    # -------- 11. UNLOCK (Decrypt) --------
    @app.post(f"{PREFIX}/unlock")
    async def ep_unlock(
        file: UploadFile = File(...),
        password: str = Form(...),
    ):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")

        data = await file.read()
        try:
            out = decrypt_pdf(data, password=password)
        except Exception as e:
            raise HTTPException(400, f"Unlock failed: {e}")

        return _stream(out, "application/pdf", "unlocked.pdf")

    # -------- 12. PDF INFO --------
    @app.post(f"{PREFIX}/info")
    async def ep_info(file: UploadFile = File(...)):
        if _too_large(file):
            raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB")
        data = await file.read()
        return JSONResponse(get_pdf_info(data))
