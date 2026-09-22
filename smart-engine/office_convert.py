"""smart-engine/office_convert.py — Word/Excel/PPT → PDF via LibreOffice headless."""

from __future__ import annotations
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

SOFFICE_BIN = os.getenv("SOFFICE_BIN", "soffice")
TIMEOUT     = int(os.getenv("SOFFICE_TIMEOUT", "120"))

ALLOWED = {".doc", ".docx", ".odt", ".rtf", ".txt",
           ".xls", ".xlsx", ".ods", ".csv",
           ".ppt", ".pptx", ".odp"}


def convert_office_to_pdf(data: bytes, filename: str) -> bytes:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"Unsupported office format: {ext}")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, f"input{ext}")
        with open(in_path, "wb") as f:
            f.write(data)

        # Run LibreOffice in headless mode
        cmd = [
            SOFFICE_BIN,
            "--headless", "--norestore", "--nolockcheck",
            "--convert-to", "pdf",
            "--outdir", tmp,
            in_path,
        ]
        try:
            subprocess.run(
                cmd, check=True, timeout=TIMEOUT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"LibreOffice failed: {e.stderr.decode(errors='ignore')}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("LibreOffice timed out")

        pdf_path = os.path.join(tmp, "input.pdf")
        if not os.path.exists(pdf_path):
            raise RuntimeError("Output PDF not produced")

        with open(pdf_path, "rb") as f:
            return f.read()
