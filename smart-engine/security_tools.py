"""smart-engine/security_tools.py — Encrypt / Decrypt PDFs (pypdf)."""

from __future__ import annotations
import io
from pypdf import PdfReader, PdfWriter


def encrypt_pdf(
    data: bytes,
    password: str,
    allow_print: bool = True,
    allow_copy: bool = False,
) -> bytes:
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    perms = 0
    if allow_print:
        perms |= 0b0000000000000100  # print
        perms |= 0b0000000000001000  # high-res print
    if allow_copy:
        perms |= 0b0000000000010000  # extract

    writer.encrypt(user_password=password, owner_password=password, permissions_flag=perms)
    out = io.BytesIO(); writer.write(out)
    return out.getvalue()


def decrypt_pdf(data: bytes, password: str) -> bytes:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        ok = reader.decrypt(password)
        if not ok:
            raise ValueError("Wrong password")
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    out = io.BytesIO(); writer.write(out)
    return out.getvalue()
