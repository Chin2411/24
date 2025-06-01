from __future__ import annotations

from pathlib import Path

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from docx import Document
from PIL import Image
import pytesseract
import fitz
from io import BytesIO

from config import PDF_IMAGE_DPI


def _pdf_preview(path: Path) -> str:
    """Return text preview for PDF using several fallbacks.

    The function tries multiple extraction methods in the following order:
    1. ``PyPDF2`` text layer extraction for the first few pages.
    2. ``pdfminer.six`` extraction if PyPDF2 returned no text.
    3. ``PyMuPDF`` OCR for scanned pages using ``pytesseract``.

    Raises ``RuntimeError`` if no text could be extracted.
    """

    PAGE_LIMIT = 3
    MAX_CHARS = 2000

    last_error = ""

    # --- Try PyPDF2 ------------------------------------------------------
    try:
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages[:PAGE_LIMIT]:
            t = page.extract_text() or ""
            text += t
            if len(text) >= MAX_CHARS:
                text = text[:MAX_CHARS]
                break
        if text.strip():
            return text.strip()
        last_error = "PyPDF2 не нашёл текст"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка PyPDF2: {exc}"

    # --- Try pdfminer ----------------------------------------------------
    try:
        text = pdfminer_extract_text(str(path), page_numbers=range(PAGE_LIMIT)) or ""
        text = text[:MAX_CHARS]
        if text.strip():
            return text.strip()
        last_error = "pdfminer не нашёл текст"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка pdfminer: {exc}"

    # --- OCR fallback using PyMuPDF + pytesseract -----------------------
    try:
        doc = fitz.open(str(path))
        text = ""
        for page in doc[:PAGE_LIMIT]:
            pix = page.get_pixmap(dpi=PDF_IMAGE_DPI)
            img = Image.open(BytesIO(pix.tobytes()))
            text += pytesseract.image_to_string(img, lang="rus+eng") + "\n"
            if len(text) >= MAX_CHARS:
                text = text[:MAX_CHARS]
                break
        doc.close()
        if text.strip():
            return text.strip()
        last_error = "Не удалось извлечь текст из скана"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка OCR: {exc}"

    raise RuntimeError(last_error)


def _docx_preview(path: Path) -> str:
    doc = Document(str(path))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
        if len(text) >= 2000:
            text = text[:2000]
            break
    return text.strip()


def _text_preview(path: Path) -> str:
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
    except Exception as exc:
        raise RuntimeError(f"Ошибка чтения файла: {exc}")
    return "".join(lines).strip()


def _image_preview(path: Path) -> str:
    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img)
    except Exception as exc:
        raise RuntimeError(f"Ошибка OCR: {exc}")
    return text.strip()


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SUPPORTED_TEXT = {".txt", ".csv"}
SUPPORTED_DOCS = {".docx", ".doc"}


def extract_preview_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return _pdf_preview(path)
        if ext in SUPPORTED_DOCS:
            return _docx_preview(path)
        if ext in SUPPORTED_TEXT:
            return _text_preview(path)
        if ext in SUPPORTED_IMAGES:
            return _image_preview(path)
    except Exception as exc:
        raise RuntimeError(str(exc))
    raise RuntimeError("Просмотр не поддерживается")
