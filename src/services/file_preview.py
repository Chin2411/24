from __future__ import annotations

from pathlib import Path

from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract


def _pdf_preview(path: Path) -> str:
    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages[:2]:
        t = page.extract_text() or ""
        text += t
        if len(text) >= 2000:
            text = text[:2000]
            break
    return text.strip()


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
