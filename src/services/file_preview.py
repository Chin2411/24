from __future__ import annotations

from pathlib import Path
from io import BytesIO
import logging

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from docx import Document
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import fitz

from config import PDF_IMAGE_DPI


logger = logging.getLogger(__name__)


def _preprocess_image(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    bw = gray.point(lambda x: 255 if x > 128 else 0, mode="1")
    bbox = bw.getbbox()
    if bbox:
        gray = gray.crop(bbox)
    return gray


def _ocr_pytesseract(img: Image.Image) -> str:
    texts = []
    for psm in (6, 4, 11, 12):
        try:
            conf = f"--psm {psm}"
            t = pytesseract.image_to_string(img, lang="rus+eng", config=conf)
            texts.append(t)
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.error("pytesseract failed for psm %s: %s", psm, exc)
    return max(texts, key=len, default="")


def _ocr_easyocr(img: Image.Image) -> str:
    try:
        import easyocr
        import numpy as np

        reader = easyocr.Reader(["ru", "en"], gpu=False)
        result = reader.readtext(np.array(img))
        return "\n".join(r[1] for r in result)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("EasyOCR error: %s", exc)
    return ""


def _ocr_paddle(img: Image.Image) -> str:
    try:
        from paddleocr import PaddleOCR
        import numpy as np

        ocr = PaddleOCR(use_angle_cls=True, lang="ru")
        result = ocr.ocr(np.array(img), cls=True)
        lines = [line[1][0] for line in result[0]]
        return "\n".join(lines)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("PaddleOCR error: %s", exc)
    return ""


def _extract_tables(path: Path, pages: str) -> str:
    tables_text = ""
    try:
        import camelot

        tables = camelot.read_pdf(str(path), pages=pages)
        for table in tables:
            tables_text += table.df.to_csv(index=False) + "\n"
        if tables_text:
            return tables_text
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("Camelot error: %s", exc)
    try:
        import pdfplumber
        import pandas as pd

        with pdfplumber.open(str(path)) as pdf:
            page_nums = [int(p) - 1 for p in pages.split(",") if p]
            for p in page_nums:
                if p >= len(pdf.pages):
                    break
                page = pdf.pages[p]
                for tbl in page.extract_tables() or []:
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    tables_text += df.to_csv(index=False) + "\n"
        return tables_text
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("pdfplumber error: %s", exc)
    return ""


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
        if len(text.strip()) > 50:
            return text.strip()
        last_error = "PyPDF2 не нашёл текст"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка PyPDF2: {exc}"
        logger.error(last_error)

    # --- Try pdfminer ----------------------------------------------------
    try:
        text = pdfminer_extract_text(str(path), page_numbers=range(PAGE_LIMIT)) or ""
        text = text[:MAX_CHARS]
        if len(text.strip()) > 50:
            return text.strip()
        last_error = "pdfminer не нашёл текст"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка pdfminer: {exc}"
        logger.error(last_error)

    pages = ",".join(str(i + 1) for i in range(PAGE_LIMIT))
    tables = _extract_tables(path, pages)

    # --- OCR fallback using PyMuPDF + pytesseract -----------------------
    try:
        doc = fitz.open(str(path))
        text = ""
        for idx, page in enumerate(doc[:PAGE_LIMIT]):
            pix = page.get_pixmap(dpi=PDF_IMAGE_DPI)
            img = Image.open(BytesIO(pix.tobytes()))
            if idx == 0:
                try:
                    Path("logs").mkdir(exist_ok=True)
                    img.save(Path("logs") / f"{path.stem}_page1.png")
                except Exception as exc:  # pragma: no cover - optional
                    logger.error("Failed to save debug image: %s", exc)
            img = _preprocess_image(img)
            ocr_text = _ocr_pytesseract(img)
            if len(ocr_text.strip()) < 20:
                ocr_text = _ocr_easyocr(img)
            if len(ocr_text.strip()) < 20:
                ocr_text = _ocr_paddle(img)
            text += ocr_text + "\n"
            if len(text) >= MAX_CHARS:
                text = text[:MAX_CHARS]
                break
        doc.close()
        text += tables
        if text.strip():
            return text.strip()
        last_error = "Не удалось извлечь текст из скана"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка OCR: {exc}"
        logger.error(last_error)

    raise RuntimeError(
        last_error
        or "OCR не справился, возможно, сложная таблица или качество недостаточно"
    )


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
            img = _preprocess_image(img)
            text = _ocr_pytesseract(img)
            if len(text.strip()) < 20:
                text = _ocr_easyocr(img)
            if len(text.strip()) < 20:
                text = _ocr_paddle(img)
            if text.strip():
                return text.strip()
            Path("logs").mkdir(exist_ok=True)
            img.save(Path("logs") / f"{Path(path).stem}_debug.png")
            raise RuntimeError(
                "OCR не справился, возможно, сложная таблица или качество недостаточно"
            )
    except Exception as exc:
        logger.error("Ошибка OCR: %s", exc)
        raise RuntimeError(f"Ошибка OCR: {exc}")


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
