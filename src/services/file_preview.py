from __future__ import annotations

from pathlib import Path
from io import BytesIO
import logging
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
import re

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from docx import Document
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

import pytesseract
import fitz
import cv2
from rapidfuzz import process, fuzz

from constants import REFERENCE_NAMES

from config import (
    PDF_IMAGE_DPI,
    TEMP_DIR,
    PREVIEW_PAGE_COUNT,
    PREVIEW_PARAGRAPH_COUNT,
)
from utils import fix_row, unpack3


logger = logging.getLogger(__name__)


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Return grayscale numpy array without converting if already gray."""
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _cache_file_name(path: Path) -> Path:
    """Return cache file path for given source path."""
    h = hashlib.md5(str(path.resolve()).encode()).hexdigest()
    return TEMP_DIR / f"{h}.preview.txt"


def _load_cached_preview(path: Path) -> str | None:
    cache_path = _cache_file_name(path)
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read cache %s: %s", cache_path, exc)
    return None


def _save_cached_preview(path: Path, text: str) -> None:
    cache_path = _cache_file_name(path)
    try:
        cache_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to write cache %s: %s", cache_path, exc)


def _deskew_image(img: Image.Image) -> Image.Image:
    """Deskew image using OpenCV."""
    try:
        gray = np.array(img.convert("L"))
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            np.array(img),
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("Deskew failed: %s", exc)
        return img


def _preprocess_image(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = _deskew_image(gray)
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    bw = gray.point(lambda x: 255 if x > 128 else 0, mode="1")
    bbox = bw.getbbox()
    if bbox:
        gray = gray.crop(bbox)
    return gray


def _clean_ocr_text(text: str) -> str:
    """Remove unreadable characters and very short lines from OCR output."""
    import re

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[^\w\s,\.№-]", "", line)
        line = line.strip()
        if len(line) < 2:
            continue
        if re.fullmatch(r"[A-Za-zА-Яа-я]{1,2}", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _valid_text(text: str) -> bool:
    """Return True if text has at least 20 alphanumeric characters."""
    return len(re.findall(r"[A-Za-zА-Яа-я0-9]", text)) >= 20


def _ocr_pytesseract_psm6(img: Image.Image) -> str:
    """Run pytesseract with fixed psm=6 and rus+eng languages."""
    try:
        return pytesseract.image_to_string(img, lang="rus+eng", config="--psm 6")
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("pytesseract error: %s", exc)
        return ""


def _ocr_pytesseract(img: Image.Image) -> str:
    texts = []
    for psm in (6, 3):
        try:
            conf = f"--psm {psm}"
            t = pytesseract.image_to_string(img, lang="rus+eng", config=conf)
            t = _clean_ocr_text(t)
            texts.append(t)
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.error("pytesseract failed for psm %s: %s", psm, exc)
    return max(texts, key=len, default="")


def _ocr_easyocr(img: Image.Image) -> str:
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("EasyOCR import error: %s", exc)
        return ""
    try:
        import numpy as np

        reader = easyocr.Reader(["ru", "en"], gpu=False)
        result = reader.readtext(np.array(img))
        text = "\n".join(r[1] for r in result)
        return _clean_ocr_text(text)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("EasyOCR error: %s", exc)
    return ""


def _ocr_paddle(img: Image.Image) -> str:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("PaddleOCR import error: %s", exc)
        return ""
    try:
        import numpy as np

        ocr = PaddleOCR(use_angle_cls=True, lang="ru")
        result = ocr.ocr(np.array(img), cls=True)
        lines = [line[1][0] for line in result[0]]
        text = "\n".join(lines)
        return _clean_ocr_text(text)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("PaddleOCR error: %s", exc)
    return ""


def _ocr_page(page_bytes: bytes) -> str:
    """OCR helper for a single PDF page."""
    img = Image.open(BytesIO(page_bytes))
    img = _preprocess_image(img)
    text = _ocr_pytesseract_psm6(img)
    return _clean_ocr_text(text)


def _ocr_table_cells(img: Image.Image) -> str:
    """Split table image into cells and run pytesseract on each."""
    cv_img = _to_gray(np.array(img))
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(1, img.height // 100))
    )
    hor_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(1, img.width // 100), 1)
    )
    vert_lines = cv2.dilate(
        cv2.erode(thresh, vert_kernel, iterations=1), vert_kernel, iterations=1
    )
    hor_lines = cv2.dilate(
        cv2.erode(thresh, hor_kernel, iterations=1), hor_kernel, iterations=1
    )
    grid = cv2.bitwise_and(vert_lines, hor_lines)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    boxes.sort(key=lambda b: (b[1], b[0]))
    rows: list[list[tuple[int, int, int, int]]] = []
    current_y = -1
    row: list[tuple[int, int, int, int]] = []
    for box in boxes:
        try:
            x, y, w, h = fix_row(box, 4, 0)
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.error("Failed to unpack box: %s", exc)
            continue
        if current_y == -1 or abs(y - current_y) <= h // 2:
            row.append((x, y, w, h))
            current_y = y
        else:
            rows.append(sorted(row, key=lambda b: b[0]))
            row = [(x, y, w, h)]
            current_y = y
    if row:
        rows.append(sorted(row, key=lambda b: b[0]))
    lines = []
    for row_boxes in rows:
        cell_texts = []
        for box in row_boxes:
            try:
                x, y, w, h = fix_row(box, 4, 0)
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.error("Failed to unpack cell box: %s", exc)
                continue
            cell_img = img.crop((x, y, x + w, y + h))
            txt = _ocr_pytesseract(cell_img).strip().replace("\n", " ")
            cell_texts.append(txt)
        lines.append(",".join(cell_texts))
    return _clean_ocr_text("\n".join(lines))


def _ocr_columns(img: Image.Image) -> str:
    """Detect vertical gaps and run OCR for each column."""
    cv_img = _to_gray(np.array(img))
    _, thresh = cv2.threshold(cv_img, 240, 255, cv2.THRESH_BINARY)
    white_counts = np.sum(thresh == 255, axis=0)
    threshold = int(img.height * 0.95)
    separators: list[int] = []
    start = None
    for i, count in enumerate(white_counts):
        if count >= threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start > 5:
                separators.append((start + i) // 2)
            start = None
    if start is not None and img.width - start > 5:
        separators.append((start + img.width) // 2)
    boundaries = [0] + separators + [img.width]
    columns_text: list[str] = []
    for b0, b1 in zip(boundaries, boundaries[1:]):
        col_img = img.crop((b0, 0, b1, img.height))
        col_img = _preprocess_image(col_img)
        txt = _ocr_pytesseract(col_img).replace("\n", " ").strip()
        columns_text.append(txt)
    return _clean_ocr_text("\n".join(columns_text))


def _extract_tables(path: Path, pages: str) -> tuple[str, str | None, str | None]:
    """Extract tables from PDF pages using pdfplumber and OCR fallbacks."""
    tables_text = ""
    last_error = ""
    row_errors: list[str] = []
    image_path: str | None = None
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover - optional dependency
        last_error = f"Ошибка импорта pdfplumber: {exc}"
        logger.error(last_error)
    else:
        try:
            with pdfplumber.open(str(path)) as pdf:
                page_nums = [int(p) - 1 for p in pages.split(",") if p]
                for p in page_nums:
                    if p >= len(pdf.pages):
                        break
                    page = pdf.pages[p]
                    extracted = False
                    for tbl in page.extract_tables() or []:
                        header = tbl[0] if tbl else []
                        n_cols = len(header)
                        lines: list[str] = []
                        lines.append(",".join(str(c) for c in fix_row(header, n_cols)))
                        for idx, row in enumerate(tbl[1:], start=1):
                            try:
                                fixed = fix_row(row, n_cols)
                            except Exception as exc:  # pragma: no cover - unexpected errors
                                err = f"Page {p + 1}, row {idx}: {exc}"
                                logger.error(err)
                                row_errors.append(err)
                                fixed = ["" for _ in range(n_cols)]
                            lines.append(",".join(str(c) for c in fixed))
                        tables_text += "\n".join(lines) + "\n"
                        extracted = True
                    if not extracted:
                        logger.info("No table found on page %s, trying OCR", p + 1)
                        pil_img = page.to_image(resolution=PDF_IMAGE_DPI).original
                        col_text = _ocr_columns(pil_img)
                        if not col_text.strip():
                            col_text = _ocr_table_cells(_preprocess_image(pil_img))
                        if col_text.strip():
                            tables_text += col_text + "\n"
                            extracted = True
                    if not extracted and image_path is None:
                        image_path = str(Path("logs") / f"{path.stem}_page{p + 1}.png")
                        Path(image_path).parent.mkdir(exist_ok=True)
                        page.to_image(resolution=PDF_IMAGE_DPI).save(image_path)
                        last_error = (
                            f"Page {p + 1}: не удалось корректно распознать таблицу"
                        )
                        logger.error(last_error)
            if tables_text:
                error_msg = "; ".join(row_errors) if row_errors else None
                return tables_text.strip(), None, error_msg
        except Exception as exc:  # pragma: no cover - optional dependency
            last_error = f"Ошибка pdfplumber: {exc}"
            logger.error(last_error)

    # Fallback using OpenCV cell detection
    try:
        tables_text = _extract_tables_cv(path, pages)
        if tables_text:
            return tables_text.strip(), None, None
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"OpenCV table extraction failed: {exc}"
        logger.error(last_error)

    return "", image_path, last_error or None


def _extract_tables_cv(path: Path, pages: str) -> str:
    """Extract table text using OpenCV-based cell detection."""
    tables_text = ""
    page_nums = [int(p) - 1 for p in pages.split(",") if p]
    doc = fitz.open(str(path))
    for p in page_nums:
        if p >= len(doc):
            break
        page = doc[p]
        pix = page.get_pixmap(dpi=PDF_IMAGE_DPI)
        img = Image.open(BytesIO(pix.tobytes()))
        img = _preprocess_image(img)
        text = _ocr_table_cells(img)
        if text.strip():
            tables_text += text + "\n"
    doc.close()
    return tables_text


def _extract_text_layer(path: Path, page_idxs: list[int], max_chars: int) -> str:
    """Try to extract text layer using PyPDF2, pdfminer and PyMuPDF."""
    text = ""
    # PyPDF2
    try:
        reader = PdfReader(str(path))
        for idx in page_idxs:
            if idx >= len(reader.pages):
                break
            page = reader.pages[idx]
            text += page.extract_text() or ""
            if len(text) >= max_chars:
                text = text[:max_chars]
                break
        if _valid_text(text):
            return text
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("PyPDF2 text layer error: %s", exc)

    # pdfminer
    try:
        text = pdfminer_extract_text(str(path), page_numbers=page_idxs) or ""
        text = text[:max_chars]
        if _valid_text(text):
            return text
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("pdfminer text layer error: %s", exc)

    # PyMuPDF text layer
    try:
        doc = fitz.open(str(path))
        txt = ""
        for idx in page_idxs:
            if idx >= len(doc):
                break
            txt += doc[idx].get_text()
            if len(txt) >= max_chars:
                txt = txt[:max_chars]
                break
        doc.close()
        if _valid_text(txt):
            return txt
        return txt
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("PyMuPDF text layer error: %s", exc)
    return text


def _extract_key_fields(text: str) -> dict[str, str]:
    """Find document number, registration number and name in text."""
    lines = [l.strip() for l in text.splitlines() if re.search(r"[A-Za-zА-Яа-я0-9]{3,}", l)]
    number = ""
    reg_number = ""
    name = ""
    number_re = re.compile(r"№\s*([\w/-]+)", re.I)
    reg_re = re.compile(r"регистрац[а-я]*\s*№?\s*([\w/-]+)", re.I)
    refs = [r["ru"] for r in REFERENCE_NAMES if r.get("ru")]
    for line in lines:
        if not number:
            m = number_re.search(line)
            if m:
                number = m.group(1)
        if not reg_number:
            m = reg_re.search(line)
            if m:
                reg_number = m.group(1)
        if not name:
            choice = process.extractOne(line, refs, scorer=fuzz.token_set_ratio)
            if choice and choice[1] > 80:
                name = choice[0]
        if number and reg_number and name:
            break
    return {"number": number, "reg_number": reg_number, "name": name}


def _pdf_preview(path: Path) -> tuple[str, str | None, str | None]:
    """Return text preview for PDF with text-layer and OCR fallbacks."""

    MAX_CHARS = 2000

    try:
        doc = fitz.open(str(path))
        total_pages = len(doc)
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("Failed to open PDF: %s", exc)
        return "", None, f"Ошибка чтения PDF: {exc}"

    page_idxs = list(range(min(PREVIEW_PAGE_COUNT, total_pages)))
    if total_pages > PREVIEW_PAGE_COUNT:
        start = max(total_pages - PREVIEW_PAGE_COUNT, PREVIEW_PAGE_COUNT)
        page_idxs += list(range(start, total_pages))
    page_idxs = sorted(set(page_idxs))

    text = _extract_text_layer(path, page_idxs, MAX_CHARS)
    if _valid_text(text):
        cleaned = _clean_ocr_text(text)[:MAX_CHARS]
        _save_cached_preview(path, cleaned)
        fields = _extract_key_fields(cleaned)
        logger.info("Key fields extracted: %s", fields)
        doc.close()
        return cleaned, None, None

    # OCR fallback
    pages_pix: list[bytes] = []
    for idx in page_idxs:
        if idx >= len(doc):
            break
        page = doc[idx]
        pix = page.get_pixmap(dpi=PDF_IMAGE_DPI)
        pages_pix.append(pix.tobytes())
    doc.close()

    text = ""
    for pb in pages_pix:
        text += _ocr_page(pb) + "\n"
        if len(text) >= MAX_CHARS:
            text = text[:MAX_CHARS]
            break

    cleaned = _clean_ocr_text(text)
    if _valid_text(cleaned):
        _save_cached_preview(path, cleaned)
        fields = _extract_key_fields(cleaned)
        logger.info("Key fields extracted after OCR: %s", fields)
        return cleaned, None, None

    # Still not readable
    image_path = str(Path("logs") / f"{path.stem}_preview.png")
    try:
        Path(image_path).parent.mkdir(exist_ok=True)
        if pages_pix:
            Image.open(BytesIO(pages_pix[0])).save(image_path)
    except Exception as exc:  # pragma: no cover - optional
        logger.error("Failed to save preview image: %s", exc)

    err = "Не удалось распознать текст документа, проверь исходник"
    logger.error(err)
    return "", image_path, err


def _docx_preview(path: Path) -> str:
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    selected = paras[:PREVIEW_PARAGRAPH_COUNT]
    if len(paras) > PREVIEW_PARAGRAPH_COUNT:
        selected += paras[-PREVIEW_PARAGRAPH_COUNT:]
    text = "\n".join(selected)
    return text.strip()[:2000]


def _text_preview(path: Path) -> str:
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            lines.extend(all_lines[:PREVIEW_PARAGRAPH_COUNT])
            if len(all_lines) > PREVIEW_PARAGRAPH_COUNT:
                lines.extend(all_lines[-PREVIEW_PARAGRAPH_COUNT:])
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
            text = _clean_ocr_text(text)
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


def extract_preview(path: Path) -> tuple[str, str | None, str | None]:
    cached = _load_cached_preview(path)
    if cached:
        return cached, None, None

    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            text, img, err = unpack3(_pdf_preview(path))
        elif ext in SUPPORTED_DOCS:
            text, img, err = unpack3((_docx_preview(path), None, None))
        elif ext in SUPPORTED_TEXT:
            text, img, err = unpack3((_text_preview(path), None, None))
        elif ext in SUPPORTED_IMAGES:
            text, img, err = unpack3((_image_preview(path), None, None))
        else:
            raise RuntimeError("Просмотр не поддерживается")
        if text:
            _save_cached_preview(path, text)
        return text, img, err
    except Exception as exc:
        raise RuntimeError(str(exc))


def extract_preview_text(path: Path) -> str:
    text, _, _ = unpack3(extract_preview(path))
    return text
