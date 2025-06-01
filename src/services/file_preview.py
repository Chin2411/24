from __future__ import annotations

from pathlib import Path
from io import BytesIO
import logging
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from docx import Document
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

import pytesseract
import fitz
import cv2

from config import PDF_IMAGE_DPI, TEMP_DIR


logger = logging.getLogger(__name__)


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
        rotated = cv2.warpAffine(np.array(img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
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


def _ocr_page(page_bytes: bytes) -> str:
    """OCR helper for a single PDF page."""
    img = Image.open(BytesIO(page_bytes))
    img = _preprocess_image(img)
    text = _ocr_pytesseract(img)
    if len(text.strip()) < 20:
        text = _ocr_easyocr(img)
    if len(text.strip()) < 20:
        text = _ocr_paddle(img)
    return text


def _ocr_table_cells(img: Image.Image) -> str:
    """Split table image into cells and run pytesseract on each."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, img.height // 100)))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, img.width // 100), 1))
    vert_lines = cv2.dilate(cv2.erode(thresh, vert_kernel, iterations=1), vert_kernel, iterations=1)
    hor_lines = cv2.dilate(cv2.erode(thresh, hor_kernel, iterations=1), hor_kernel, iterations=1)
    grid = cv2.bitwise_and(vert_lines, hor_lines)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    boxes.sort(key=lambda b: (b[1], b[0]))
    rows: list[list[tuple[int, int, int, int]]] = []
    current_y = -1
    row: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        if current_y == -1 or abs(y - current_y) <= h // 2:
            row.append(box)
            current_y = y
        else:
            rows.append(sorted(row, key=lambda b: b[0]))
            row = [box]
            current_y = y
    if row:
        rows.append(sorted(row, key=lambda b: b[0]))
    lines = []
    for row_boxes in rows:
        cell_texts = []
        for x, y, w, h in row_boxes:
            cell_img = img.crop((x, y, x + w, y + h))
            txt = _ocr_pytesseract(cell_img).strip().replace("\n", " ")
            cell_texts.append(txt)
        lines.append(",".join(cell_texts))
    return "\n".join(lines)


def _ocr_columns(img: Image.Image) -> str:
    """Detect vertical gaps and run OCR for each column."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
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
    return "\n".join(columns_text)


def _extract_tables(path: Path, pages: str) -> tuple[str, str | None, str | None]:
    """Extract tables from PDF pages using pdfplumber and OCR fallbacks."""
    tables_text = ""
    last_error = ""
    row_errors: list[str] = []
    image_path: str | None = None
    try:
        import pdfplumber

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

                    def _fix_row(r: list[str] | None, row_idx: int) -> list[str]:
                        """Return row with exactly ``n_cols`` columns.

                        Any missing values are replaced with an empty string,
                        extra values are discarded.  All errors are logged so
                        processing of other rows can continue.
                        """
                        try:
                            cells = list(r) if r else []
                            if len(cells) < n_cols:
                                cells.extend([""] * (n_cols - len(cells)))
                            elif len(cells) > n_cols:
                                cells = cells[:n_cols]
                            return [str(c) if c is not None else "" for c in cells]
                        except Exception as exc:
                            err = f"Page {p + 1}, row {row_idx}: {exc}"
                            logger.error(err)
                            row_errors.append(err)
                            return ["" for _ in range(n_cols)]

                    lines.append(",".join(_fix_row(header, 0)))
                    for idx, row in enumerate(tbl[1:], start=1):
                        lines.append(",".join(_fix_row(row, idx)))

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
                    last_error = f"Page {p + 1}: не удалось корректно распознать таблицу"
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


def _pdf_preview(path: Path) -> tuple[str, str | None, str | None]:
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

    pages = ",".join(str(i + 1) for i in range(PAGE_LIMIT))

    # --- Try table extraction first ------------------------------------
    table_text, image, table_err = _extract_tables(path, pages)
    if table_text:
        _save_cached_preview(path, table_text)
        return table_text, None, table_err
    if table_err and image:
        return "", image, table_err

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
            return text.strip(), None
        last_error = "PyPDF2 не нашёл текст"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка PyPDF2: {exc}"
        logger.error(last_error)

    # --- Try pdfminer ----------------------------------------------------
    try:
        text = pdfminer_extract_text(str(path), page_numbers=range(PAGE_LIMIT)) or ""
        text = text[:MAX_CHARS]
        if len(text.strip()) > 50:
            return text.strip(), None
        last_error = "pdfminer не нашёл текст"
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка pdfminer: {exc}"
        logger.error(last_error)

    pages = ",".join(str(i + 1) for i in range(PAGE_LIMIT))

    # --- OCR fallback using PyMuPDF + pytesseract -----------------------
    text = ""
    try:
        doc = fitz.open(str(path))
        pages_pix: list[bytes] = []
        for idx, page in enumerate(doc[:PAGE_LIMIT]):
            pix = page.get_pixmap(dpi=PDF_IMAGE_DPI)
            pages_pix.append(pix.tobytes())
            if idx == 0:
                try:
                    Path("logs").mkdir(exist_ok=True)
                    Image.open(BytesIO(pix.tobytes())).save(
                        Path("logs") / f"{path.stem}_page1.png"
                    )
                except Exception as exc:  # pragma: no cover - optional
                    logger.error("Failed to save debug image: %s", exc)
        doc.close()

        start_all = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4) as ex:
            futures = []
            start_times: list[float] = []
            for pb in pages_pix:
                futures.append(ex.submit(_ocr_page, pb))
                start_times.append(time.perf_counter())

            for idx, fut in enumerate(futures):
                ocr_text = fut.result()
                duration = time.perf_counter() - start_times[idx]
                if duration > 5:
                    logger.warning(
                        "OCR page %s took %.2f sec", idx + 1, duration
                    )
                text += ocr_text + "\n"
                if len(text) >= MAX_CHARS:
                    text = text[:MAX_CHARS]
                    break
        total = time.perf_counter() - start_all
        logger.info("OCR of %s pages completed in %.2f sec", len(pages_pix), total)
    except Exception as exc:  # pragma: no cover - unexpected errors
        last_error = f"Ошибка OCR: {exc}"
        logger.error(last_error)

    if len(text.strip()) < 50:
        table_text, _, _ = _extract_tables(path, pages)
        if not table_text:
            table_text = _extract_tables_cv(path, pages)
        text += table_text

    if text.strip():
        _save_cached_preview(path, text.strip())
        return text.strip(), None, None

    image_path = None
    try:
        doc = fitz.open(str(path))
        pix = doc[0].get_pixmap(dpi=PDF_IMAGE_DPI)
        image_path = str(Path("logs") / f"{path.stem}_preview.png")
        Path(image_path).parent.mkdir(exist_ok=True)
        Image.open(BytesIO(pix.tobytes())).save(image_path)
        doc.close()
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("Failed to save preview image: %s", exc)

    logger.error(
        "Не удалось корректно распознать таблицу: %s", last_error or "unknown"
    )
    return "", image_path, last_error or "Не удалось корректно распознать таблицу"


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


def extract_preview(path: Path) -> tuple[str, str | None, str | None]:
    cached = _load_cached_preview(path)
    if cached:
        return cached, None, None

    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            text, img, err = _pdf_preview(path)
        elif ext in SUPPORTED_DOCS:
            text, img, err = _docx_preview(path), None, None
        elif ext in SUPPORTED_TEXT:
            text, img, err = _text_preview(path), None, None
        elif ext in SUPPORTED_IMAGES:
            text, img, err = _image_preview(path), None, None
        else:
            raise RuntimeError("Просмотр не поддерживается")
        if text:
            _save_cached_preview(path, text)
        return text, img, err
    except Exception as exc:
        raise RuntimeError(str(exc))


def extract_preview_text(path: Path) -> str:
    text, _, _ = extract_preview(path)
    return text
