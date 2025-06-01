from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

import fitz
import pytesseract
from rapidfuzz import fuzz, process

from constants import REFERENCE_NAMES
from config import FUZZY_THRESHOLD, PDF_IMAGE_DPI
from .file_metadata import extract_metadata
from .file_preview import _preprocess_image, _clean_ocr_text

logger = logging.getLogger(__name__)


def _alnum_len(text: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-я0-9]", text))


def _extract_fields(text: str) -> tuple[str, str]:
    number = ""
    name = ""
    m = re.search(r"№\s?[A-ZА-Яа-я0-9-/]+", text)
    if m:
        number = m.group(0)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    refs = []
    for r in REFERENCE_NAMES:
        if r.get("ru"):
            refs.append(r["ru"])
        if r.get("en"):
            refs.append(r["en"])
    best_ref = ""
    best_score = 0
    for line in lines:
        choice = process.extractOne(line, refs, scorer=fuzz.token_set_ratio)
        if choice and choice[1] > best_score:
            best_ref = choice[0]
            best_score = choice[1]
    if best_ref:
        name = best_ref if best_score >= FUZZY_THRESHOLD else best_ref
    return number, name


def _pages_to_check(total: int) -> list[int]:
    idxs = list(range(min(2, total)))
    if total > 2:
        start = max(total - 2, 2)
        idxs += list(range(start, total))
    return sorted(set(idxs))


def smart_pdf_extract(path: str | Path) -> Dict[str, Any]:
    """Intelligently extract text and key fields from a PDF file."""
    p = Path(path)
    result: Dict[str, Any] = {
        "text": "",
        "number": "",
        "name": "",
        "metadata": {
            "file_name": p.name,
            "format": "",
            "language": "",
            "pages": "",
        },
        "error": None,
    }

    count, language, paper = extract_metadata(p)
    result["metadata"].update({"format": paper, "language": language, "pages": count})

    try:
        doc = fitz.open(str(p))
    except Exception as exc:
        err = f"ошибка парсинга PDF: {exc}"
        logger.error(err)
        result["error"] = err
        return result

    page_idxs = _pages_to_check(len(doc))

    text = ""
    for idx in page_idxs:
        try:
            if idx < len(doc):
                text += doc[idx].get_text()
        except Exception as exc:
            logger.error("PyMuPDF text layer error on page %s: %s", idx + 1, exc)
    doc.close()

    cleaned = _clean_ocr_text(text)
    if _alnum_len(cleaned) >= 30:
        result["text"] = cleaned
        num, name = _extract_fields(cleaned)
        result["number"] = num
        result["name"] = name
        return result

    # OCR fallback
    try:
        from pdf2image import convert_from_path
    except Exception as exc:  # pragma: no cover - optional dependency
        err = f"ошибка импорта pdf2image: {exc}"
        logger.error(err)
        result["error"] = err
        return result

    try:
        images = convert_from_path(str(p), dpi=PDF_IMAGE_DPI, page_numbers=[i + 1 for i in page_idxs])
    except Exception as exc:
        err = f"ошибка конвертации PDF: {exc}"
        logger.error(err)
        result["error"] = err
        return result

    ocr_text = ""
    for img in images:
        try:
            proc = _preprocess_image(img)
            txt = pytesseract.image_to_string(proc, lang="rus+eng", config="--psm 6")
            ocr_text += txt + "\n"
        except Exception as exc:
            logger.error("ошибка OCR: %s", exc)
            if not result["error"]:
                result["error"] = f"ошибка OCR: {exc}"

    cleaned = _clean_ocr_text(ocr_text)
    if _alnum_len(cleaned) < 30:
        err = "Не удалось распознать текст документа, проверь исходник"
        logger.error(err)
        result["error"] = err
        return result

    result["text"] = cleaned
    num, name = _extract_fields(cleaned)
    result["number"] = num
    result["name"] = name
    return result

