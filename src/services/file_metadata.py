from __future__ import annotations

from pathlib import Path
from typing import Tuple

from langdetect import detect, LangDetectException
from PyPDF2 import PdfReader
from docx import Document
import logging

logger = logging.getLogger(__name__)


def _detect_lang(text: str) -> str:
    snippet = text[:5000]
    try:
        return detect(snippet)
    except LangDetectException:
        return "-"


def _compare_dims(dims: Tuple[float, float], target: Tuple[float, float], tol: float = 5.0) -> bool:
    return abs(dims[0] - target[0]) <= tol and abs(dims[1] - target[1]) <= tol


def _paper_format_from_dimensions(width_mm: float, height_mm: float) -> str:
    dims = tuple(sorted((round(width_mm), round(height_mm))))
    if _compare_dims(dims, (210, 297)):
        return "A4"
    if _compare_dims(dims, (297, 420)):
        return "A3"
    if _compare_dims(dims, (148, 210)):
        return "A5"
    if _compare_dims(dims, (216, 279)):
        return "Letter"
    return "-"


def _pdf_metadata(path: Path) -> Tuple[str, str, str]:
    try:
        logger.info("Чтение PDF для метаданных: %s", path)
        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        text = ""
        for page in reader.pages[:2]:
            text += page.extract_text() or ""
        language = _detect_lang(text)
        page0 = reader.pages[0]
        width_pt = float(page0.mediabox.width)
        height_pt = float(page0.mediabox.height)
        mm_width = width_pt * 0.3527778
        mm_height = height_pt * 0.3527778
        paper = _paper_format_from_dimensions(mm_width, mm_height)
        result = (str(num_pages), language, paper)
        logger.info(
            "PDF метаданные %s: pages=%s lang=%s paper=%s",
            path,
            result[0],
            result[1],
            result[2],
        )
        return result
    except Exception as exc:
        logger.exception("Ошибка чтения PDF %s", path)
        err = f"Ошибка: {exc}"
        return err, err, err


def _docx_metadata(path: Path) -> Tuple[str, str, str]:
    try:
        logger.info("Чтение DOCX для метаданных: %s", path)
        doc = Document(str(path))
        paragraphs = doc.paragraphs
        text = " ".join(p.text for p in paragraphs[:20])
        language = _detect_lang(text)
        count = str(len(paragraphs))
        try:
            sec = doc.sections[0]
            width_mm = sec.page_width / 914400 * 25.4
            height_mm = sec.page_height / 914400 * 25.4
            paper = _paper_format_from_dimensions(width_mm, height_mm)
        except Exception:
            paper = "-"
        result = (count, language, paper)
        logger.info(
            "DOCX метаданные %s: count=%s lang=%s paper=%s",
            path,
            result[0],
            result[1],
            result[2],
        )
        return result
    except Exception as exc:
        logger.exception("Ошибка чтения DOCX %s", path)
        err = f"Ошибка: {exc}"
        return err, err, err


def _text_metadata(path: Path) -> Tuple[str, str, str]:
    try:
        logger.info("Чтение текстового файла для метаданных: %s", path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        text_sample = " ".join(lines[:50])
        language = _detect_lang(text_sample)
        result = (str(len(lines)), language, "-")
        logger.info(
            "TEXT метаданные %s: lines=%s lang=%s",
            path,
            result[0],
            result[1],
        )
        return result
    except Exception as exc:
        logger.exception("Ошибка чтения текста %s", path)
        err = f"Ошибка: {exc}"
        return err, err, err


def extract_metadata(path: Path) -> Tuple[str, str, str]:
    logger.info("Извлечение метаданных из %s", path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        result = _pdf_metadata(path)
        logger.info("Завершено извлечение метаданных PDF %s", path)
        return result
    if ext in {".docx", ".doc"}:
        result = _docx_metadata(path)
        logger.info("Завершено извлечение метаданных DOCX %s", path)
        return result
    if ext in {".txt", ".csv"}:
        result = _text_metadata(path)
        logger.info("Завершено извлечение метаданных TEXT %s", path)
        return result
    msg = "Неподдерживаемый формат"
    return msg, msg, msg

