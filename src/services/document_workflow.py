from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from constants import REFERENCE_NAMES
import pandas as pd
from rapidfuzz import process, fuzz

from config import EXTRACTED_FILES_DIR, FUZZY_THRESHOLD
from services.file_metadata import extract_metadata
from services.file_preview import extract_preview_text
from gui.workers import extract_archive


logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    path: Path
    name: str
    ext: str
    language: str = "-"
    paper: str = "-"
    count: str = "-"
    preview: str = ""
    found_line: str = ""
    reference: str = ""
    number: str = ""
    score: int = 0
    error: str | None = None


class DocumentWorkflow:
    """Simple API for processing document files."""

    def __init__(self, fuzzy_threshold: int = FUZZY_THRESHOLD):
        self.files: list[FileInfo] = []
        self.threshold = fuzzy_threshold

    # ------------------------------------------------------------------
    def add_paths(self, paths: Iterable[str]) -> None:
        """Load files or archives and collect FileInfo objects."""
        for p in paths:
            try:
                path = Path(p)
                if not path.exists():
                    logger.warning("Path %s does not exist", p)
                    continue
                if path.is_dir():
                    logger.info("Skipping directory %s", p)
                    continue
                lower = path.suffix.lower()
                if lower in {
                    ".zip",
                    ".rar",
                    ".7z",
                    ".tar",
                    ".tar.gz",
                    ".tgz",
                    ".tar.bz2",
                    ".tbz2",
                }:
                    dest = EXTRACTED_FILES_DIR / path.stem
                    try:
                        extracted = extract_archive(str(path), dest)
                        self.add_paths(extracted)
                    except Exception as exc:
                        logger.error("Archive %s error: %s", p, exc)
                    continue
                self.files.append(
                    FileInfo(path=path, name=path.name, ext=path.suffix.lstrip("."))
                )
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.error("Failed to add %s: %s", p, exc)

    # ------------------------------------------------------------------
    def extract_metadata(self) -> None:
        for fi in self.files:
            if fi.error:
                continue
            try:
                count, language, paper = extract_metadata(fi.path)
                fi.count = count
                fi.language = language
                fi.paper = paper
                if count.startswith("Ошибка") or count == "Неподдерживаемый формат":
                    fi.error = count
            except Exception as exc:  # pragma: no cover - unexpected errors
                fi.error = str(exc)
                logger.error("Metadata error for %s: %s", fi.path, exc)

    # ------------------------------------------------------------------
    def quick_preview(self) -> None:
        for fi in self.files:
            if fi.error:
                continue
            try:
                fi.preview = extract_preview_text(fi.path)
            except Exception as exc:  # pragma: no cover - unexpected errors
                fi.error = str(exc)
                logger.error("Preview error for %s: %s", fi.path, exc)

    # ------------------------------------------------------------------
    def verify(self) -> None:
        refs = [r["ru"] for r in REFERENCE_NAMES if r.get("ru")]
        if not refs:
            logger.error("REFERENCE_NAMES list is empty")
            for fi in self.files:
                fi.error = fi.error or "Справочник пуст"
            return
        for fi in self.files:
            if fi.error:
                continue
            try:
                best_line = ""
                best_ref = ""
                best_score = 0
                for line in fi.preview.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    choice = process.extractOne(line, refs, scorer=fuzz.token_set_ratio)
                    if choice:
                        ref, score = choice
                        if score > best_score:
                            best_score = score
                            best_line = line
                            best_ref = ref
                import re

                m = re.search(r"№\s*\S+", fi.preview)
                fi.number = m.group(0) if m else ""
                fi.found_line = best_line
                fi.reference = best_ref if best_score >= self.threshold else ""
                fi.score = best_score
                if best_score < self.threshold:
                    fi.error = fi.error or "Наименование не сопоставлено"
            except Exception as exc:  # pragma: no cover - unexpected errors
                fi.error = "Ошибка сверки"
                logger.error("Verification error for %s: %s", fi.path, exc)

    # ------------------------------------------------------------------
    def export_inventory(self, out_path: str | Path) -> None:
        records = []
        for fi in self.files:
            records.append(
                {
                    "Имя файла": fi.name,
                    "Наименование": fi.reference,
                    "KKS": "",
                    "Номер": fi.number,
                    "Язык": fi.language,
                    "Формат": fi.paper,
                    "Страницы": fi.count,
                }
            )
        df = pd.DataFrame(records)
        out_p = Path(out_path)
        if out_p.suffix.lower() in {".xlsx", ".xls"}:
            df.to_excel(out_p, index=False)
        else:
            df.to_csv(out_p, index=False, encoding="utf-8")
