from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import List
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rapidfuzz import process, fuzz

from constants import REFERENCE_NAMES

import rarfile
import py7zr
from PyQt6.QtCore import QThread, pyqtSignal


def extract_archive(archive_path: str, dest_dir: Path) -> List[str]:
    """Extract supported archive formats to destination directory.

    Returns list of extracted file paths.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_lower = archive_path.lower()
    extracted: List[str] = []

    if archive_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                zf.extract(member, dest_dir)
                extracted.append(str(dest_dir / member.filename))
    elif archive_lower.endswith(".rar"):
        with rarfile.RarFile(archive_path) as rf:
            for member in rf.infolist():
                if member.isdir():
                    continue
                rf.extract(member, dest_dir)
                extracted.append(str(dest_dir / member.filename))
    elif archive_lower.endswith(".7z"):
        with py7zr.SevenZipFile(archive_path, mode="r") as sz:
            sz.extractall(path=dest_dir)
            for name in sz.getnames():
                if not name.endswith("/"):
                    extracted.append(str(dest_dir / name))
    elif archive_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.isdir():
                    continue
                tf.extract(member, dest_dir)
                extracted.append(str(dest_dir / member.name))
    else:
        raise ValueError("Неподдерживаемый формат архива")

    return extracted


class ArchiveExtractWorker(QThread):
    """Worker thread for archive extraction."""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, archive_path: str, dest_dir: Path):
        super().__init__()
        self.archive_path = archive_path
        self.dest_dir = dest_dir

    def run(self) -> None:
        try:
            files = extract_archive(self.archive_path, self.dest_dir)
            self.finished.emit(files)
        except Exception as exc:
            self.error.emit(str(exc))


from services.file_metadata import extract_metadata
from services.file_preview import extract_preview, extract_preview_text


class FileMetadataWorker(QThread):
    """Worker thread for extracting metadata from files."""

    result = pyqtSignal(str, str, str, str)  # path, language, paper, pages/lines
    error = pyqtSignal(str, str)  # path, error message

    def __init__(self, files: list[str]):
        super().__init__()
        self.files = files

    def run(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {
                executor.submit(extract_metadata, Path(p)): p for p in self.files
            }
            start_times: dict[str, float] = {
                fut: time.perf_counter() for fut in future_map
            }
            for fut in as_completed(future_map):
                path = future_map[fut]
                duration = time.perf_counter() - start_times[fut]
                try:
                    count, language, paper = fut.result()
                    if duration > 5:
                        logging.getLogger(__name__).warning(
                            "Metadata extraction for %s took %.2f sec", path, duration
                        )
                    if count.startswith("Ошибка") or count == "Неподдерживаемый формат":
                        self.error.emit(path, count)
                    self.result.emit(path, language, paper, count)
                except Exception as exc:
                    self.error.emit(path, str(exc))


class FilePreviewWorker(QThread):
    """Worker thread for extracting file preview text or image."""

    finished = pyqtSignal(str, str)  # path, preview text
    imageReady = pyqtSignal(str, str)  # path, image path
    error = pyqtSignal(str, str)  # path, error message

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self) -> None:
        start = time.perf_counter()
        try:
            text, image, err = extract_preview(Path(self.path))
            duration = time.perf_counter() - start
            if duration > 5:
                logging.getLogger(__name__).warning(
                    "Preview extraction for %s took %.2f sec", self.path, duration
                )
            if image:
                self.imageReady.emit(self.path, image)
            if text:
                self.finished.emit(self.path, text)
            if err:
                self.error.emit(self.path, err)
            elif not text and not image:
                self.error.emit(self.path, "Не удалось создать превью")
        except Exception as exc:
            self.error.emit(self.path, str(exc))


class QuickPreviewWorker(QThread):
    """Extract preview text for a batch of files."""

    finished = pyqtSignal(str, str)  # path, text
    error = pyqtSignal(str, str)

    def __init__(self, files: list[str]):
        super().__init__()
        self.files = files

    def run(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as ex:
            future_map = {
                ex.submit(extract_preview_text, Path(p)): p for p in self.files
            }
            for fut in as_completed(future_map):
                path = future_map[fut]
                try:
                    text = fut.result()
                    self.finished.emit(path, text)
                except Exception as exc:
                    self.error.emit(path, str(exc))


class VerificationWorker(QThread):
    """Run fuzzy verification against reference names."""

    result = pyqtSignal(str, str, str, str, int)  # path, line, ref, number, score
    finished = pyqtSignal()

    def __init__(self, data: dict[str, str], threshold: int):
        super().__init__()
        self.data = data
        self.threshold = threshold

    def run(self) -> None:
        import re

        references = [r["ru"] for r in REFERENCE_NAMES]
        for path, text in self.data.items():
            best_line = ""
            best_ref = ""
            best_score = 0
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                ref, score = process.extractOne(
                    line, references, scorer=fuzz.token_set_ratio
                )
                if score > best_score:
                    best_score = score
                    best_line = line
                    best_ref = ref
            num_match = re.search(r"№\s*\S+", text)
            number = num_match.group(0) if num_match else ""
            self.result.emit(path, best_line, best_ref, number, best_score)
        self.finished.emit()
