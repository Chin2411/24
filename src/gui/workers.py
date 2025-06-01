from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import List
import logging

import rarfile
import py7zr
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


def extract_archive(archive_path: str, dest_dir: Path) -> List[str]:
    """Extract supported archive formats to destination directory.

    Returns list of extracted file paths.
    """
    logger.info("Начало извлечения архива %s", archive_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_lower = archive_path.lower()
    extracted: List[str] = []

    if archive_lower.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                zf.extract(member, dest_dir)
                extracted.append(str(dest_dir / member.filename))
    elif archive_lower.endswith('.rar'):
        with rarfile.RarFile(archive_path) as rf:
            for member in rf.infolist():
                if member.isdir():
                    continue
                rf.extract(member, dest_dir)
                extracted.append(str(dest_dir / member.filename))
    elif archive_lower.endswith('.7z'):
        with py7zr.SevenZipFile(archive_path, mode='r') as sz:
            sz.extractall(path=dest_dir)
            for name in sz.getnames():
                if not name.endswith('/'):
                    extracted.append(str(dest_dir / name))
    elif archive_lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2')):
        with tarfile.open(archive_path, 'r:*') as tf:
            for member in tf.getmembers():
                if member.isdir():
                    continue
                tf.extract(member, dest_dir)
                extracted.append(str(dest_dir / member.name))
    else:
        raise ValueError('Неподдерживаемый формат архива')

    logger.info("Извлечено %s файлов в %s", len(extracted), dest_dir)
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
            logger.info(
                "Извлечение архива %s в %s", self.archive_path, self.dest_dir
            )
            files = extract_archive(self.archive_path, self.dest_dir)
            self.finished.emit(files)
        except Exception as exc:
            logger.exception("Ошибка извлечения архива %s", self.archive_path)
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
        for path in self.files:
            logger.info("Извлечение метаданных для %s", path)
            count, language, paper = extract_metadata(Path(path))
            if count.startswith("Ошибка") or count == "Неподдерживаемый формат":
                logger.warning("Ошибка метаданных %s: %s", path, count)
                self.error.emit(path, count)
            self.result.emit(path, language, paper, count)


class FilePreviewWorker(QThread):
    """Worker thread for extracting file preview text or image."""

    finished = pyqtSignal(str, str)  # path, preview text
    imageReady = pyqtSignal(str, str)  # path, image path
    error = pyqtSignal(str, str)  # path, error message

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            logger.info("Извлечение превью для %s", self.path)
            text, image = extract_preview(Path(self.path))
            if image:
                self.imageReady.emit(self.path, image)
            if text:
                self.finished.emit(self.path, text)
            elif not image:
                logger.warning("Не удалось создать превью для %s", self.path)
                self.error.emit(self.path, "Не удалось создать превью")
        except Exception as exc:
            logger.exception("Ошибка создания превью %s", self.path)
            self.error.emit(self.path, str(exc))
