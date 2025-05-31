from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import List

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

