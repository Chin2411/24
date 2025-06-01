from __future__ import annotations

from pathlib import Path
import shutil
import logging

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTextEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtGui import QTextCursor

logger = logging.getLogger(__name__)


class LogViewerDialog(QDialog):
    """Диалоговое окно просмотра логов."""

    def __init__(self, log_file: Path, parent=None, max_lines: int = 200) -> None:
        super().__init__(parent)
        self.log_file = log_file
        self.max_lines = max_lines
        self.setWindowTitle("Логи")
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text)

        btn_layout = QHBoxLayout()
        self.refreshButton = QPushButton("Обновить")
        self.saveButton = QPushButton("Сохранить как файл")
        btn_layout.addWidget(self.refreshButton)
        btn_layout.addWidget(self.saveButton)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.refreshButton.clicked.connect(self.load_logs)
        self.saveButton.clicked.connect(self.save_logs)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.load_logs()

    def _append_colored(self, line: str) -> None:
        color = "black"
        if "[ERROR]" in line:
            color = "red"
        elif "[WARNING]" in line:
            color = "orange"
        self.text.append(f'<span style="color:{color}">{line}</span>')

    def load_logs(self) -> None:
        """Load last N lines from log file."""
        try:
            if not self.log_file.exists():
                self.text.clear()
                QMessageBox.information(self, "Логи", "Файл логов отсутствует")
                return
            lines = self.log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            lines = lines[-self.max_lines :]
            self.text.clear()
            for line in lines:
                self._append_colored(line)
            self.text.moveCursor(QTextCursor.MoveOperation.End)
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Ошибка чтения логов: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить логи: {exc}")

    def save_logs(self) -> None:
        """Save current log file to user selected location."""
        try:
            if not self.log_file.exists():
                QMessageBox.information(self, "Логи", "Файл логов отсутствует")
                return
            dest_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить лог",
                str(Path.home() / "application_logs.txt"),
                "Text Files (*.txt);;All Files (*)",
            )
            if dest_path:
                shutil.copy(str(self.log_file), dest_path)
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Ошибка сохранения логов: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить логи: {exc}")
