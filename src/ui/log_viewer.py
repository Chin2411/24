from __future__ import annotations

from pathlib import Path
import shutil
import logging
from src.common.paths import LOG_PATH
from utils.logging_utils import logger_flush

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

    def __init__(self, log_file: Path = LOG_PATH, parent=None, max_lines: int = 200) -> None:
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

    def _load_log(self) -> None:
        """Internal helper to read log file and display its tail."""
        logger_flush()
        try:
            lines = self.log_file.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            self.text.setPlainText(
                "Лог-файл ещё не создан — выполните действие в программе и нажмите «Обновить»."
            )
            return
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Ошибка чтения логов: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить логи: {exc}")
            return

        lines = lines[-self.max_lines :]
        self.text.clear()
        for line in lines:
            self._append_colored(line)
        self.text.moveCursor(QTextCursor.MoveOperation.End)

    def load_logs(self) -> None:
        """Public slot to refresh log contents."""
        self._load_log()

    def save_logs(self) -> None:
        """Save current log file to user selected location."""
        logger_flush()
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить лог",
            str(Path.home() / "application.log"),
            "Text Files (*.txt);;All Files (*)",
        )
        if not dest_path:
            return
        try:
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(self.text.toPlainText())
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Ошибка сохранения логов: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить логи: {exc}")
