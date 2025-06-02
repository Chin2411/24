from __future__ import annotations

from pathlib import Path
import shutil
import logging
from src.common.paths import LOG_PATH

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtGui import QTextCursor, QPalette, QColor
from collections import deque
from src.utils.logging_utils import logger_flush

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
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text.setStyleSheet(
            """
            QPlainTextEdit {
                background:#121212;
                color:#FFFFFF;
                selection-background-color:#404040;
                font-family: Menlo, monospace;
                font-size:11px;
            }
            """
        )
        layout.addWidget(self._text)

        p = self._text.palette()
        p.setColor(QPalette.ColorRole.Base, QColor("#121212"))
        p.setColor(QPalette.ColorRole.Text, QColor("#FFFFFF"))
        self._text.setPalette(p)

        btn_layout = QHBoxLayout()
        self.refreshButton = QPushButton("Обновить")
        self.refreshButton.setStyleSheet("")
        self.saveButton = QPushButton("Сохранить как файл")
        self.saveButton.setStyleSheet("")
        btn_layout.addWidget(self.refreshButton)
        btn_layout.addWidget(self.saveButton)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.refreshButton.clicked.connect(self.refresh)
        self.saveButton.clicked.connect(self.save_logs)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.refresh()

    def _append_colored(self, line: str) -> None:
        """Append a log line to the text widget."""
        self._text.appendPlainText(line)

    def refresh(self) -> None:
        """Reload the log file and show its contents."""
        logger_flush()
        try:
            with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
                if self.max_lines:
                    lines = deque(f, self.max_lines)
                    text = "".join(lines)
                else:
                    text = f.read()
        except FileNotFoundError:
            self._text.setPlainText(
                "Лог-файл ещё не создан — выполните действие в программе и нажмите «Обновить»."
            )
            return
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Ошибка чтения логов: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить логи: {exc}")
            return

        self._text.setPlainText(text)
        self._text.moveCursor(QTextCursor.MoveOperation.End)

    def load_logs(self) -> None:
        """Backward compatibility wrapper for :py:meth:`refresh`."""
        self.refresh()

    def save_logs(self) -> None:
        """Save current log file to user selected location."""
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
                f.write(self._text.toPlainText())
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Ошибка сохранения логов: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить логи: {exc}")
