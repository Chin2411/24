from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        """Configure basic widgets."""
        label = QLabel("Добро пожаловать в Document Processor")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(label)
        self.setCentralWidget(central_widget)
        self.setWindowTitle("Document Processor")

    # ------------------------------------------------------------------
    # Settings handling
    # ------------------------------------------------------------------
    def _load_settings(self) -> None:
        """Load window geometry from ``settings.json`` if available."""
        settings_path = Path("settings.json")
        if not settings_path.exists():
            return
        try:
            with settings_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # pragma: no cover - best effort load
            self.logger.warning("Failed to read settings: %s", exc)
            return

        window = data.get("window", {})
        width = window.get("width", 800)
        height = window.get("height", 600)
        x = window.get("x", 100)
        y = window.get("y", 100)
        self.setGeometry(x, y, width, height)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Persist window geometry to ``settings.json`` on exit."""
        settings_path = Path("settings.json")
        try:
            if settings_path.exists():
                with settings_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            else:
                data = {}
        except Exception:
            data = {}

        if "window" not in data:
            data["window"] = {}
        geom = self.geometry()
        data["window"].update(
            {
                "width": geom.width(),
                "height": geom.height(),
                "x": geom.x(),
                "y": geom.y(),
            }
        )

        try:
            with settings_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=4)
        except Exception as exc:  # pragma: no cover - best effort save
            self.logger.warning("Failed to save settings: %s", exc)

        super().closeEvent(event)
