from pathlib import Path
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QMessageBox


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        ui_file = Path(__file__).resolve().parent.parent.parent / "data" / "ui" / "main_window.ui"
        uic.loadUi(str(ui_file), self)

        self.loadArchiveButton.clicked.connect(self._not_implemented)
        self.loadFilesButton.clicked.connect(self._not_implemented)
        self.clearBufferButton.clicked.connect(self._not_implemented)
        self.runVerificationButton.clicked.connect(self._not_implemented)
        self.viewLogsButton.clicked.connect(self._not_implemented)

    def _not_implemented(self):
        QMessageBox.information(self, "Info", "Функция не реализована.")
