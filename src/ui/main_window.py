from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
)

from config import EXTRACTED_FILES_DIR
from src.gui.workers import ArchiveExtractWorker


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        ui_file = Path(__file__).resolve().parent.parent.parent / "data" / "ui" / "main_window.ui"
        uic.loadUi(str(ui_file), self)

        # Table to display extracted files
        self.fileTable = QTableWidget(0, 5, self)
        self.fileTable.setHorizontalHeaderLabels(
            [
                "Имя файла",
                "Тип",
                "Язык",
                "Формат",
                "Страниц/Строк",
            ]
        )
        self.verticalLayout.insertWidget(0, self.fileTable)

        self.loadArchiveButton.clicked.disconnect()
        self.loadArchiveButton.clicked.connect(self.load_archive)
        self.loadFilesButton.clicked.connect(self._not_implemented)
        self.clearBufferButton.clicked.connect(self._not_implemented)
        self.runVerificationButton.clicked.connect(self._not_implemented)
        self.viewLogsButton.clicked.connect(self._not_implemented)

    def _not_implemented(self):
        QMessageBox.information(self, "Info", "Функция не реализована.")

    def load_archive(self) -> None:
        """Open file dialog and extract selected archive in a background thread."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите архив",
            str(Path.home()),
            "Archives (*.zip *.rar *.7z *.tar *.tar.gz *.tar.bz2)",
        )
        if not file_path:
            return

        dest = EXTRACTED_FILES_DIR / Path(file_path).stem
        self.archive_worker = ArchiveExtractWorker(file_path, dest)
        self.archive_worker.finished.connect(self.on_archive_extracted)
        self.archive_worker.error.connect(self.on_archive_error)
        self.archive_worker.start()

    def on_archive_extracted(self, files: list[str]) -> None:
        for file_path in files:
            row = self.fileTable.rowCount()
            self.fileTable.insertRow(row)
            path = Path(file_path)
            self.fileTable.setItem(row, 0, QTableWidgetItem(path.name))
            self.fileTable.setItem(row, 1, QTableWidgetItem(path.suffix.lstrip(".")))
            self.fileTable.setItem(row, 2, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 3, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 4, QTableWidgetItem("-"))

        QMessageBox.information(self, "Успех", "Архив успешно загружен")

    def on_archive_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)
