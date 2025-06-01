"""Main application window."""

from pathlib import Path
import sys

# Ensure the project root is in sys.path so that config can be imported even
# when running this module directly from the ``src/ui`` directory.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from config import EXTRACTED_FILES_DIR
from gui.workers import ArchiveExtractWorker, FileMetadataWorker



class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Document Processor")
        self.resize(1024, 768)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Верхняя панель кнопок
        top_panel = QHBoxLayout()
        self.loadArchiveButton = QPushButton("Загрузить архив")
        self.loadFilesButton = QPushButton("Загрузить файлы")
        self.clearBufferButton = QPushButton("Очистить буфер")
        self.runVerificationButton = QPushButton("Выполнить сверку")
        self.viewLogsButton = QPushButton("Логи")
        self.referenceButton = QPushButton("Эталонный справочник")
        for btn in (
            self.loadArchiveButton,
            self.loadFilesButton,
            self.clearBufferButton,
            self.runVerificationButton,
            self.viewLogsButton,
            self.referenceButton,
        ):
            top_panel.addWidget(btn)
        main_layout.addLayout(top_panel)

        # Рабочая область
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.fileTable = QTableWidget()
        self.fileTable.setColumnCount(5)
        self.fileTable.setHorizontalHeaderLabels(
            [
                "Имя файла",
                "Формат файла",
                "Язык",
                "Формат бумаги",
                "Количество страниц/строк/слайдов",
            ]
        )
        self.fileTable.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.fileTable.horizontalHeader().setStretchLastSection(True)

        splitter.addWidget(self.fileTable)

        self.textPreview = QTextEdit()
        self.textPreview.setReadOnly(True)
        self.imagePreview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.unsupportedLabel = QLabel(
            "Просмотр не поддерживается", alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.previewStack = QStackedWidget()
        self.previewStack.addWidget(self.textPreview)
        self.previewStack.addWidget(self.imagePreview)
        self.previewStack.addWidget(self.unsupportedLabel)
        self.previewStack.setCurrentWidget(self.unsupportedLabel)

        splitter.addWidget(self.previewStack)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)

        main_layout.addWidget(splitter, 1)

        # mapping from file path to row index for metadata updates
        self._row_map: dict[str, int] = {}

        # Нижняя панель кнопок
        bottom_panel = QHBoxLayout()
        self.downloadPrepButton = QPushButton("Скачать предопись")
        self.loadPrepButton = QPushButton("Загрузить предопись")
        self.applyCodingButton = QPushButton("Нанести кодировку")
        self.renameFilesButton = QPushButton("Переименовать файлы")
        self.downloadArchiveButton = QPushButton("Скачать архив")
        self.downloadOpisButton = QPushButton("Скачать опись")
        for btn in (
            self.downloadPrepButton,
            self.loadPrepButton,
            self.applyCodingButton,
            self.renameFilesButton,
            self.downloadArchiveButton,
            self.downloadOpisButton,
        ):
            bottom_panel.addWidget(btn)
        main_layout.addLayout(bottom_panel)

        # Соединяем кнопки с заглушками
        for btn in (
            self.loadFilesButton,
            self.clearBufferButton,
            self.runVerificationButton,
            self.viewLogsButton,
            self.referenceButton,
            self.downloadPrepButton,
            self.loadPrepButton,
            self.applyCodingButton,
            self.renameFilesButton,
            self.downloadArchiveButton,
            self.downloadOpisButton,
        ):
            btn.clicked.connect(self._not_implemented)

        self.loadArchiveButton.clicked.connect(self.load_archive)

        self.fileTable.itemSelectionChanged.connect(self._preview_selected)


    def _preview_selected(self) -> None:
        """Обработчик выбора файла в таблице (заглушка)."""
        self.previewStack.setCurrentWidget(self.unsupportedLabel)

    def _not_implemented(self) -> None:
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

        # notify user about chosen archive path
        QMessageBox.information(self, "Выбран архив", file_path)

        dest = EXTRACTED_FILES_DIR / Path(file_path).stem
        self.archive_worker = ArchiveExtractWorker(file_path, dest)
        self.archive_worker.finished.connect(self.on_archive_extracted)
        self.archive_worker.error.connect(self.on_archive_error)
        self.archive_worker.start()

    def on_archive_extracted(self, files: list[str]) -> None:
        self._row_map.clear()
        for file_path in files:
            row = self.fileTable.rowCount()
            self.fileTable.insertRow(row)
            path = Path(file_path)
            self.fileTable.setItem(row, 0, QTableWidgetItem(path.name))
            self.fileTable.setItem(row, 1, QTableWidgetItem(path.suffix.lstrip(".")))
            self.fileTable.setItem(row, 2, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 3, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 4, QTableWidgetItem("-"))
            self._row_map[str(path)] = row

        # start metadata extraction in background
        self.meta_worker = FileMetadataWorker(files)
        self.meta_worker.result.connect(self._update_metadata_row)
        self.meta_worker.start()

        QMessageBox.information(self, "Успех", "Архив успешно загружен")

    def on_archive_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)

    def _update_metadata_row(self, path: str, language: str, paper: str, count: str) -> None:
        row = self._row_map.get(path)
        if row is None:
            return
        self.fileTable.setItem(row, 2, QTableWidgetItem(language))
        self.fileTable.setItem(row, 3, QTableWidgetItem(paper))
        self.fileTable.setItem(row, 4, QTableWidgetItem(count))
