"""Main application window."""

from pathlib import Path
import sys
import logging

# Ensure the project root is in sys.path so that config can be imported even
# when running this module directly from the ``src/ui`` directory.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QColor, QPixmap
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
    QMenu,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from config import EXTRACTED_FILES_DIR
import difflib
from data_analysis.reference_data import reference_data
from src.common.paths import LOG_PATH
from ui.log_viewer import LogViewerDialog
from gui.workers import (
    ArchiveExtractWorker,
    FileMetadataWorker,
    FilePreviewWorker,
)
from services.file_preview import (
    SUPPORTED_IMAGES,
    SUPPORTED_TEXT,
    SUPPORTED_EXCEL,
    SUPPORTED_DOCS,
)
from concurrent.futures import ThreadPoolExecutor

SUPPORTED_PREVIEW_EXTS = (
    SUPPORTED_IMAGES
    | SUPPORTED_TEXT
    | SUPPORTED_EXCEL
    | SUPPORTED_DOCS
    | {".pdf"}
)


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Document Processor")
        self.resize(1024, 768)
        self.logger.info("Главное окно инициализировано")

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

        # enable custom context menu for table
        self.fileTable.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.fileTable.customContextMenuRequested.connect(
            self._show_file_menu
        )

        # mapping from file path to row index for metadata updates
        self._row_map: dict[str, int] = {}
        # mapping from file path to processing error message
        self._error_map: dict[str, str] = {}
        # set of all added file paths to prevent duplicates
        self._all_paths: set[str] = set()
        # active worker threads
        self._workers: list[QThread] = []
        # executor for possible background tasks
        self.executor = ThreadPoolExecutor()

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

        # Настраиваем подключения сигналов и слотов
        self.setup_connections()

        self.fileTable.itemSelectionChanged.connect(self._preview_selected)

    def setup_connections(self) -> None:
        """Connect UI buttons to their respective slots."""
        for btn in (
            self.clearBufferButton,
            self.runVerificationButton,
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
        self.loadFilesButton.clicked.connect(self.load_files)
        self.clearBufferButton.clicked.connect(self.clear_buffer)
        self.runVerificationButton.clicked.connect(self.perform_check)
        self.viewLogsButton.clicked.connect(self.show_logs)

    def _start_worker(self, worker: QThread) -> None:
        """Start and track a QThread worker."""
        self._workers.append(worker)

        def _cleanup(w=worker) -> None:
            try:
                self._workers.remove(w)
            except ValueError:
                pass

        worker.finished.connect(_cleanup)
        worker.start()

    def load_files(self) -> None:
        """Open file dialog and add selected files to the table."""
        self.logger.info("Кнопка 'Загрузить файлы' нажата")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите файлы",
            str(Path.home()),
            "Все файлы (*)",
        )
        if not files:
            return

        self.logger.info("Загружено файлов: %s", len(files))

        new_files: list[str] = []
        for f in files:
            abs_path = str(Path(f).resolve())
            if abs_path in self._all_paths:
                continue
            row = self.fileTable.rowCount()
            self.fileTable.insertRow(row)
            path_obj = Path(abs_path)
            name_item = QTableWidgetItem(path_obj.name)
            name_item.setData(Qt.ItemDataRole.UserRole, abs_path)
            self.fileTable.setItem(row, 0, name_item)
            self.fileTable.setItem(
                row, 1, QTableWidgetItem(path_obj.suffix.lstrip("."))
            )
            self.fileTable.setItem(row, 2, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 3, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 4, QTableWidgetItem("-"))
            self._row_map[abs_path] = row
            self._all_paths.add(abs_path)
            new_files.append(abs_path)
            self.logger.info("Добавлен файл %s", abs_path)
            if path_obj.suffix.lower() not in SUPPORTED_PREVIEW_EXTS:
                self._mark_unsupported(row)

        if not new_files:
            return

        self.meta_worker = FileMetadataWorker(new_files)
        self.meta_worker.result.connect(self._update_metadata_row)
        self.meta_worker.error.connect(self._on_meta_error)
        self.logger.info("Запуск потока извлечения метаданных")
        self._start_worker(self.meta_worker)
        self.logger.info("Загрузка файлов завершена")

    def _preview_selected(self) -> None:
        selected = self.fileTable.selectedItems()
        if not selected:
            self.previewStack.setCurrentWidget(self.unsupportedLabel)
            return
        # first column item contains path in UserRole
        row = self.fileTable.currentRow()
        item = self.fileTable.item(row, 0)
        if item is None:
            self.previewStack.setCurrentWidget(self.unsupportedLabel)
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            self.previewStack.setCurrentWidget(self.unsupportedLabel)
            return

        self.logger.info("Создание превью для %s", path)

        self.textPreview.setPlainText("Загрузка...")
        self.previewStack.setCurrentWidget(self.textPreview)
        self.preview_worker = FilePreviewWorker(path)
        self.preview_worker.finished.connect(self._on_preview_ready)
        self.preview_worker.imageReady.connect(self._on_preview_image)
        self.preview_worker.error.connect(self._on_preview_error)
        self._start_worker(self.preview_worker)

    def _not_implemented(self) -> None:
        QMessageBox.information(self, "Info", "Функция не реализована.")

    def load_archive(self) -> None:
        """Open file dialog and extract selected archive in a background thread."""
        self.logger.info("Кнопка 'Загрузить архив' нажата")
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
        self.logger.info("Запуск распаковки архива %s", file_path)
        self.archive_worker = ArchiveExtractWorker(file_path, dest)
        self.archive_worker.finished.connect(self.on_archive_extracted)
        self.archive_worker.error.connect(self.on_archive_error)
        self._start_worker(self.archive_worker)
        self.logger.info("Поток распаковки архива запущен")

    def on_archive_extracted(self, files: list[str]) -> None:
        self.logger.info("Архив распакован, файлов: %s", len(files))
        new_files: list[str] = []
        for file_path in files:
            abs_path = str(Path(file_path).resolve())
            if abs_path in self._all_paths:
                continue
            row = self.fileTable.rowCount()
            self.fileTable.insertRow(row)
            path = Path(abs_path)
            name_item = QTableWidgetItem(path.name)
            name_item.setData(Qt.ItemDataRole.UserRole, abs_path)
            self.fileTable.setItem(row, 0, name_item)
            self.fileTable.setItem(row, 1, QTableWidgetItem(path.suffix.lstrip(".")))
            self.fileTable.setItem(row, 2, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 3, QTableWidgetItem("-"))
            self.fileTable.setItem(row, 4, QTableWidgetItem("-"))
            self._row_map[abs_path] = row
            self._all_paths.add(abs_path)
            new_files.append(abs_path)
            if path.suffix.lower() not in SUPPORTED_PREVIEW_EXTS:
                self._mark_unsupported(row)

        if new_files:
            # start metadata extraction in background
            self.meta_worker = FileMetadataWorker(new_files)
            self.meta_worker.result.connect(self._update_metadata_row)
            self.meta_worker.error.connect(self._on_meta_error)
            self._start_worker(self.meta_worker)

        self.logger.info("Распаковка архива завершена")

        QMessageBox.information(self, "Успех", "Архив успешно загружен")

    def on_archive_error(self, message: str) -> None:
        self.logger.error("Ошибка распаковки архива: %s", message)
        QMessageBox.critical(self, "Ошибка", message)

    def _update_metadata_row(
        self, path: str, language: str, paper: str, count: str
    ) -> None:
        row = self._row_map.get(path)
        if row is None:
            return
        self.fileTable.setItem(row, 2, QTableWidgetItem(language))
        self.fileTable.setItem(row, 3, QTableWidgetItem(paper))
        self.fileTable.setItem(row, 4, QTableWidgetItem(count))
        if path in self._error_map:
            self._highlight_row(row, self._error_map[path])

    def _on_preview_ready(self, path: str, text: str) -> None:
        self.logger.info("Превью готово для %s", path)
        self.textPreview.setPlainText(text)
        self.previewStack.setCurrentWidget(self.textPreview)

    def _on_preview_image(self, path: str, image_path: str) -> None:
        self.logger.warning("Показ изображения превью для %s", path)
        pixmap = QPixmap(image_path)
        self.imagePreview.setPixmap(
            pixmap.scaled(
                self.imagePreview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        QMessageBox.warning(self, "OCR", "Не удалось корректно распознать таблицу")
        self.previewStack.setCurrentWidget(self.imagePreview)

    def _on_preview_error(self, path: str, message: str) -> None:
        self.logger.error("Ошибка превью %s: %s", path, message)
        self.textPreview.setPlainText(message)
        self.previewStack.setCurrentWidget(self.textPreview)
        self._error_map[path] = message
        row = self._row_map.get(path)
        if row is not None:
            if "неподдерж" in message.lower():
                self._mark_unsupported(row)
            else:
                self._highlight_row(row, message)

    def _on_meta_error(self, path: str, message: str) -> None:
        self.logger.warning("Ошибка метаданных %s: %s", path, message)
        self._error_map[path] = message
        row = self._row_map.get(path)
        if row is not None:
            if "неподдерж" in message.lower():
                self._mark_unsupported(row)
            else:
                self._highlight_row(row, message)

    def _highlight_row(self, row: int, message: str) -> None:
        for col in range(self.fileTable.columnCount()):
            item = self.fileTable.item(row, col)
            if item is None:
                item = QTableWidgetItem("-")
                self.fileTable.setItem(row, col, item)
            item.setBackground(QColor("#ffc0cb"))
            if not item.toolTip():
                item.setToolTip(message)

    def _mark_unsupported(self, row: int) -> None:
        for col in range(self.fileTable.columnCount()):
            item = self.fileTable.item(row, col)
            if item is None:
                item = QTableWidgetItem("-")
                self.fileTable.setItem(row, col, item)
            item.setBackground(QColor(255, 180, 180))
            if not item.toolTip():
                item.setToolTip("Неподдерживаемый формат")

    def perform_check(self) -> None:
        """Check file names against reference_data using fuzzy matching."""
        self.logger.info("Кнопка 'Выполнить сверку' нажата")
        results: list[str] = []
        keys_lower = {k.lower(): k for k in reference_data}
        for path, row in self._row_map.items():
            name = Path(path).stem
            match = None
            matches = difflib.get_close_matches(name.lower(), keys_lower.keys(), n=1, cutoff=0.8)
            if matches:
                match = keys_lower[matches[0]]
                translation = reference_data[match]
                results.append(f"{name} → {translation}")
            else:
                results.append(f"{name} → нет совпадений")
        if results:
            QMessageBox.information(self, "Результаты сверки", "\n".join(results))
        else:
            QMessageBox.information(self, "Результаты сверки", "Нет файлов для проверки")

    def clear_buffer(self) -> None:
        """Remove all files from the table and internal lists."""
        self.logger.info("Кнопка 'Очистить буфер' нажата")
        self.logger.info("Очистка буфера")
        self.fileTable.setRowCount(0)
        self._row_map.clear()
        self._error_map.clear()
        self._all_paths.clear()
        self.fileTable.clearSelection()
        self.textPreview.clear()
        self.imagePreview.clear()
        self.previewStack.setCurrentWidget(self.unsupportedLabel)
        self.logger.info("Буфер очищен")

    def _show_file_menu(self, pos) -> None:
        index = self.fileTable.indexAt(pos)
        if not index.isValid():
            return
        menu = QMenu(self.fileTable)
        delete_action = menu.addAction("Удалить файл")
        action = menu.exec(self.fileTable.viewport().mapToGlobal(pos))
        if action == delete_action:
            row = index.row()
            item = self.fileTable.item(row, 0)
            if item is None:
                return
            path = item.data(Qt.ItemDataRole.UserRole)
            self._remove_file(path, row)

    def _remove_file(self, path: str, row: int) -> None:
        """Delete file information from table and internal structures."""
        self.logger.info("Удаление файла %s", path)
        self.fileTable.removeRow(row)
        self.fileTable.clearSelection()
        self._row_map.pop(path, None)
        self._error_map.pop(path, None)
        self._all_paths.discard(path)

        # adjust row indices after removed row
        for p, r in list(self._row_map.items()):
            if r > row:
                self._row_map[p] = r - 1

        self.textPreview.clear()
        self.imagePreview.clear()
        self.previewStack.setCurrentWidget(self.unsupportedLabel)

    def show_logs(self) -> None:
        """Display log viewer dialog."""
        self.logger.info("Кнопка 'Логи' нажата")
        try:
            self.logger.info("Открытие окна логов")
            dlg = LogViewerDialog(LOG_PATH, self)
            dlg.exec()
            self.logger.info("Окно логов закрыто")
        except Exception as exc:  # pragma: no cover - runtime errors
            self.logger.exception("Ошибка отображения логов")
            QMessageBox.critical(self, "Ошибка", str(exc))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Ensure all background threads are properly terminated."""
        for worker in list(self._workers):
            if worker.isRunning():
                worker.quit()
                worker.wait()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
        super().closeEvent(event)
