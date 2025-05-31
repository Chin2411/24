"""Main application window."""

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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


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
            self.loadArchiveButton,
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

        self.fileTable.itemSelectionChanged.connect(self._preview_selected)

    def _preview_selected(self) -> None:
        """Обработчик выбора файла в таблице (заглушка)."""
        self.previewStack.setCurrentWidget(self.unsupportedLabel)

    def _not_implemented(self) -> None:
        QMessageBox.information(self, "Info", "Функция не реализована.")
