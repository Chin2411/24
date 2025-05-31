"""Main application window."""

from pathlib import Path



class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()


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


        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите архив",
            str(Path.home()),

