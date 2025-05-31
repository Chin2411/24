from pathlib import Path
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QFileDialog,
)
import zipfile
import rarfile
import py7zr
from config import EXTRACTED_FILES_DIR


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        ui_file = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "ui"
            / "main_window.ui"
        )
        uic.loadUi(str(ui_file), self)

        self.loaded_files = []
        self.loadArchiveButton.clicked.connect(self._load_archive)
        self.loadFilesButton.clicked.connect(self._not_implemented)
        self.clearBufferButton.clicked.connect(self._not_implemented)
        self.runVerificationButton.clicked.connect(self._not_implemented)
        self.viewLogsButton.clicked.connect(self._not_implemented)

    def _not_implemented(self):
        QMessageBox.information(self, "Info", "Функция не реализована.")

    def _load_archive(self):
        """Открыть диалог выбора архива и распаковать его."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите архив",
            str(Path.home()),
            "Архивы (*.zip *.rar *.7z)",
        )

        if not file_path:
            return

        archive = Path(file_path)

        try:
            if archive.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(EXTRACTED_FILES_DIR)
                    extracted = [
                        EXTRACTED_FILES_DIR / name
                        for name in zf.namelist()
                        if not name.endswith("/")
                    ]
            elif archive.suffix.lower() == ".rar":
                with rarfile.RarFile(archive) as rf:
                    rf.extractall(EXTRACTED_FILES_DIR)
                    extracted = [
                        EXTRACTED_FILES_DIR / info.filename
                        for info in rf.infolist()
                        if not info.isdir()
                    ]
            elif archive.suffix.lower() == ".7z":
                with py7zr.SevenZipFile(archive, mode="r") as z:
                    z.extractall(path=str(EXTRACTED_FILES_DIR))
                    extracted = [EXTRACTED_FILES_DIR / name for name in z.getnames()]
            else:
                raise ValueError("Неподдерживаемый формат архива")

            self.loaded_files.extend(extracted)
            QMessageBox.information(self, "Info", "Архив загружен")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить архив: {e}")
