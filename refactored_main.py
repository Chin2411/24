import sys
import logging
import logging.config
import os
from pathlib import Path

# Базовая конфигурация логирования до импортов PyQt
LOG_FILE_PATH = Path(__file__).resolve().parent / "application_logs.txt"
logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("Базовая конфигурация логирования инициализирована")

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from config import LOGGING, LOG_FILE

# Настройка логирования
def setup_logging():
    """Configure application logging."""
    LOGGING['handlers']['file']['filename'] = str(LOG_FILE)

    if not os.access(LOG_FILE.parent, os.W_OK):
        raise RuntimeError(f"Нет прав записи в {LOG_FILE.parent}")

    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.touch(exist_ok=True)

    # Применяем конфигурацию
    logging.config.dictConfig(LOGGING)
    logger = logging.getLogger(__name__)
    logger.info("Настройки логирования успешно применены")
    logger.info("Тестовая запись логирования при старте приложения")

def main():
    """Точка входа в приложение."""
    try:
        # Настройка логирования
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Приложение запущено")
        
        # Создание приложения
        app = QApplication(sys.argv)
        
        # Создание и отображение главного окна
        logger.info("Запуск главного окна приложения")
        window = MainWindow()
        window.show()
        
        # Запуск цикла обработки событий
        exit_code = app.exec()
        logger.info("Приложение завершилось с кодом %s", exit_code)
        sys.exit(exit_code)
        
    except Exception as e:
        # Если логгер еще не настроен, выводим ошибку в консоль
        try:
            logger.exception("Критическая ошибка: %s", e)
        except:
            print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
