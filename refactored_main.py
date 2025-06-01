import sys
import logging
import logging.config
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from config import LOGGING, LOG_FILE

# Настройка логирования
def setup_logging():
    # Создаем директорию для логов, если она не существует
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Обновляем путь к файлу логов в конфигурации
    LOGGING['handlers']['file']['filename'] = str(LOG_FILE)
    
    # Применяем конфигурацию
    logging.config.dictConfig(LOGGING)
    logger = logging.getLogger(__name__)
    logger.info("Настройки логирования успешно применены")

def main():
    """Точка входа в приложение."""
    try:
        # Настройка логирования
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Создание приложения
        app = QApplication(sys.argv)
        
        # Создание и отображение главного окна
        window = MainWindow()
        window.show()
        
        # Запуск цикла обработки событий
        sys.exit(app.exec())
        
    except Exception as e:
        # Если логгер еще не настроен, выводим ошибку в консоль
        try:
            logger.critical(f"Критическая ошибка: {str(e)}")
        except:
            print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
