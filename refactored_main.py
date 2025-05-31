import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))
import logging
import logging.config
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from config import LOGGING

# Настройка логирования
def setup_logging():
    # Создаем директорию для логов, если она не существует
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Обновляем путь к файлу логов в конфигурации
    log_file = log_dir / "application.log"
    LOGGING['handlers']['file']['filename'] = str(log_file)
    
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
