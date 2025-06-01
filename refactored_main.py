import sys
import logging
from src import setup_logging

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

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
        try:
            logging.getLogger(__name__).exception("Критическая ошибка: %s", e)
        except Exception:
            print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
