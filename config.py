from pathlib import Path
import logging
from typing import Dict, List
from src.common.paths import LOG_PATH

# Пути
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
EXTRACTED_FILES_DIR = BASE_DIR / "extracted_files"
# Основной лог-файл приложения
LOG_FILE = LOG_PATH

# Создаем необходимые директории
TEMP_DIR.mkdir(exist_ok=True)
EXTRACTED_FILES_DIR.mkdir(exist_ok=True)

# Настройки безопасности
ALLOWED_EXTENSIONS: Dict[str, List[str]] = {
    'archives': ['.zip', '.rar', '.7z'],
    'documents': ['.pdf', '.docx', '.doc', '.txt', '.rtf'],
    'images': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
}

# Собираем все разрешенные расширения в один список
ALLOWED_EXTENSIONS_LIST: List[str] = [
    ext for exts in ALLOWED_EXTENSIONS.values() for ext in exts
]

MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
BUFFER_SIZE = 1024 * 1024  # 1MB

# Настройки обработки PDF
PDF_PAGES_TO_EXTRACT = 5  # Количество страниц для извлечения с начала и конца
PDF_SAMPLE_INTERVAL = 20  # Интервал выборки страниц
PDF_IMAGE_DPI = 300  # DPI для конвертации в изображение
PDF_IMAGE_THRESHOLD = 128  # Порог бинаризации для OCR
PDF_MAX_WORKERS = 4  # Максимальное количество потоков для обработки

# Настройки обработки DOC/DOCX
DOC_MAX_WORKERS = 4  # Максимальное количество потоков для обработки
DOC_MIN_TEXT_LENGTH = 50  # Минимальная длина текста для проверки качества извлечения
DOC_IMAGE_DPI = 300  # DPI для конвертации в изображение при необходимости OCR
DOC_IMAGE_THRESHOLD = 128  # Порог бинаризации для OCR

# Настройки логирования
LOG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.DEBUG  # Используем константу из модуля logging

# Конфигурация логирования
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': LOG_FORMAT,
            'datefmt': LOG_DATEFMT,
        },
    },
    'handlers': {
        'console': {
            'level': LOG_LEVEL,
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'level': LOG_LEVEL,
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOG_FILE),
            'formatter': 'standard',
            'encoding': 'utf-8',
            'maxBytes': 1_048_576,  # 1 MB
            'backupCount': 3,
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': LOG_LEVEL,
    },
}
