# Document Processor

Приложение для обработки и анализа технической документации с возможностью архивации и добавления штампов.

## Требования


- Python 3.11+
- PyQt6
- Дополнительные зависимости указаны в `requirements.txt`
- Для корректного определения языка используется `langid`
- Для улучшенного OCR потребуются `easyocr`, `paddleocr`, `camelot-py`, `pdfplumber`, `tabula-py` и `opencv-python`

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/document-processor.git
cd document-processor
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate  # Для Linux/MacOS
# или
venv\Scripts\activate.bat  # Для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск

Для запуска приложения выполните:

```bash
python refactored_main.py
```

## Testing

Быстрая проверка выполняется командами:

```bash
time python -m py_compile $(git ls-files '*.py')
python benchmarks/profile_preview.py sample_dir/
```

Превью отображает только первую страницу документа ради производительности.

## Структура проекта

```
document-processor/
│
├── refactored_main.py       # Точка входа в приложение
├── config.py                # Конфигурация и настройки приложения
├── requirements.txt         # Зависимости проекта
│
├── src/                     # Исходный код приложения
│   ├── analysis/            # Модули для анализа документов
│   │   └── type_detector.py # Определение типа документа
│   │
│   ├── core/                # Ядро приложения
│   │   └── interfaces.py    # Интерфейсы и базовые классы
│   │
│   ├── gui/                 # Компоненты GUI
│   │   └── workers.py       # Рабочие потоки для фоновых задач
│   │
│   ├── services/            # Сервисы для обработки данных
│   │   ├── file_analyzer.py        # Анализ файлов
│   │   ├── document_stamper.py     # Добавление штампов
│   │   └── archive_builder.py      # Создание архивов
│   │
│   ├── ui/                  # Пользовательский интерфейс
│   │   ├── main_window.py   # Главное окно приложения
│   │   ├── icons.py         # Иконки и графические ресурсы
│   │   └── styles.py        # Стили для UI компонентов
│   │
│   └── utils/               # Вспомогательные утилиты
│       ├── reference_loader.py     # Загрузка справочных данных
│       └── zip_handler.py          # Работа с ZIP-архивами
│
├── data/                    # Данные и ресурсы
│   └── reference_data.json  # Справочные данные
│
├── logs/                    # Логи приложения
│
└── tests/                   # Тесты
    └── test_reference_loader.py  # Тесты для модуля загрузки данных
```

## Функциональность

- Загрузка и анализ документов различных форматов (PDF, DOCX, XLSX, изображения)
- Предпросмотр Excel-файлов (первые строки таблицы)
- Извлечение метаданных и определение типа документа
- Добавление штампов на документы
- Создание архивов с переименованными файлами
- Сохранение результатов анализа в Excel

## Лицензия

Этот проект распространяется под лицензией MIT.
