from pathlib import Path

# Directory containing this file: src/common
ROOT_DIR = Path(__file__).resolve().parents[2]

# Path to the application log file in ``logs`` directory next to main.py
LOG_PATH = ROOT_DIR / "logs" / "application.log"
