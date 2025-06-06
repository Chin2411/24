from pathlib import Path
import sys
import time
from services.file_preview import extract_preview_text


def main(dir_path: str) -> None:
    start = time.perf_counter()
    count = 0
    for p in Path(dir_path).glob('*'):
        if p.is_file():
            try:
                extract_preview_text(p)
            except Exception:
                pass
            count += 1
    duration = time.perf_counter() - start
    print(f"Processed {count} files in {duration:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: profile_preview.py DIR")
        sys.exit(1)
    main(sys.argv[1])
