"""Загрузка датасета Wine Reviews (Kaggle: zynicide/wine-reviews) в data/raw/.

Использование:
    python data/download.py

Нужен Kaggle API токен: ~/.kaggle/kaggle.json
или переменные окружения KAGGLE_USERNAME / KAGGLE_KEY.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "zynicide/wine-reviews"
FILENAME = "winemag-data-130k-v2.csv"
RAW_DIR = Path(__file__).resolve().parent / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / FILENAME
    if target.exists():
        print(f"Уже скачано: {target}")
        return

    cache_dir = Path(kagglehub.dataset_download(DATASET))
    shutil.copy2(cache_dir / FILENAME, target)
    print(f"Сохранено: {target}")


if __name__ == "__main__":
    main()
