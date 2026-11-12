# data/

Сам датасет в репозиторий не кладётся (см. `.gitignore`).

```bash
python data/download.py
```

После запуска файл появится в `data/raw/winemag-data-130k-v2.csv`.
Сплиты train/val/test (80/10/10, seed=42) сохраняются в `data/processed/`.

На Kaggle датасет подключается через *Add Input* (`zynicide/wine-reviews`), скачивать его не нужно.
