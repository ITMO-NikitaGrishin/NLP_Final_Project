# data/

Сам датасет в репозиторий не кладётся (см. `.gitignore`).

```bash
python data/download.py
```

После запуска файл появится в `data/raw/winemag-data-130k-v2.csv`.
Промежуточные файлы (очищенные данные, сплиты) сохраняются в `data/processed/`.
