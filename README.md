# Wine Reviews NLP

Итоговый проект по курсу NLP: **предсказание оценки вина (80–100 баллов) и его цены по тексту дегустационной заметки сомелье**.

**Исследовательский вопрос:** насколько хорошо оценку и цену можно предсказать только по тексту и насколько помогает добавление метаданных (страна, сорт, регион)?

**Данные:** [Wine Reviews (Kaggle, zynicide/wine-reviews)](https://www.kaggle.com/datasets/zynicide/wine-reviews), файл `winemag-data-130k-v2.csv`: около 130 тыс. отзывов Wine Enthusiast. Колонки: `description`, `points`, `price`, `country`, `province`, `region_1`, `variety`, `winery`, `taster_name`, `title`.

## Постановка задачи

| Таргет | Тип | Метрики |
|---|---|---|
| `points` | регрессия | RMSE, MAE, R² |
| `log(price)` | регрессия | MAE, R² |
| класс оценки: 2 класса (80–89 / 90+) и 4 класса (80–84 / 85–89 / 90–94 / 95–100) | классификация | Accuracy |

## Подготовка данных

- Удаляются дубликаты по `description`, чтобы одинаковые тексты не попали и в train, и в test.
- Удаляются строки без `price`. Цена логарифмируется.
- Используется одно разбиение train/val/test **80/10/10** с `seed=42`, сохранённое в `data/processed/`.
- Отдельно собирается подвыборка как в статье IACIS 2022: 12 стран, 26 сортов, 12 дегустаторов, только строки с ценой.

## Эксперименты

| № | Эксперимент | Ноутбук |
|---|---|---|
| 1 | Предсказание средним | `02_baselines` |
| 2 | Только метаданные + CatBoost | `02_baselines` |
| 3 | TF-IDF + Ridge | `02_baselines` |
| 4 | Эмбеддинги sentence-transformers + CatBoost: только текст / текст + метаданные | `03_embeddings` |
| 5 | Дообучение RoBERTa-base с регрессионной головой: только текст | `04_transformer` |
| 6 | RoBERTa-base, метаданные добавлены в текст: `Country: ... Variety: ... Description: ...` | `04_transformer` |
| 7 | Previous art: R² на подвыборке IACIS 2022 | `05_interpretation` |
| 8 | Классификация на 2 и 4 класса, сравнение с BERT из Journal of Wine Economics 2022 | `05_interpretation` |
| 9 | SHAP-интерпретация и анализ ошибок | `05_interpretation` |

## Структура репозитория

```
├── README.md
├── CLAUDE.md          # бриф проекта и правила работы
├── requirements.txt
├── data/
│   └── download.py    # загрузка датасета с Kaggle (сами данные не коммитятся)
├── notebooks/
│   ├── 00_data_prep.ipynb       # очистка, сплиты train/val/test, подвыборка IACIS
│   ├── 01_eda.ipynb             # статистика, распределения оценок, цен и длин текстов
│   ├── 02_baselines.ipynb       # эксперименты 1–3
│   ├── 03_embeddings.ipynb      # эксперимент 4
│   ├── 04_transformer.ipynb     # эксперименты 5–6 (локальная GPU)
│   └── 05_interpretation.ipynb  # эксперименты 7–9
├── src/
│   ├── config.py      # пути, seed, названия колонок
│   ├── data.py        # загрузка, очистка, log(price), подвыборка IACIS, классы оценок
│   ├── split.py       # разбиение 80/10/10
│   └── metrics.py     # метрики, запись в results/metrics.csv, сводные таблицы
└── results/
    ├── metrics.csv    # experiment, target, metric, value
    └── figures/       # графики
```

Ноутбуки содержат только эксперименты. Весь общий код вынесен в `src/`.

## Как запустить

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# torch с поддержкой CUDA (нужна для ноутбука 04; RTX 50xx требует CUDA 12.8+)
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu130

python data/download.py          # нужен Kaggle API токен (~/.kaggle/kaggle.json)
jupyter notebook notebooks/
```

Ноутбуки выполняются по порядку. При первом вызове `load_splits()` разбиение сохраняется в `data/processed/`, и дальше все ноутбуки используют его.

RoBERTa-base (`04_transformer`) обучалась локально на NVIDIA RTX 5060 (8 ГБ) в режиме fp16.

## Результаты

Метрики на тестовой выборке. Все значения лежат в `results/metrics.csv`.

### Оценка (`points`)

| Эксперимент | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|
| 1. Среднее | — | — | — |
| 2. Метаданные + CatBoost | — | — | — |
| 3. TF-IDF + Ridge | — | — | — |
| 4a. Эмбеддинги + CatBoost (текст) | — | — | — |
| 4b. Эмбеддинги + CatBoost (текст + метаданные) | — | — | — |
| 5. RoBERTa-base (текст) | — | — | — |
| 6. RoBERTa-base (текст + метаданные) | — | — | — |

### Цена (`log(price)`)

| Эксперимент | MAE ↓ | R² ↑ |
|---|---|---|
| 1. Среднее | — | — |
| 2. Метаданные + CatBoost | — | — |
| 3. TF-IDF + Ridge | — | — |
| 4a. Эмбеддинги + CatBoost (текст) | — | — |
| 4b. Эмбеддинги + CatBoost (текст + метаданные) | — | — |
| 5. RoBERTa-base (текст) | — | — |
| 6. RoBERTa-base (текст + метаданные) | — | — |

### Сравнение с previous art

| Работа | Постановка | Их результат | Наш результат |
|---|---|---|---|
| IACIS 2022 | регрессия `points` на подвыборке (12 стран, 26 сортов, 12 дегустаторов) | R² = 0.593 | — |
| Journal of Wine Economics 2022 (BERT) | 2 класса (80–89 / 90+) | Accuracy = 89% | — |
| Journal of Wine Economics 2022 (BERT) | 4 класса (80–84 / 85–89 / 90–94 / 95–100) | Accuracy = 82% | — |
