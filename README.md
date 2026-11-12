# Wine Reviews NLP

Финальный проект по NLP: **предсказание оценки вина (`points`) и его цены (`price`) по тексту дегустационного отзыва**, отдельно и вместе с метаданными (страна, сорт, регион и т. д.).

**Исследовательский вопрос:** сколько информации об оценке и цене вина содержится в тексте отзыва и насколько современные модели (эмбеддинги, дообученные трансформеры) превосходят классические подходы и метаданные?

**Данные:** [Wine Reviews (Kaggle, zynicide/wine-reviews)](https://www.kaggle.com/datasets/zynicide/wine-reviews): ~130 тыс. отзывов WineEnthusiast. Поля: `description`, `points` (80–100), `price`, `country`, `province`, `region_1`, `variety`, `winery`, `taster_name`.

## Постановка задачи

| Таргет | Тип | Метрики |
|---|---|---|
| `points` | регрессия | RMSE, MAE, R² |
| `log(price)` | регрессия | MAE, R² |
| класс оценки (2 и 4 класса) | классификация | Accuracy (для сравнения с Journal of Wine Economics) |

## Подготовка данных

- Удаляются повторяющиеся отзывы (~10 тыс.), чтобы одинаковые тексты не попали и в train, и в test.
- Удаляются строки без цены. Цена логарифмируется: её распределение сильно скошено (от 4 до 3300 $).
- Используется одно разбиение train/val/test **80/10/10** с фиксированным seed (`SEED = 42`), общее для всех экспериментов.
- Отдельно собирается подвыборка по условиям статьи IACIS 2022 (12 стран, 26 сортов, 12 дегустаторов) для сравнения с previous art.

## Эксперименты

| Этап | Модель | Вход | Ноутбук |
|---|---|---|---|
| 1 | Предсказание средним | — | `02_baselines` |
| 1 | CatBoost (аналог работы CS229) | метаданные + цена | `02_baselines` |
| 1 | TF-IDF + Ridge | текст | `02_baselines` |
| 2 | Эмбеддинги (multilingual-e5 / all-mpnet) + CatBoost | текст | `03_embeddings` |
| 2 | Эмбеддинги + CatBoost | текст + метаданные | `03_embeddings` |
| 3 | RoBERTa / DeBERTa-v3, регрессионная голова | текст | `04_transformer` |
| 3 | RoBERTa / DeBERTa-v3 | текст + метаданные в тексте¹ | `04_transformer` |
| 4 | Лучшая модель на подвыборке IACIS; классификация на 2 и 4 класса | — | `05_interpretation` |
| 5 | SHAP для TF-IDF и анализ ошибок | — | `05_interpretation` |
| доп. | LLM zero-shot (если останется время) | текст | — |

¹ Метаданные добавляются в начало текста: `Country: Italy. Variety: Sangiovese. Description: ...`

Признак `price` для таргета `log(price)` не используется.

## Структура репозитория

```
├── README.md          # задача, как запустить, итоговая таблица результатов
├── requirements.txt
├── data/
│   └── download.py    # загрузка датасета с Kaggle (сам датасет не коммитится)
├── notebooks/
│   ├── 01_eda.ipynb             # статистика, распределения оценок, цен и длин текстов
│   ├── 02_baselines.ipynb       # среднее, CatBoost на метаданных, TF-IDF + Ridge
│   ├── 03_embeddings.ipynb      # эмбеддинги предложений + CatBoost
│   ├── 04_transformer.ipynb     # дообучение RoBERTa / DeBERTa-v3
│   └── 05_interpretation.ipynb  # previous art, SHAP, анализ ошибок
├── src/
│   ├── config.py      # пути, seed, названия колонок
│   ├── data.py        # загрузка, очистка, log(price), подвыборка IACIS
│   ├── split.py       # единое разбиение 80/10/10
│   └── metrics.py     # метрики регрессии и классификации, запись в results.csv
└── results/
    ├── tables/        # results.csv: все метрики всех моделей
    └── figures/       # графики
```

## Как запустить

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/download.py          # нужен Kaggle API токен (~/.kaggle/kaggle.json)
jupyter notebook notebooks/
```

Ноутбуки выполняются по порядку, с `01` по `05`. При первом вызове `load_splits()` разбиение сохраняется в `data/processed/`, и дальше все ноутбуки используют его.

Трансформеры (`04_transformer`) обучались на бесплатном GPU Kaggle/Colab. RoBERTa-base на ~100 тыс. текстов обучается примерно 1–2 часа.

## Результаты

Метрики на тестовой выборке. Полная таблица лежит в `results/tables/results.csv`.

### Оценка (`points`)

| Модель | Вход | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| Среднее | — | — | — | — |
| CatBoost | метаданные | — | — | — |
| TF-IDF + Ridge | текст | — | — | — |
| Эмбеддинги + CatBoost | текст | — | — | — |
| Эмбеддинги + CatBoost | текст + метаданные | — | — | — |
| RoBERTa / DeBERTa-v3 | текст | — | — | — |
| RoBERTa / DeBERTa-v3 | текст + метаданные | — | — | — |

### Цена (`log(price)`)

| Модель | Вход | MAE ↓ | R² ↑ |
|---|---|---|---|
| Среднее | — | — | — |
| CatBoost | метаданные | — | — |
| TF-IDF + Ridge | текст | — | — |
| Эмбеддинги + CatBoost | текст + метаданные | — | — |
| RoBERTa / DeBERTa-v3 | текст + метаданные | — | — |

### Сравнение с previous art

| Работа | Постановка | Их результат | Наш результат |
|---|---|---|---|
| IACIS 2022 | регрессия `points`, подвыборка (12 стран, 26 сортов, 12 дегустаторов) | R² = 0.593 | — |
| Journal of Wine Economics (BERT) | классификация оценок, 2 класса | — | — |
| Journal of Wine Economics (BERT) | классификация оценок, 4 класса | — | — |

У Journal of Wine Economics другой датасет, поэтому это сравнение служит ориентиром, а не строгим SotA.
