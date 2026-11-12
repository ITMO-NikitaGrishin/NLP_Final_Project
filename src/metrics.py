"""Метрики и запись результатов в results/metrics.csv.

Формат файла (long): experiment, target, metric, value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .config import METRICS_FILE

COLUMNS = ["experiment", "target", "metric", "value"]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE / MAE / R². Для цены передавать log(price)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Для сравнения с Journal of Wine Economics (2 и 4 класса оценок)."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def load_metrics() -> pd.DataFrame:
    if METRICS_FILE.exists():
        return pd.read_csv(METRICS_FILE)
    return pd.DataFrame(columns=COLUMNS)


def log_metrics(experiment: str, target: str, metrics: dict[str, float]) -> pd.DataFrame:
    """Добавляет метрики в results/metrics.csv.

    Строки с тем же (experiment, target, metric) перезаписываются,
    поэтому повторный запуск ноутбука не плодит дубликаты.
    """
    new = pd.DataFrame(
        [(experiment, target, m, v) for m, v in metrics.items()], columns=COLUMNS
    )
    df = load_metrics()
    if not df.empty:
        key = ["experiment", "target", "metric"]
        old_keys = df[key].apply(tuple, axis=1)
        new_keys = set(new[key].apply(tuple, axis=1))
        df = df[~old_keys.isin(new_keys)]
    df = pd.concat([df, new], ignore_index=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_FILE, index=False)
    return df


def results_table(target: str) -> pd.DataFrame:
    """Сводная таблица эксперимент × метрика для одного таргета."""
    df = load_metrics()
    return df[df["target"] == target].pivot(
        index="experiment", columns="metric", values="value"
    )
