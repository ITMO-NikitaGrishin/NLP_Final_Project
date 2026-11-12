"""Метрики и сохранение результатов в общую таблицу."""
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

from .config import TABLES

RESULTS_FILE = TABLES / "results.csv"


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE / MAE / R². Для цены передавать log(price)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Для сравнения с Journal of Wine Economics (2 и 4 класса оценок)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def log_result(
    model: str,
    target: str,
    split: str,
    metrics: dict[str, float],
    notebook: str = "",
) -> pd.DataFrame:
    """Добавляет (или перезаписывает) строку в results/tables/results.csv."""
    TABLES.mkdir(parents=True, exist_ok=True)
    row = {"model": model, "target": target, "split": split, "notebook": notebook, **metrics}
    key = ["model", "target", "split"]
    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
        same = (df[key] == pd.Series(row)[key]).all(axis=1)
        df = pd.concat([df[~same], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_FILE, index=False)
    return df
