"""Загрузка и подготовка данных (этап 0)."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import LOG_PRICE_COL, POINTS_COL, PRICE_COL, RAW_FILE, TEXT_COL


def load_raw() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_FILE} не найден. Запустите: python data/download.py"
        )
    return pd.read_csv(RAW_FILE, index_col=0)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубликаты отзывов и строки без цены, добавляет log(price)."""
    df = df.dropna(subset=[TEXT_COL, POINTS_COL, PRICE_COL])
    df = df.drop_duplicates(subset=[TEXT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df[LOG_PRICE_COL] = np.log(df[PRICE_COL])
    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def filter_subset(
    df: pd.DataFrame,
    countries: Iterable[str],
    varieties: Iterable[str],
    tasters: Iterable[str],
) -> pd.DataFrame:
    """Подвыборка по условиям статьи IACIS 2022 (12 стран, 26 сортов, 12 дегустаторов).

    Конкретные списки нужно взять из статьи.
    """
    mask = (
        df["country"].isin(countries)
        & df["variety"].isin(varieties)
        & df["taster_name"].isin(tasters)
    )
    return df[mask].reset_index(drop=True)
