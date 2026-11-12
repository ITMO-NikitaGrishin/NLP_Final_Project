"""Единое разбиение train/val/test 80/10/10 для всех экспериментов."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PROCESSED, POINTS_COL, SEED


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.10,
    val_size: float = 0.10,
    stratify_col: str | None = POINTS_COL,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat = df[stratify_col] if stratify_col else None
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat
    )
    strat = train_val[stratify_col] if stratify_col else None
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=strat,
    )
    return train, val, test


def save_splits(train, val, test) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_parquet(DATA_PROCESSED / f"{name}.parquet")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает сохранённые сплиты; если их нет — создаёт и сохраняет."""
    paths = [DATA_PROCESSED / f"{n}.parquet" for n in ("train", "val", "test")]
    if not all(p.exists() for p in paths):
        from .data import load_clean

        save_splits(*make_splits(load_clean()))
    return tuple(pd.read_parquet(p) for p in paths)


def load_iacis_splits(
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Подвыборка IACIS 2022, отфильтрованная из основных сплитов.

    Строки остаются в той же части (train/val/test), что и в основном
    разбиении, поэтому test подвыборки не пересекается с train основных моделей.
    """
    from .data import filter_subset, iacis_categories

    path = DATA_PROCESSED / "iacis.parquet"
    if rebuild or not path.exists():
        train, val, test = load_splits()
        full = pd.concat(
            [p.assign(split=n) for n, p in
             {"train": train, "val": val, "test": test}.items()]
        )
        cats = iacis_categories(full)
        subset = filter_subset(
            full, cats["country"], cats["variety"], cats["taster_name"]
        )
        subset.to_parquet(path)
    subset = pd.read_parquet(path)
    return tuple(
        subset[subset["split"] == n].drop(columns="split").reset_index(drop=True)
        for n in ("train", "val", "test")
    )
