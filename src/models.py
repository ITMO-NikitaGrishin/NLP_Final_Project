"""Модели, общие для нескольких ноутбуков: CatBoost и TF-IDF + Ridge."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .config import META_COLS, SEED


def meta_features(df: pd.DataFrame, cols: Sequence[str] = META_COLS) -> pd.DataFrame:
    """Категориальные метаданные для CatBoost: пропуск становится отдельной категорией "NA"."""
    return df[list(cols)].fillna("NA").astype(str)


CATBOOST_DEFAULTS = dict(
    iterations=15000,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=SEED,
    od_type="Iter",
    od_wait=200,
    use_best_model=True,
    verbose=0,
    thread_count=-1,
    allow_writing_files=False,
)


def fit_catboost(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    cat_features: Sequence[str] | None = None,
    **params,
) -> CatBoostRegressor:
    """CatBoost-регрессор с early stopping по val (лучшая итерация выбирается по val RMSE)."""
    model = CatBoostRegressor(**{**CATBOOST_DEFAULTS, **params})
    cat = list(cat_features) if cat_features is not None else None
    model.fit(
        Pool(X_train, y_train, cat_features=cat),
        eval_set=Pool(X_val, y_val, cat_features=cat),
    )
    return model


def tfidf_vectorizer(**params) -> TfidfVectorizer:
    """TF-IDF по словам и биграммам; sublinear_tf гасит влияние повторов слова."""
    defaults = dict(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b",
    )
    return TfidfVectorizer(**{**defaults, **params})


def fit_ridge(
    X_train, y_train, X_val, y_val, alphas: Sequence[float] = (0.3, 1, 3, 10, 30)
) -> tuple[Ridge, pd.Series]:
    """Подбирает alpha по val RMSE. Возвращает лучшую модель и RMSE на val для каждой alpha."""
    scores, models = {}, {}
    for a in alphas:
        m = Ridge(alpha=a, random_state=SEED).fit(X_train, y_train)
        scores[a] = float(np.sqrt(np.mean((m.predict(X_val) - np.asarray(y_val)) ** 2)))
        models[a] = m
    scores = pd.Series(scores, name="val_RMSE").rename_axis("alpha")
    return models[scores.idxmin()], scores
