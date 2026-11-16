"""Эмбеддинги текста через sentence-transformers с кэшем на диске (эксперимент 4)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED

EMB_DIR = DATA_PROCESSED / "embeddings"


def _cache_path(model_name: str, split: str):
    return EMB_DIR / f"{model_name.replace('/', '__')}__{split}.npy"


def encode(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 128,
    max_length: int = 256,
) -> np.ndarray:
    """Кодирует тексты на GPU (fp16), если он есть. Векторы нормированы на единичную длину."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length
    if device == "cuda":
        model.half()
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb.astype(np.float32)


def load_or_encode(
    splits: dict[str, pd.DataFrame], text_col: str, model_name: str, **kwargs
) -> dict[str, np.ndarray]:
    """Эмбеддинги для каждого сплита; посчитанные один раз сохраняются в data/processed/embeddings."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in splits.items():
        path = _cache_path(model_name, name)
        if path.exists():
            emb = np.load(path)
            if len(emb) == len(df):
                out[name] = emb
                continue
        out[name] = encode(df[text_col], model_name, **kwargs)
        np.save(path, out[name])
    return out


def emb_frame(emb: np.ndarray, index=None, prefix: str = "e") -> pd.DataFrame:
    """Матрица эмбеддингов как DataFrame, чтобы склеить её с категориальными метаданными для CatBoost."""
    return pd.DataFrame(emb, index=index, columns=[f"{prefix}{i}" for i in range(emb.shape[1])])
