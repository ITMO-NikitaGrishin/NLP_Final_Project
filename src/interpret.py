"""SHAP для дообученной RoBERTa и вспомогательные функции анализа ошибок (эксперимент 9)."""
from __future__ import annotations

import pickle
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED
from .transformer import META_LABELS, Scaler, TrainConfig, load_finetuned, predict

SHAP_DIR = DATA_PROCESSED / "shap"
_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def explain(run_name: str, texts: Sequence[str], scaler: Scaler, output_names: Sequence[str],
            cfg: TrainConfig | None = None, batch_size: int = 128):
    """SHAP-значения (PartitionExplainer по токенам) для модели из models/<run_name>/.

    Результат кэшируется в data/processed/shap/<run_name>.pkl: пересчёт ~1 с на текст на GPU.
    """
    import shap
    import torch

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    path = SHAP_DIR / f"{run_name}.pkl"
    texts = list(texts)
    if path.exists():
        cached = pickle.load(open(path, "rb"))
        if cached["texts"] == texts:
            return cached["sv"]

    cfg = cfg or TrainConfig()
    model, tok = load_finetuned(run_name)
    f = lambda batch: predict(model, tok, scaler, list(batch), cfg)
    explainer = shap.Explainer(f, shap.maskers.Text(tok), output_names=list(output_names))
    sv = explainer(texts, batch_size=batch_size, silent=True)
    pickle.dump({"texts": texts, "sv": sv}, open(path, "wb"))
    del model
    torch.cuda.empty_cache()
    return sv


def word_attributions(sv, output: int) -> pd.DataFrame:
    """Склеивает BPE-токены в слова и суммирует их SHAP. Одна строка на вхождение слова.

    У токенов RoBERTa пробел стоит в конце ("Nap", "a "), поэтому слово заканчивается
    на токене, после которого идёт пробел. Слово приводится к нижнему регистру, пунктуация по краям отбрасывается.
    """
    rows = []
    for doc, (tokens, values) in enumerate(zip(sv.data, sv.values)):
        word, val = "", 0.0
        for t, v in zip(tokens, values[:, output]):
            word += t
            val += v
            if t.endswith((" ", "\n")):
                w = _PUNCT.sub("", word.strip()).lower()
                if w:
                    rows.append((doc, w, val))
                word, val = "", 0.0
        w = _PUNCT.sub("", word.strip()).lower()
        if w:
            rows.append((doc, w, val))
    return pd.DataFrame(rows, columns=["doc", "word", "shap"])


def top_words(words: pd.DataFrame, min_docs: int = 10, n: int = 15) -> pd.DataFrame:
    """Слова с наибольшим средним вкладом в обе стороны среди слов, встретившихся хотя бы в min_docs отзывах."""
    g = words.groupby("word").agg(mean_shap=("shap", "mean"), docs=("doc", "nunique"))
    g = g[g["docs"] >= min_docs]
    return pd.concat([g.nlargest(n, "mean_shap"), g.nsmallest(n, "mean_shap")])


SEGMENTS = [*META_LABELS.values(), "Description"]


def segment_attributions(sv, output: int) -> pd.DataFrame:
    """Сумма SHAP по полям входа text_with_meta: Country, ..., Taster, Description. Строка на отзыв."""
    out = []
    for tokens, values in zip(sv.data, sv.values):
        text = "".join(tokens)
        starts = [(text.index(f"{s}: "), s) for s in SEGMENTS]
        bounds = [b for b, _ in starts] + [len(text) + 1]
        res = dict.fromkeys(SEGMENTS, 0.0)
        pos = 0
        for t, v in zip(tokens, values[:, output]):
            seg = next((s for (b, s), nb in zip(starts, bounds[1:]) if b <= pos < nb), SEGMENTS[0])
            res[seg] += v
            pos += len(t)
        out.append(res)
    return pd.DataFrame(out, columns=SEGMENTS)
