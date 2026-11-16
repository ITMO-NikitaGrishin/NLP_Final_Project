"""Дообучение трансформера (RoBERTa-base) на регрессию, эксперименты 5–6.

Одна модель предсказывает оба таргета сразу (points и log_price): голова с двумя выходами,
таргеты стандартизованы по train, лосс — сумма MSE. Так вдвое меньше прогонов на 8-гигабайтной GPU,
а текстовое представление общее для обоих таргетов.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ROOT, SEED, TEXT_COL

MODELS_DIR = ROOT / "models"

# подписи полей в тексте с метаданными (эксперимент 6)
META_LABELS = {
    "country": "Country",
    "province": "Province",
    "region_1": "Region",
    "variety": "Variety",
    "winery": "Winery",
    "taster_name": "Taster",
}


def text_with_meta(df: pd.DataFrame, cols: Sequence[str] = tuple(META_LABELS)) -> pd.Series:
    """Метаданные дописываются в текст: "Country: Italy. Variety: Nebbiolo. ... Description: ...".

    Пропуски пишем как "unknown", чтобы формат был одинаковым для всех отзывов.
    """
    parts = [META_LABELS[c] + ": " + df[c].fillna("unknown").astype(str) + "." for c in cols]
    prefix = parts[0]
    for p in parts[1:]:
        prefix = prefix + " " + p
    return prefix + " Description: " + df[TEXT_COL]


@dataclass
class TrainConfig:
    model_name: str = "roberta-base"
    max_length: int = 192
    batch_size: int = 32
    eval_batch_size: int = 128
    lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.06
    evals_per_epoch: int = 4
    seed: int = SEED


class _TextDataset(Dataset):
    def __init__(self, texts, targets=None):
        self.texts = list(texts)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return i


class Scaler:
    """Стандартизация таргетов по train; обратное преобразование для предсказаний."""

    def __init__(self, y: np.ndarray):
        self.mean, self.std = y.mean(0), y.std(0)

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse(self, y):
        return y * self.std + self.mean


def _loader(ds: _TextDataset, tokenizer, batch_size: int, max_length: int, shuffle: bool, seed: int):
    def collate(idx):
        enc = tokenizer(
            [ds.texts[i] for i in idx],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        if ds.targets is not None:
            enc["labels"] = torch.from_numpy(ds.targets[idx])
        return enc

    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict_scaled(model, tokenizer, texts, cfg: TrainConfig) -> np.ndarray:
    """Предсказания в стандартизованной шкале. Тексты сортируются по длине, чтобы меньше паддинга."""
    model.eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    ds = _TextDataset([texts[i] for i in order])
    out = []
    for batch in _loader(ds, tokenizer, cfg.eval_batch_size, cfg.max_length, False, cfg.seed):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**batch).logits.float().cpu().numpy())
    pred = np.concatenate(out)
    res = np.empty_like(pred)
    res[order] = pred
    return res


def train_regressor(
    train_texts,
    y_train: np.ndarray,
    val_texts,
    y_val: np.ndarray,
    cfg: TrainConfig,
    run_name: str,
):
    """Дообучает модель на стандартизованных таргетах (n × k), отбор лучшего чекпойнта по val MSE.

    Возвращает (model, tokenizer, scaler, history). Лучшие веса сохраняются в models/<run_name>/.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    y_train, y_val = np.asarray(y_train, np.float32), np.asarray(y_val, np.float32)
    n_targets = y_train.shape[1]
    scaler = Scaler(y_train)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=n_targets, problem_type="regression"
    ).to("cuda")

    train_dl = _loader(
        _TextDataset(train_texts, scaler.transform(y_train)),
        tokenizer, cfg.batch_size, cfg.max_length, True, cfg.seed,
    )
    # у свежей головы lr выше, чем у предобученного энкодера
    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body_decay = [p for n, p in model.named_parameters()
                  if not n.startswith("classifier") and not any(k in n for k in ("bias", "LayerNorm"))]
    body_nodecay = [p for n, p in model.named_parameters()
                    if not n.startswith("classifier") and any(k in n for k in ("bias", "LayerNorm"))]
    optim = torch.optim.AdamW([
        {"params": body_decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": body_nodecay, "lr": cfg.lr, "weight_decay": 0.0},
        {"params": head, "lr": cfg.head_lr, "weight_decay": 0.0},
    ])
    total = len(train_dl) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total), total)
    eval_every = math.ceil(len(train_dl) / cfg.evals_per_epoch)

    out_dir = MODELS_DIR / run_name
    best, history, step, t0, running = math.inf, [], 0, time.time(), []
    val_texts = list(val_texts)
    y_val_s = scaler.transform(y_val)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            loss = torch.nn.functional.mse_loss(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running.append(loss.item())
            step += 1

            if step % eval_every == 0 or step == total:
                pred = predict_scaled(model, tokenizer, val_texts, cfg)
                val_mse = ((pred - y_val_s) ** 2).mean(0)
                rec = {"step": step, "epoch": step / len(train_dl),
                       "train_loss": float(np.mean(running)), "val_loss": float(val_mse.sum()),
                       "minutes": (time.time() - t0) / 60}
                rec.update({f"val_mse_{j}": float(v) for j, v in enumerate(val_mse)})
                history.append(rec)
                running = []
                improved = rec["val_loss"] < best
                if improved:
                    best = rec["val_loss"]
                    model.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                print(f"epoch {rec['epoch']:.2f}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}"
                      f"{'  *' if improved else ''}  {rec['minutes']:.1f} мин", flush=True)
                model.train()

    # возвращаем лучший чекпойнт
    del model
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to("cuda")
    return model, tokenizer, scaler, pd.DataFrame(history)


def predict(model, tokenizer, scaler: Scaler, texts, cfg: TrainConfig) -> np.ndarray:
    """Предсказания в исходной шкале таргетов (n × k)."""
    return scaler.inverse(predict_scaled(model, tokenizer, texts, cfg))
