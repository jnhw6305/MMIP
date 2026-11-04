from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device_name() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds = time.perf_counter() - self.start


def classification_metrics(y_true, y_pred, probabilities=None) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    if probabilities is not None:
        try:
            out["macro_auc"] = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
            )
        except ValueError:
            out["macro_auc"] = None
    return out

