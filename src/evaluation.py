"""Metrik dihitung dari prediksi test set vs label sebenarnya."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_predictions(y_true, y_pred, class_names: list[str]) -> dict[str, Any]:
    """Accuracy, macro & weighted P/R/F1, metrik per kelas, dan confusion matrix berlabel."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true dan y_pred harus berukuran sama dan tidak kosong.")
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=class_names, zero_division=0)
    result: dict[str, Any] = {"accuracy": float(accuracy_score(y_true, y_pred)), "n_test": int(len(y_true))}
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, labels=class_names, average=avg, zero_division=0)
        result[f"precision_{avg}"], result[f"recall_{avg}"], result[f"f1_{avg}"] = float(pa), float(ra), float(fa)
    result["per_class"] = {
        c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(class_names)
    }
    result["classes"] = list(class_names)
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_names).tolist()
    return result


def measure_inference_ms(model, X, repeats: int = 3) -> float:
    """Rata-rata waktu prediksi per sampel (ms), tidak termasuk ekstraksi fitur."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        best = min(best, time.perf_counter() - t0)
    return best / len(X) * 1000.0
