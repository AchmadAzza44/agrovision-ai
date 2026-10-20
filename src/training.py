"""Alur eksperimen: fitur -> split tetap -> latih -> evaluasi di test set -> simpan checkpoint & catatan."""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .dataset import audit_dataset, clean_frame, scan_dataset, split_dataset
from .evaluation import evaluate_predictions, measure_inference_ms
from .experiment_store import save_experiment
from .models import create_model
from .preprocessing import augment_pil, extract_features, open_image


def build_features(frame: pd.DataFrame, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Ekstraksi fitur; ``augment=True`` menambah varian flip/rotasi (hanya untuk TRAIN)."""
    X, y = [], []
    for path, label in zip(frame["path"], frame["label"]):
        image = open_image(path)
        X.append(extract_features(image))
        y.append(label)
        if augment:
            for variant in augment_pil(image):
                X.append(extract_features(variant))
                y.append(label)
    return np.vstack(X), np.asarray(y)


def run_experiment(data_dir: str | Path, model_names: list[str], seed: int = 42, test_size: float = 0.2,
                   augment: bool = True, out_dir: str | Path = "artifacts", cnn_epochs: int = 20) -> dict[str, Any]:
    """Latih & bandingkan model pada split yang sama. Mengembalikan ringkasan per model."""
    raw = scan_dataset(data_dir)
    audit = audit_dataset(raw)
    if audit["errors"]:
        raise ValueError("Dataset tidak layak: " + " ".join(audit["errors"]))
    frame = clean_frame(raw)
    train_df, test_df = split_dataset(frame, test_size, seed)
    classes = sorted(frame["label"].unique())

    t0 = time.perf_counter()
    X_train, y_train = build_features(train_df, augment=augment)
    X_test, y_test = build_features(test_df, augment=False)  # test TIDAK diaugmentasi
    feat_ms = (time.perf_counter() - t0) / (len(train_df) + len(test_df)) * 1000

    out_dir = Path(out_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: dict[str, Any] = {}
    for name in model_names:
        if name == "cnn":
            from .cnn import train_and_evaluate

            ckpt = out_dir / "models" / f"cnn_{stamp}.pt"
            metrics, y_pred = train_and_evaluate(train_df, test_df, classes, seed, ckpt, epochs=cnn_epochs)
            metrics["feature_extraction_ms_per_image"] = None
            n_train_samples = int(len(train_df))
        else:
            model = create_model(name, seed)
            t_fit = time.perf_counter()
            model.fit(X_train, y_train)
            fit_s = time.perf_counter() - t_fit
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, classes)
            metrics.update(fit_seconds=fit_s, inference_ms_per_image=measure_inference_ms(model, X_test),
                           feature_extraction_ms_per_image=feat_ms)
            ckpt = out_dir / "models" / f"{name}_{stamp}.joblib"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "classes": classes, "name": name}, ckpt)
            metrics["model_size_kb"] = round(ckpt.stat().st_size / 1024, 1)
            n_train_samples = int(len(X_train))
        config = {
            "model": name, "seed": seed, "test_size": test_size, "augment_train_only": augment,
            "n_train_images": len(train_df), "n_train_samples_after_augmentation": n_train_samples,
            "n_test_images": len(test_df), "classes": classes, "dataset_dir": str(data_dir),
            "dataset_audit": {k: audit[k] for k in ("per_class", "duplicates", "imbalance_ratio", "warnings")},
            "test_sha256": sorted(test_df["sha256"].tolist()), "checkpoint": str(ckpt),
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
            "created_utc": stamp,
        }
        if name == "cnn":
            import torch

            config["versions"]["torch"] = torch.__version__
            config["cnn_epochs_max"] = cnn_epochs
        exp_dir = save_experiment(out_dir / "experiments", f"{stamp}_{name}", config, metrics)
        pd.DataFrame({"path": test_df["path"], "label": y_test, "prediksi": y_pred}).to_csv(
            exp_dir / "predictions.csv", index=False)
        results[name] = {"metrics": metrics, "checkpoint": str(ckpt)}
    return {"audit": audit, "classes": classes, "results": results}
