"""Inferensi gambar baru dengan preprocessing yang identik dengan training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .preprocessing import extract_features, open_image

LOW_CONFIDENCE = 0.6
DISCLAIMER = ("Skor adalah keluaran model, bukan probabilitas benar yang terkalibrasi, dan prediksi "
              "bukan diagnosis agronomis. Verifikasi dengan ahli.")


def list_checkpoints(models_dir: str | Path) -> list[Path]:
    """Semua checkpoint (.joblib untuk model klasik, .pt untuk CNN), terbaru lebih dulu menurut stempel waktu pada nama."""
    root = Path(models_dir)
    if not root.exists():
        return []
    files = [p for pattern in ("*.joblib", "*.pt") for p in root.glob(pattern)]
    return sorted(files, key=lambda p: (p.stem.split("_", 1)[-1], p.name), reverse=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    if path.suffix == ".pt":
        import torch

        from .cnn import CNNClassifier

        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("kind") != "cnn":
            raise ValueError("Checkpoint tidak valid.")
        return {"model": CNNClassifier.from_state(raw), "classes": raw["classes"], "name": "cnn", "kind": "cnn"}
    bundle = joblib.load(path)
    if not {"model", "classes"} <= set(bundle):
        raise ValueError("Checkpoint tidak valid.")
    return bundle


def predict_image(bundle: dict[str, Any], source, name: str = "", top_k: int = 3) -> dict[str, Any]:
    image = open_image(source, name)
    model = bundle["model"]
    if bundle.get("kind") == "cnn":
        from .cnn import image_to_array

        features = image_to_array(image)[None]
    else:
        features = extract_features(image).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    classes = list(model.classes_)
    order = np.argsort(proba)[::-1][:top_k]
    top = [{"kelas": str(classes[i]), "skor": float(proba[i])} for i in order]
    return {
        "predicted_class": top[0]["kelas"], "confidence": top[0]["skor"], "top": top,
        "needs_verification": top[0]["skor"] < LOW_CONFIDENCE, "disclaimer": DISCLAIMER,
        "model": bundle.get("name", "?"),
    }
