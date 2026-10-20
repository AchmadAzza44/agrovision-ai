"""CNN kecil (PyTorch) + Grad-CAM. Opsional: hanya aktif bila `torch` terpasang.

Dilatih dari nol pada gambar yang sama dengan model klasik dan split yang sama. Bobot pralatih (transfer
learning) tidak dipakai: dilaporkan apa adanya sebagai batasan, bukan diklaim.
Seleksi epoch memakai validation split dari data TRAIN; test set tidak pernah dipakai untuk tuning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .evaluation import evaluate_predictions
from .preprocessing import open_image

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - dipakai bila torch tidak terpasang
    torch = None
    nn = None
    TORCH_AVAILABLE = False

IMAGE_SIZE = 96
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cnn_available() -> bool:
    return TORCH_AVAILABLE


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("CNN memerlukan PyTorch: pip install torch")


def image_to_array(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """PIL RGB -> float32 CHW ternormalisasi. Jalur yang sama untuk training dan inferensi."""
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)


def load_arrays(frame: pd.DataFrame, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([image_to_array(open_image(p), size) for p in frame["path"]])
    return X, frame["label"].to_numpy()


if TORCH_AVAILABLE:

    class SmallCNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()

            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

            self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), nn.Conv2d(64, 64, 3, padding=1),
                                          nn.BatchNorm2d(64), nn.ReLU())  # lapisan konv terakhir = target Grad-CAM
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

        def forward(self, x):
            return self.head(self.features(x))


def _augment_batch(x):
    """Flip/rotasi 90 acak per batch - HANYA dipanggil pada data training."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    return torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), dims=[2, 3])


class CNNClassifier:
    """Antarmuka mirip sklearn (predict, predict_proba, classes_) di atas SmallCNN."""

    def __init__(self, classes: list[str], seed: int = 42, epochs: int = 20, batch_size: int = 32, lr: float = 2e-3):
        _require_torch()
        self.classes_ = np.array(sorted(classes))
        self.seed, self.epochs, self.batch_size, self.lr = seed, epochs, batch_size, lr
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0
        torch.manual_seed(seed)
        self.net = SmallCNN(len(self.classes_))

    def _y(self, labels):
        index = {c: i for i, c in enumerate(self.classes_)}
        return torch.tensor([index[l] for l in labels], dtype=torch.long)

    def fit(self, X: np.ndarray, y: np.ndarray, val_fraction: float = 0.15, augment: bool = True) -> "CNNClassifier":
        from sklearn.model_selection import train_test_split

        tr_idx, va_idx = train_test_split(np.arange(len(X)), test_size=val_fraction, random_state=self.seed, stratify=y)
        Xtr, ytr = torch.from_numpy(X[tr_idx]), self._y(y[tr_idx])
        Xva, yva = torch.from_numpy(X[va_idx]), self._y(y[va_idx])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            order = torch.randperm(len(Xtr), generator=gen)
            for i in range(0, len(order), self.batch_size):
                idx = order[i : i + self.batch_size]
                xb = _augment_batch(Xtr[idx]) if augment else Xtr[idx]
                opt.zero_grad()
                loss_fn(self.net(xb), ytr[idx]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                out = self.net(Xva)
                val_loss = float(loss_fn(out, yva))
                val_acc = float((out.argmax(1) == yva).float().mean())
            self.history.append({"epoch": epoch, "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)})
            if val_loss < best - 1e-4:
                best, stale, self.best_epoch = val_loss, 0, epoch
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= 5:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.from_numpy(np.asarray(X, dtype=np.float32))), dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]

    def state(self) -> dict[str, Any]:
        return {"kind": "cnn", "state_dict": self.net.state_dict(), "classes": list(self.classes_), "size": IMAGE_SIZE,
                "best_epoch": self.best_epoch, "history": self.history}

    @classmethod
    def from_state(cls, bundle: dict[str, Any]) -> "CNNClassifier":
        obj = cls(bundle["classes"])
        obj.net.load_state_dict(bundle["state_dict"])
        obj.net.eval()
        obj.best_epoch, obj.history = bundle.get("best_epoch", 0), bundle.get("history", [])
        return obj


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, classes: list[str], seed: int, ckpt: Path,
                       epochs: int = 20) -> tuple[dict[str, Any], np.ndarray]:
    """Latih CNN pada TRAIN, evaluasi sekali pada TEST dengan metrik yang sama seperti model klasik."""
    _require_torch()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    Xtr, ytr = load_arrays(train_df)
    Xte, yte = load_arrays(test_df)
    t0 = time.perf_counter()
    model = CNNClassifier(classes, seed=seed, epochs=epochs).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    y_pred = model.predict(Xte)
    metrics = evaluate_predictions(yte, y_pred, classes)
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        model.predict(Xte)
        best = min(best, time.perf_counter() - t)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state(), ckpt)
    metrics.update(fit_seconds=fit_s, inference_ms_per_image=best / len(Xte) * 1000, best_epoch=model.best_epoch,
                   epochs_run=len(model.history), model_size_kb=round(ckpt.stat().st_size / 1024, 1))
    return metrics, y_pred


def grad_cam(model: "CNNClassifier", image: Image.Image, target: int | None = None) -> tuple[np.ndarray, int]:
    """Peta Grad-CAM (H x W, 0..1) untuk kelas target (default: kelas prediksi). Penjelasan model, bukan bukti kausal."""
    _require_torch()
    net = model.net.eval()
    x = torch.from_numpy(image_to_array(image)[None]).requires_grad_(True)
    feats = net.features(x)
    feats.retain_grad()
    logits = net.head(feats)
    cls = int(logits.argmax(1)) if target is None else target
    net.zero_grad()
    logits[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(1))[0].detach().numpy()
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)) / 255.0, cls


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize((IMAGE_SIZE * 2, IMAGE_SIZE * 2))
    heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(base.size, Image.Resampling.BILINEAR)) / 255.0
    color = np.stack([heat, 0.2 * (1 - heat), 1 - heat], axis=2) * 255  # biru (rendah) -> merah (tinggi)
    return Image.fromarray(((1 - alpha) * np.asarray(base) + alpha * color).astype(np.uint8))
