"""Model yang dibandingkan. Preprocessing & fitur sama untuk semua; hanya algoritma yang berbeda."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_LABELS = {
    "random_forest": "Random Forest (fitur warna/tekstur)",
    "svm": "SVM RBF (fitur warna/tekstur)",
    "mlp": "MLP jaringan saraf (fitur warna/tekstur; BUKAN CNN)",
    "cnn": "CNN kecil dari nol (gambar langsung, PyTorch)",
}


def create_model(name: str, seed: int = 42) -> Pipeline:
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    elif name == "svm":
        clf = CalibratedClassifierCV(SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed), cv=3, ensemble=False)
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    elif name == "cnn":
        raise NotImplementedError("CNN bekerja pada gambar, bukan fitur: gunakan src.cnn.train_and_evaluate.")
    else:
        raise ValueError(f"Model tidak dikenal: {name}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def list_available_models() -> list[str]:
    """Model yang dapat dijalankan; CNN hanya bila PyTorch terpasang."""
    from .cnn import cnn_available

    return [m for m in MODEL_LABELS if m != "cnn" or cnn_available()]
