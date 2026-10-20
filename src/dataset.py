"""Audit dataset, label dari struktur folder, dan split reproducible tanpa kebocoran duplikat."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import SUPPORTED_EXTENSIONS, ImageError, open_image

MIN_PER_CLASS = 10


def scan_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Baca ``data_dir/<kelas>/<gambar>`` -> DataFrame(path, label, sha256).

    Subfolder ``train``/``test`` tidak dipakai: split dibuat ulang agar seed & rasio tercatat.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for img in sorted(class_dir.rglob("*")):
            if img.is_file() and img.suffix.lower() in SUPPORTED_EXTENSIONS:
                rows.append({"path": str(img), "label": class_dir.name,
                             "sha256": hashlib.sha256(img.read_bytes()).hexdigest()})
    return pd.DataFrame(rows, columns=["path", "label", "sha256"])


def audit_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Periksa kelas, gambar rusak, duplikat (isi identik), dan ketidakseimbangan."""
    report: dict[str, Any] = {"errors": [], "warnings": [], "corrupt": [], "duplicates": 0,
                              "per_class": {}, "imbalance_ratio": None, "total": int(len(frame))}
    if frame.empty:
        report["errors"].append("Tidak ada gambar ditemukan. Susun data sebagai data/<kelas>/<gambar>.jpg.")
        return report
    for path in frame["path"]:
        try:
            open_image(path)
        except ImageError as exc:
            report["corrupt"].append(f"{path}: {exc}")
    if report["corrupt"]:
        report["warnings"].append(f"{len(report['corrupt'])} gambar rusak akan dikeluarkan dari dataset.")
    clean = frame[~frame["path"].isin([c.split(": ")[0] for c in report["corrupt"]])]
    counts = clean["label"].value_counts().to_dict()
    report["per_class"] = {k: int(v) for k, v in sorted(counts.items())}
    if len(counts) < 2:
        report["errors"].append("Minimal 2 kelas diperlukan untuk klasifikasi.")
    small = [k for k, v in counts.items() if v < MIN_PER_CLASS]
    if small:
        report["errors"].append(f"Kelas dengan < {MIN_PER_CLASS} gambar: {', '.join(sorted(small))}.")
    report["duplicates"] = int(clean.duplicated("sha256").sum())
    if report["duplicates"]:
        report["warnings"].append(f"{report['duplicates']} gambar duplikat (isi identik) akan dibuang sebelum split.")
    cross = clean.groupby("sha256")["label"].nunique()
    if (cross > 1).any():
        report["errors"].append(f"{int((cross > 1).sum())} gambar identik memiliki label berbeda (label konflik).")
    if counts:
        report["imbalance_ratio"] = round(max(counts.values()) / max(min(counts.values()), 1), 2)
        if report["imbalance_ratio"] > 2:
            report["warnings"].append(
                f"Kelas tidak seimbang (rasio {report['imbalance_ratio']}); utamakan macro-F1 dan recall per kelas.")
    return report


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Buang gambar rusak dan duplikat isi identik (pertahankan kemunculan pertama)."""
    ok = []
    for path in frame["path"]:
        try:
            open_image(path)
            ok.append(True)
        except ImageError:
            ok.append(False)
    out = frame[pd.Series(ok, index=frame.index)]
    return out.drop_duplicates("sha256").reset_index(drop=True)


def split_dataset(frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratified & reproducible. Test set ditetapkan di sini, SEBELUM tuning model apa pun."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size harus di antara 0.05 dan 0.5")
    train, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame["label"])
    if set(train["sha256"]) & set(test["sha256"]):
        raise AssertionError("Kebocoran data: gambar identik ada di train dan test.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def label_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["label"].value_counts().sort_index()
    return pd.DataFrame({"kelas": counts.index, "jumlah": counts.values, "persen": (counts / counts.sum() * 100).round(1).values})
