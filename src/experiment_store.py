"""Simpan & muat konfigurasi + metrik eksperimen agar dapat direproduksi."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_experiment(experiment_dir: str | Path, experiment_name: str, config: dict[str, Any],
                    metrics: dict[str, Any]) -> Path:
    target = Path(experiment_dir) / experiment_name
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def list_experiments(experiment_dir: str | Path) -> list[str]:
    root = Path(experiment_dir)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "metrics.json").exists())


def load_experiment(experiment_dir: str | Path, experiment_name: str) -> dict[str, Any]:
    root = Path(experiment_dir) / experiment_name
    if not (root / "metrics.json").exists():
        raise FileNotFoundError(f"Eksperimen tidak ditemukan: {experiment_name}")
    return {
        "config": json.loads((root / "config.json").read_text(encoding="utf-8")),
        "metrics": json.loads((root / "metrics.json").read_text(encoding="utf-8")),
    }
