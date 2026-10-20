"""Laporan eksperimen (Markdown) dari config + metrik yang tersimpan. Tidak menambahkan klaim di luar angka terukur."""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def build_report(experiments: dict[str, dict[str, Any]]) -> str:
    """``experiments``: nama eksperimen -> {"config": ..., "metrics": ...} (hasil ``load_experiment``)."""
    if not experiments:
        raise ValueError("Tidak ada eksperimen untuk dilaporkan.")
    first = next(iter(experiments.values()))["config"]
    lines = ["# Laporan Eksperimen AgroVision AI", "",
             f"- Dataset: `{first['dataset_dir']}`; kelas: {', '.join(first['classes'])}",
             f"- Gambar latih: {first['n_train_images']}; gambar uji: {first['n_test_images']} (test_size={first['test_size']}, seed={first['seed']})",
             f"- Distribusi kelas: {first['dataset_audit']['per_class']}; rasio ketidakseimbangan: {first['dataset_audit']['imbalance_ratio']}",
             f"- Augmentasi hanya pada data latih: {first['augment_train_only']}", "",
             "## Perbandingan model (test set)", "",
             "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Inferensi ms/gambar | Ukuran KB |",
             "|---|---|---|---|---|---|---|"]
    for name, exp in experiments.items():
        m = exp["metrics"]
        lines.append(f"| {exp['config']['model']} | {_fmt(m['accuracy'])} | {_fmt(m['precision_macro'])} | {_fmt(m['recall_macro'])} | "
                     f"{_fmt(m['f1_macro'])} | {_fmt(m['inference_ms_per_image'])} | {_fmt(m['model_size_kb'], 1)} |")
    for name, exp in experiments.items():
        m, c = exp["metrics"], exp["config"]
        lines += ["", f"## {c['model']}", "", f"Konfigurasi dan versi: seed={c['seed']}, versi={c['versions']}, checkpoint `{c['checkpoint']}`.", "",
                  "| Kelas | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} | {v['support']} |" for k, v in m["per_class"].items()]
        lines += ["", "Confusion matrix (baris = label sebenarnya, kolom = prediksi; urutan: " + ", ".join(m["classes"]) + "):", "",
                  "```", *[" ".join(f"{v:4d}" for v in row) for row in m["confusion_matrix"]], "```"]
        if "best_epoch" in m:
            lines.append(f"\nEpoch terbaik (validation loss pada data latih): {m['best_epoch']} dari {m['epochs_run']} epoch dijalankan.")
    lines += ["", "## Keterbatasan", "",
              f"- Test set hanya {first['n_test_images']} gambar: selisih beberapa poin persen antar model dapat berupa variasi acak, bukan keunggulan sebenarnya.",
              "- Tidak ada metadata daun/sesi pemotretan: gambar dari daun yang sama dapat jatuh di train dan test sehingga skor mungkin optimistis.",
              "- Skor model bukan probabilitas terkalibrasi; prediksi bukan diagnosis agronomis.",
              "- Model hanya mengenal kelas pada data latihnya. Hasil pada foto lapangan (latar, cahaya berbeda) belum diuji."]
    return "\n".join(lines) + "\n"
