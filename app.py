"""AgroVision AI - klasifikasi kondisi daun dan laboratorium perbandingan model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dataset import audit_dataset, label_statistics, scan_dataset
from src.experiment_store import list_experiments, load_experiment
from src.inference import list_checkpoints, load_checkpoint, predict_image
from src.models import MODEL_LABELS, list_available_models
from src.reporting import build_report
from src.preprocessing import ImageError
from src.training import run_experiment

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="AgroVision AI", page_icon="🌿", layout="wide")
st.title("AgroVision AI")
st.caption("Klasifikasi kondisi daun dan laboratorium perbandingan model. Prediksi BUKAN diagnosis agronomis pasti.")


def confusion_figure(matrix, classes):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    peak = max(max(r) for r in matrix) or 1
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > peak / 2 else "black")
    fig.tight_layout()
    return fig


tab_exp, tab_cmp, tab_pred, tab_about = st.tabs(["Eksperimen", "Perbandingan", "Prediksi", "Dataset & keterbatasan"])

with tab_exp:
    st.subheader("Jalankan eksperimen")
    data_dir = st.text_input("Folder dataset (struktur data/<kelas>/<gambar>)", str(ROOT / "data" / ("plantvillage_tomato" if (ROOT / "data" / "plantvillage_tomato").exists() else "synthetic_leaves")))
    if st.button("Audit dataset"):
        try:
            frame = scan_dataset(data_dir)
            audit = audit_dataset(frame)
            for e in audit["errors"]:
                st.error(e)
            for w in audit["warnings"]:
                st.warning(w)
            if audit["per_class"]:
                st.dataframe(label_statistics(frame), hide_index=True)
        except FileNotFoundError as exc:
            st.error(f"{exc}. Buat data contoh sintetis: `python scripts/make_synthetic_dataset.py`.")
    available = list_available_models()
    if "cnn" not in available:
        st.info("CNN dinonaktifkan karena PyTorch belum terpasang (`pip install torch`).")
    models = st.multiselect("Model (semua memakai split yang sama)", available, default=available, format_func=MODEL_LABELS.get)
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Proporsi test set", 0.1, 0.5, 0.2, 0.05)
    seed = int(c2.number_input("Random seed", value=42, step=1))
    augment = c3.checkbox("Augmentasi (hanya data training)", value=True)
    cnn_epochs = int(st.number_input("Maks. epoch CNN (early stopping pada validasi dari data latih)", 5, 200, 40, 5))
    if st.button("Latih & evaluasi", type="primary", disabled=not models):
        try:
            with st.spinner("Mengekstraksi fitur dan melatih model..."):
                outcome = run_experiment(data_dir, models, seed, test_size, augment, ARTIFACTS, cnn_epochs)
            st.session_state["last_outcome"] = outcome
            st.success(f"Selesai: {len(models)} model dievaluasi pada test set yang tidak dipakai untuk training.")
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Eksperimen gagal: {exc}")
    outcome = st.session_state.get("last_outcome")
    if outcome:
        rows = [{"model": k, "accuracy": m["metrics"]["accuracy"], "precision (macro)": m["metrics"]["precision_macro"],
                 "recall (macro)": m["metrics"]["recall_macro"], "F1 (macro)": m["metrics"]["f1_macro"],
                 "inferensi ms/gambar": m["metrics"]["inference_ms_per_image"], "ukuran KB": m["metrics"]["model_size_kb"],
                 "n test": m["metrics"]["n_test"]} for k, m in outcome["results"].items()]
        table = pd.DataFrame(rows).round(4)
        st.dataframe(table, hide_index=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        width = 0.2
        for i, col in enumerate(["accuracy", "precision (macro)", "recall (macro)", "F1 (macro)"]):
            ax.bar([x + i * width for x in range(len(table))], table[col], width, label=col)
        ax.set_xticks([x + 1.5 * width for x in range(len(table))], table["model"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Skor pada test set")
        ax.set_title("Perbandingan metrik per model")
        ax.legend(fontsize=7, loc="lower right")
        st.pyplot(fig)
        for name, res in outcome["results"].items():
            with st.expander(f"Detail {name}"):
                m = res["metrics"]
                st.dataframe(pd.DataFrame(m["per_class"]).T.round(4))
                st.pyplot(confusion_figure(m["confusion_matrix"], m["classes"]))
        st.caption("Jumlah sampel test kecil -> selisih antar model bisa sekadar variasi. Ini bukan klaim performa pada daun nyata.")

with tab_cmp:
    st.subheader("Riwayat eksperimen tersimpan")
    names = list_experiments(ARTIFACTS / "experiments")
    if not names:
        st.info("Belum ada eksperimen tersimpan.")
    else:
        table = []
        for n in names:
            e = load_experiment(ARTIFACTS / "experiments", n)
            table.append({"eksperimen": n, "model": e["config"]["model"], "seed": e["config"]["seed"],
                          "test_size": e["config"]["test_size"], "n_test": e["metrics"]["n_test"],
                          "accuracy": e["metrics"]["accuracy"], "F1 macro": e["metrics"]["f1_macro"]})
        st.dataframe(pd.DataFrame(table).round(4), hide_index=True)
        pick = st.selectbox("Lihat konfigurasi lengkap", names)
        st.json(load_experiment(ARTIFACTS / "experiments", pick)["config"], expanded=False)

        pred_file = ARTIFACTS / "experiments" / pick / "predictions.csv"
        if pred_file.exists():
            preds = pd.read_csv(pred_file)
            wrong, right = preds[preds["label"] != preds["prediksi"]], preds[preds["label"] == preds["prediksi"]]
            st.markdown(f"**Contoh prediksi** (test set): {len(right)} benar, {len(wrong)} salah")
            for title, subset in (("Salah", wrong), ("Benar", right)):
                st.caption(title)
                cols = st.columns(4)
                for col, (_, row) in zip(cols, subset.head(4).iterrows()):
                    if Path(row["path"]).exists():
                        col.image(row["path"], caption=f"asli: {row['label']} | prediksi: {row['prediksi']}", width=150)
        chosen = st.multiselect("Eksperimen untuk laporan", names, default=names[-4:])
        if chosen:
            st.download_button("Unduh laporan eksperimen (Markdown)",
                               build_report({n: load_experiment(ARTIFACTS / "experiments", n) for n in chosen}),
                               "laporan_eksperimen.md", "text/markdown")

with tab_pred:
    st.subheader("Prediksi gambar daun")
    ckpts = list_checkpoints(ARTIFACTS / "models")
    if not ckpts:
        st.info("Belum ada model. Jalankan eksperimen di tab 'Eksperimen' terlebih dahulu.")
    else:
        ckpt = st.selectbox("Checkpoint", ckpts, format_func=lambda p: p.name)
        upload = st.file_uploader("Unggah gambar (JPG/PNG/BMP, maks 10 MB)", type=["jpg", "jpeg", "png", "bmp"])
        if upload is not None:
            try:
                result = predict_image(load_checkpoint(ckpt), upload.getvalue(), upload.name)
            except ImageError as exc:
                st.error(str(exc))
            else:
                st.image(upload, width=240)
                st.metric("Kelas prediksi", result["predicted_class"], f"skor model {result['confidence']:.0%}", delta_color="off")
                if result["needs_verification"]:
                    st.warning("Skor rendah: hasil PERLU VERIFIKASI manual.")
                st.dataframe(pd.DataFrame(result["top"]), hide_index=True)
                st.caption(result["disclaimer"])
                bundle = load_checkpoint(ckpt)
                if bundle.get("kind") == "cnn":
                    from src.cnn import grad_cam, overlay_cam
                    from src.preprocessing import open_image

                    img = open_image(upload.getvalue(), upload.name)
                    cam, _ = grad_cam(bundle["model"], img)
                    st.image(overlay_cam(img, cam), width=240,
                             caption="Grad-CAM: area yang paling memengaruhi skor kelas prediksi (merah = tinggi). "
                                     "Penjelasan model, bukan bukti kausal penyakit.")
        st.caption("Model hanya mengenal kelas yang ada pada data latihnya; gambar non-daun tetap akan diberi salah satu kelas.")

with tab_about:
    st.markdown(
        """
**Dataset contoh sintetis bawaan** (`scripts/make_synthetic_dataset.py`), dibuat untuk menguji alur. Skor tinggi pada
data ini tidak menunjukkan performa pada daun nyata. Untuk hasil bermakna gunakan dataset publik berlisensi jelas
(lihat `docs/DATASET_CARD.md`) dengan struktur `data/<kelas>/*.jpg`.

**Keterbatasan:** model klasik memakai fitur warna/tekstur global; CNN kecil dilatih dari nol (tanpa bobot pralatih);
skor model bukan probabilitas terkalibrasi; test set kecil; tanpa metadata daun sehingga kebocoran antar-foto-daun-sama mungkin terjadi; evaluasi macro-F1/recall per kelas lebih informatif daripada akurasi saat kelas tidak seimbang.
"""
    )
