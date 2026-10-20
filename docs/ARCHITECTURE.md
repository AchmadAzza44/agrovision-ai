# Architecture

## Komponen

```
agrovision-ai/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── README.md
├── artifacts/
│   ├── models/
│   └── experiments/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Validasi data, label, split, statistik
│   ├── preprocessing.py    # Resize, normalisasi, augmentasi training
│   ├── models.py           # Implementasi model yang dibandingkan
│   ├── training.py         # Loop training dan checkpoint
│   ├── evaluation.py       # Metrik, confusion matrix, laporan
│   ├── inference.py        # Validasi dan prediksi gambar baru
│   └── experiment_store.py # Simpan konfigurasi dan metrik
├── tests/
└── docs/
    ├── DATASET_CARD.md
    ├── EXPERIMENTS.md
    ├── ARCHITECTURE.md
    └── PROGRESS.md
```

## Alur Data

1. Dataset gambar disimpan di `data/` dengan struktur per kelas.
2. `src/dataset.py` memvalidasi dan membagi dataset (train/test).
3. `src/preprocessing.py` melakukan resize dan normalisasi.
4. `src/models.py` membuat instance model (Random Forest, SVM, CNN).
5. `src/training.py` melatih model dan menyimpan checkpoint.
6. `src/evaluation.py` menghitung metrik pada test set.
7. `src/inference.py` melakukan prediksi pada gambar baru.
8. `src/experiment_store.py` menyimpan konfigurasi dan hasil eksperimen.

## Alasan Pemilihan Teknologi

- **Python:** Bahasa pemrograman utama.
- **Scikit-learn:** Machine learning klasik (Random Forest, SVM).
- **PyTorch/TensorFlow:** Untuk CNN (versi lanjutan).
- **Pillow:** Manipulasi gambar.
- **Streamlit:** Antarmuka web untuk prediksi dan eksperimen.

## Batasan

- Prediksi bukan diagnosis agronomis pasti.
- Confidence model harus disertai peringatan keterbatasan.
- Augmentasi hanya pada data training untuk mencegah kebocoran data.

## Alur data (aktual)
`data/<kelas>/*` -> `scan_dataset` -> `audit_dataset` -> `clean_frame` (buang rusak/duplikat) -> `split_dataset` (test tetap)
-> `extract_features` (train: + augmentasi) -> `create_model(...).fit` -> `evaluate_predictions` pada test -> `save_experiment` + checkpoint (joblib).
Inferensi: `open_image` (validasi) -> `extract_features` (fungsi yang sama) -> `predict_proba`.

## Modul tambahan
- `cnn.py`: SmallCNN, CNNClassifier (antarmuka mirip sklearn), Grad-CAM; hanya aktif bila torch terpasang.
- `reporting.py`: laporan Markdown dari config + metrik tersimpan.

## Keputusan
- Fitur buatan tangan + model klasik agar berjalan di laptop tanpa GPU; CNN kecil dari nol ditambahkan sebagai opsi (torch opsional); transfer learning belum karena bobot pralatih tidak dapat diunduh.
- SVM dikalibrasi (`CalibratedClassifierCV`) untuk skor; kalibrasi tetap tidak menjamin probabilitas benar pada data baru.
