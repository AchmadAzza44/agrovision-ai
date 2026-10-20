# AgroVision AI

Klasifikasi Kondisi Daun dan Laboratorium Perbandingan Model.

## Deskripsi

AgroVision AI adalah aplikasi pembelajaran machine learning yang menerima gambar
daun, memprediksi kelas yang telah dipelajari model, dan menyajikan evaluasi
beberapa algoritma pada dataset yang sama. Nilai akademik utamanya bukan hanya
hasil prediksi, melainkan eksperimen yang adil, evaluasi yang benar, dan
dokumentasi keterbatasan.

## Fitur yang sudah berjalan

- Audit dataset: gambar rusak, duplikat isi identik, label konflik, kelas terlalu kecil, ketidakseimbangan.
- Split train/test stratified & reproducible (seed tercatat) tanpa kebocoran duplikat identik; augmentasi hanya untuk train.
- Empat model pada split yang sama: Random Forest, SVM, MLP (fitur warna/tekstur) dan **CNN kecil dari nol (PyTorch, opsional)**. Transfer learning belum dilakukan.
- Grad-CAM untuk model CNN pada tab Prediksi; contoh prediksi benar/salah; analisis per kelas; grafik perbandingan; ekspor laporan Markdown.
- Metrik dari test set: accuracy, precision/recall/F1 macro & weighted, metrik per kelas, confusion matrix, waktu inferensi, ukuran model.
- Riwayat eksperimen (config + seed + versi library + hash test set) dan checkpoint.
- Prediksi gambar baru dengan validasi format/ukuran, skor top-3, penanda "perlu verifikasi" bila skor < 0.6.

## Instalasi dan menjalankan

```bash
pip install -r requirements.txt
python scripts/make_synthetic_dataset.py   # dataset SINTETIS untuk demo
python scripts/fetch_plantvillage.py       # subset foto daun tomat NYATA (butuh git + internet; periksa lisensi)
pip install torch                          # opsional, untuk CNN dan Grad-CAM
python -m streamlit run app.py
python -m pytest -q
```

Di aplikasi: tab *Eksperimen* -> latih -> tab *Prediksi*.

## Dataset

Foto nyata: subset PlantVillage tomat (3 kelas x 300) lewat `fetch_plantvillage.py` (gambar tidak di-commit; lisensi eksplisit tidak tercantum di sumber). Demo sintetis: `make_synthetic_dataset.py`. Struktur umum `data/<kelas>/*.jpg`;
lihat `docs/DATASET_CARD.md`. Hasil percobaan: `docs/EXPERIMENTS.md`.

## Batasan

Prediksi bukan diagnosis agronomis. Skor model bukan probabilitas terkalibrasi. Hasil pada data laboratorium PlantVillage kemungkinan optimistis (kemungkinan kebocoran antar-foto daun sama; latar seragam).
Model hanya mengenal kelas pada data latihnya.
## Struktur folder
```text
app.py                # titik masuk Streamlit (4 tab)
src/                  # dataset, preprocessing, models, cnn (PyTorch + Grad-CAM), training, evaluation, inference, experiment_store, reporting
data/                 # tempat dataset (plantvillage_tomato/ dan synthetic_leaves/ dibuat oleh skrip, tidak di-commit)
artifacts/            # checkpoint model dan catatan eksperimen (dibuat saat training, tidak di-commit)
scripts/              # fetch_plantvillage.py, make_synthetic_dataset.py
tests/                # 30 tes pytest
docs/                 # DATASET_CARD, EXPERIMENTS, ARCHITECTURE, TEST_PLAN, PROGRESS, LAPORAN, screenshots/
```

