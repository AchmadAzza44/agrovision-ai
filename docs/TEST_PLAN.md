# Test Plan
`python -m pytest -q` -> **30 lulus** (Python 3.13, scikit-learn 1.9, PyTorch 2.14 untuk tes CNN; tes CNN dilewati otomatis bila torch tidak terpasang).
| Area | Yang diuji | Tes |
|---|---|---|
| Dataset | label sama dengan nama folder; audit mendeteksi gambar rusak, duplikat identik, label konflik, kelas terlalu kecil, dataset kosong/satu kelas | test_scan_*, test_audit_* |
| Split | reproducible dengan seed, stratified, tanpa kebocoran hash train/test, rasio tidak valid ditolak | test_split_reproducible_stratified_no_leak |
| Praproses | fitur panjang tetap dan deterministik; augmentasi hanya pada train | test_features_*, test_augmentation_only_in_train |
| Input tidak valid | gambar kosong/rusak/format tak didukung/terlalu besar byte; **resolusi di atas batas piksel** (celah ditemukan dan ditutup); mode L/RGBA/P/CMYK; ukuran ekstrem 1x1, 3x500, 2000x2 | test_invalid_images_rejected, test_decompression_bomb_pixels_rejected, test_non_rgb_modes_..., test_extreme_sizes_... |
| Metrik | accuracy, precision/recall/F1 per kelas dan macro, confusion matrix cocok hitungan tangan | test_metrics_match_hand_computation |
| Eksperimen | alur end-to-end (latih, evaluasi di test set, simpan config+versi+hash test, checkpoint, inferensi); reproducible; menolak dataset tidak layak; `predictions.csv` tersimpan | test_end_to_end_..., test_experiment_is_reproducible, test_experiment_refuses_bad_dataset |
| Laporan | angka terukur dan keterbatasan ada di Markdown | test_report_contains_measured_numbers_and_limitations |
| CNN | latih/evaluasi/simpan/muat/prediksi; reproducible dengan seed; Grad-CAM (bentuk, rentang, kelas target, overlay) | test_cnn_*, test_gradcam_* |
| Checkpoint | daftar mencakup `.joblib` **dan** `.pt` (bug UI ditemukan lewat browser) | test_list_checkpoints_includes_cnn_and_classic |
| UI | aplikasi tampil dengan disclaimer | test_app_smoke.py |

## Verifikasi di browser (Playwright/Chromium, dataset foto nyata)
Audit dataset (900 gambar, 0 rusak/duplikat) -> tab Perbandingan: 4 eksperimen, contoh prediksi benar dan salah dengan gambar -> tab Prediksi: checkpoint CNN dipilih, gambar daun hawar awal diunggah -> kelas `hawar_awal`, skor 76%, tabel top-3, Grad-CAM tampil. Tidak ada galat JavaScript. Screenshot: `docs/screenshots/`.

## Temuan dari pengujian ini yang diperbaiki
Peringatan dtype/`probability` SVC; batas piksel gambar tidak ditegakkan (Pillow hanya memperingatkan di bawah 2x batas); checkpoint CNN tidak pernah muncul di UI.
Belum diuji: foto lapangan (non-laboratorium), kalibrasi probabilitas, uji beban.
