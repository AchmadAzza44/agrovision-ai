# Progress

## M0 — Foundation
- [x] Repository dan struktur folder
- [x] Environment dan dependencies
- [x] README

## M1 — Dataset audit
- [x] Kelas, distribusi, duplikasi, label konflik (otomatis)
- [x] Validasi file gambar
- [x] Dataset nyata (subset PlantVillage tomat); lisensi eksplisit tidak tercantum di sumber (dicatat)

## M2 — Baseline
- [x] Split data dan model awal
- [x] Training dapat dijalankan ulang dengan konfigurasi tercatat

## M3 — Evaluation
- [x] Metrik, confusion matrix, laporan per kelas
- [x] Test set tidak dipakai untuk melatih model (dites)

## M4 — Prediction UI
- [x] Unggah gambar dan inferensi

## M5 — Model comparison
- [x] Empat model (RF, SVM, MLP, CNN) pada prosedur sama

## M6 — Explainability
- [x] Grad-CAM (CNN), contoh benar/salah, analisis per kelas
- [ ] Analisis kebocoran per daun (metadata tidak tersedia)

## M7 — Release
- [x] Dokumentasi, tes (30 lulus), eksperimen pada data nyata, ekspor laporan
- [x] Screenshot (`docs/screenshots/`), laporan (`LAPORAN.md`), rencana uji (`TEST_PLAN.md`)
