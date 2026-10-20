# Laporan Singkat - AgroVision AI
**Materi kuliah:** Machine Learning (computer vision). PRD: `docs/PRD_Tiga_Implementasi_Informatika.md` (root repo), PRD 2.

## Masalah dan tujuan
Model klasifikasi sering ditampilkan tanpa evaluasi yang transparan. AgroVision membangun eksperimen yang adil (split sama, praproses sama, test set tidak disentuh saat tuning), menampilkan metrik per kelas, dan mencatat keterbatasan.

## Hubungan fitur dengan materi ML
| Konsep | Implementasi | Bukti |
|---|---|---|
| Dataset, label, bias | subset PlantVillage tomat (3 kelas x 300), audit otomatis, dataset card | `DATASET_CARD.md` |
| Pembagian data dan kebocoran | split stratified bereed; duplikat identik dibuang; hash train/test dicek; augmentasi hanya train | `test_split_reproducible_stratified_no_leak` |
| Baseline dan perbandingan model | RF, SVM, MLP (fitur warna/tekstur) vs CNN dari nol; prosedur sama | `EXPERIMENTS.md` |
| Evaluasi | accuracy, precision/recall/F1 macro dan per kelas, confusion matrix, waktu inferensi, ukuran model | `test_metrics_match_hand_computation` |
| Overfitting/seleksi model | CNN: early stopping pada validation split dari data TRAIN | `src/cnn.py` |
| Interpretabilitas | Grad-CAM untuk CNN (penjelasan, bukan bukti kausal) | screenshot 03 |
| Reproduksibilitas | seed, versi library, hash test set, config tersimpan | `config.json` tiap eksperimen |

## Hasil nyata (180 gambar uji, seed 42)
| Model | Accuracy | F1 macro |
|---|---|---|
| Random Forest | 0,933 | 0,933 |
| SVM RBF | 0,939 | 0,939 |
| MLP | 0,939 | 0,939 |
| CNN kecil | 0,944 | 0,944 |
Kesalahan hampir semuanya antara kedua kelas hawar; daun sehat hampir selalu benar (recall 0,97-1,00). **Selisih antar model hanya 1-2 gambar (<= 1,1 poin) sehingga tidak ada model yang terbukti lebih baik.**

## Keterbatasan
Skor kemungkinan optimistis: satu daun dapat difoto berkali-kali (pengelompokan per daun tidak tersedia), latar laboratorium seragam, hanya 3 kelas dari satu tanaman. Performa pada foto lapangan belum diuji. Transfer learning belum (bobot pralatih tidak dapat diunduh). Skor model bukan probabilitas terkalibrasi. Lisensi eksplisit dataset tidak tercantum di sumber (gambar tidak di-commit).

## Demonstrasi
`python scripts/fetch_plantvillage.py` -> `python -m streamlit run app.py` -> Eksperimen (Audit, lalu Latih dan evaluasi) -> Perbandingan -> Prediksi (pilih checkpoint CNN, unggah daun) -> unduh laporan.
