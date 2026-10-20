# Eksperimen

Semua angka di bawah berasal dari eksekusi nyata `run_experiment(...)` dan tersimpan di `artifacts/experiments/` (tidak di-commit; dapat dibuat ulang).

## Eksperimen A - foto nyata: PlantVillage tomat (900 gambar)
Konfigurasi: seed 42, test 20% = **180 gambar** (60/kelas), augmentasi flip/rotasi hanya pada train, CNN dilatih dari nol 96x96 (maks. 80 epoch, early stopping pada validasi dari data latih; epoch terbaik 25 dari 30 dijalankan). Versi library tercatat di `config.json` tiap eksperimen.

| Model | Accuracy | Precision macro | Recall macro | F1 macro | Inferensi ms/gambar | Ukuran KB |
|---|---|---|---|---|---|---|
| Random Forest (fitur warna/tekstur) | 0.933 | 0.934 | 0.933 | 0.933 | 0.071 | 3349.8 |
| SVM RBF (fitur warna/tekstur) | 0.939 | 0.940 | 0.939 | 0.939 | 0.029 | 149.2 |
| MLP (fitur warna/tekstur) | 0.939 | 0.940 | 0.939 | 0.939 | 0.002 | 90.9 |
| CNN kecil dari nol | 0.944 | 0.944 | 0.944 | 0.944 | 0.949 | 252.2 |

Waktu inferensi tidak termasuk ekstraksi fitur (klasik) dan bergantung pada mesin (CPU 4 inti).

**Per kelas (recall):** `sehat` 0,97-1,00 pada semua model; `hawar_akhir` 0,88-0,92 dan `hawar_awal` 0,92-0,95. Confusion matrix menunjukkan kesalahan hampir semuanya **antara dua kelas hawar** (baris/kolom: hawar_akhir, hawar_awal, sehat), mis. CNN `[[55,4,1],[5,55,0],[0,0,60]]`, SVM `[[53,7,0],[3,57,0],[1,0,59]]`. Model hampir tidak pernah salah menyebut daun sakit sebagai sehat, tetapi sering menukar jenis hawar.

**Yang boleh disimpulkan:** keempat model menghasilkan hasil setara. Selisih 0,933-0,944 hanyalah 2 gambar dari 180 (1 gambar = 0,56 poin); itu berada dalam variasi acak, jadi **tidak ada model yang terbukti lebih baik**. CNN tidak unggul jelas atas SVM berfitur warna sederhana pada dataset ini, kemungkinan karena kelasnya dibedakan kuat oleh warna dan latarnya seragam.
**Yang tidak boleh disimpulkan:** akurasi pada foto lapangan, atau bahwa model mendiagnosis penyakit. Skor kemungkinan optimistis karena kemungkinan kebocoran antar-foto daun yang sama dan latar laboratorium (lihat `DATASET_CARD.md`).

## Eksperimen B - dataset sintetis (180 gambar, hanya uji alur)
RF 1.000, SVM 0.972, MLP 1.000 (36 uji; seed 42). Tidak bermakna selain membuktikan alur berjalan.

## Belum dilakukan
- Transfer learning (bobot pralatih ImageNet tidak dapat diunduh pada lingkungan pengembangan ini).
- Split berbasis pengelompokan daun, uji pada foto lapangan, validasi silang, kalibrasi probabilitas.
