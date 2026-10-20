# Dataset Card

## 1. Dataset utama: subset PlantVillage - daun tomat (FOTO NYATA)
- **Sumber:** PlantVillage Dataset, https://github.com/spMohanty/PlantVillage-Dataset (`raw/color/`). Makalah: Mohanty, Hughes, Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science 7:1419.
- **Pengambilan:** `python scripts/fetch_plantvillage.py` (klon sparse + sampel acak, seed 42). Gambar **tidak di-commit** ke repo.
- **Kelas dan jumlah (subset):** `sehat` (Tomato___healthy) 300, `hawar_awal` (Early blight) 300, `hawar_akhir` (Late blight) 300 = 900 gambar, JPG 256x256, seimbang (rasio 1,0). Sumber penuh memiliki 1.591 / 1.000 / 1.909 gambar untuk ketiga kelas ini.
- **Lisensi:** README repo menyebut dataset "open access" tetapi **tidak mencantumkan lisensi eksplisit**. Versi Kaggle/TFDS dilaporkan CC0, tetapi itu belum diverifikasi dari sumber ini. Periksa sebelum redistribusi; karena itu gambar tidak disertakan.
- **Audit otomatis (hasil aktual):** 0 gambar rusak, 0 duplikat isi identik, 0 label konflik.

### Potensi bias dan kebocoran (belum terselesaikan)
- Foto diambil di laboratorium dengan latar seragam; model dapat mempelajari artefak latar/pencahayaan, bukan gejala penyakit. Performa pada foto lapangan **belum diuji**.
- Satu daun dapat difoto berulang. Nama berkas hanya berisi UUID unik per gambar sehingga **pengelompokan per daun tidak bisa direkonstruksi**. Gambar daun yang sama dapat jatuh di train dan test, sehingga skor kemungkinan optimistis. Versi Hugging Face dataset ini menyediakan split yang menghormati pengelompokan daun dan merupakan perbaikan yang disarankan.
- Hanya 3 kelas dari satu tanaman; bukan keragaman penyakit sebenarnya.

## 2. Dataset demo sintetis: `data/synthetic_leaves`
Gambar prosedural (`scripts/make_synthetic_dataset.py`, seed 42), 3 kelas x 60, 96x96. Hanya untuk menguji alur dan tes; skor tinggi tidak mewakili apa pun pada daun nyata.

## 3. Aturan metodologi yang diterapkan di kode
- Struktur `data/<kelas>/<gambar>`; subfolder `train/test` sumber diabaikan, split dibuat ulang (stratified, seed tercatat).
- Test set ditetapkan sebelum model dilatih; tidak ada tuning pada test. CNN memakai validation split dari data TRAIN untuk memilih epoch.
- Duplikat identik dibuang sebelum split dan dicek tidak muncul lintas train/test; augmentasi (flip/rotasi) hanya pada train.
- Fitur/praproses sama untuk semua model klasik dan inferensi; CNN memakai jalur normalisasi yang sama di training dan inferensi.
