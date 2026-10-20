# Dataset Card

## Persiapan Dataset

AgroVision AI membutuhkan dataset gambar daun dengan label kondisi (misalnya:
sehat, terkena penyakit A, terkena penyakit B, dst.).

### Struktur Direktori yang Direkomendasikan

```
data/
├── train/
│   ├── sehat/
│   ├── penyakit_a/
│   └── penyakit_b/
└── test/
    ├── sehat/
    ├── penyakit_a/
    └── penyakit_b/
```

### Dokumentasi yang Diperlukan

Setiap dataset harus mencatat:
- Asal dataset dan sumber
- Lisensi penggunaan
- Kelas dan jumlah sampel per kelas
- Resolusi gambar
- Potensi bias dalam pengumpulan

### Aturan Penting

- Cegah kebocoran data: gambar duplikat atau dari sumber/sesi yang sama
  sebaiknya tidak tersebar antara train dan test.
- Tetapkan test set sebelum tuning model. Jangan memakai test set untuk
  memilih hyperparameter.
- Terapkan preprocessing yang sama dan adil saat membandingkan model.
- Augmentasi hanya diterapkan pada data training.
- Laporkan metrik per kelas, bukan hanya akurasi keseluruhan.
