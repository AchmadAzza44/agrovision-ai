"""Ambil SUBSET PlantVillage (foto daun tomat nyata) ke data/plantvillage_tomato/<kelas>/*.jpg.

Sumber: https://github.com/spMohanty/PlantVillage-Dataset (klon sparse tanpa riwayat).
Lisensi: README repo menyebut "open access" tetapi TIDAK mencantumkan lisensi eksplisit. Versi Kaggle/TFDS
dilaporkan CC0, namun itu belum diverifikasi dari sumber ini. Karena itu gambar TIDAK di-commit ke repo;
jalankan skrip ini sendiri dan periksa lisensi sebelum redistribusi.

Pengambilan sampel acak dengan seed tetap (reproducible).
"""

import argparse
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/spMohanty/plantvillage-dataset"
CLASSES = {  # nama folder sumber -> nama kelas lokal
    "Tomato___healthy": "sehat",
    "Tomato___Early_blight": "hawar_awal",
    "Tomato___Late_blight": "hawar_akhir",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/plantvillage_tomato")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", help="Folder klon PlantVillage yang sudah ada (lewati git clone)")
    a = ap.parse_args()

    workdir = None
    if a.source:
        root = Path(a.source)
    else:
        workdir = Path(tempfile.mkdtemp())
        root = workdir / "pv"
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", "--no-cone",
                        *[f"/raw/color/{c}/" for c in CLASSES]], check=True)

    rng = random.Random(a.seed)
    out = Path(a.out)
    for src_name, label in CLASSES.items():
        files = sorted((root / "raw" / "color" / src_name).glob("*.JPG"))
        if len(files) < a.per_class:
            raise SystemExit(f"{src_name}: hanya {len(files)} gambar")
        (out / label).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(rng.sample(files, a.per_class)):
            shutil.copy(f, out / label / f"{label}_{i:04d}.jpg")
        print(f"{label}: {a.per_class} gambar")
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
