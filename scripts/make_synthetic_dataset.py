"""Buat dataset daun SINTETIS (gambar prosedural) untuk demo & tes alur. BUKAN foto daun nyata.

Hasil pada data ini tidak boleh dilaporkan sebagai performa pada daun sesungguhnya.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ("sehat", "bercak_coklat", "menguning")


def make_leaf(label: str, rng: np.random.Generator, size: int = 96) -> Image.Image:
    bg = tuple(int(v) for v in rng.integers(200, 240, 3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    base = {"sehat": (40, 140, 50), "bercak_coklat": (50, 130, 55), "menguning": (150, 160, 50)}[label]
    col = tuple(int(np.clip(c + rng.integers(-18, 18), 0, 255)) for c in base)
    m = int(rng.integers(8, 16))
    box = (m, m + int(rng.integers(0, 8)), size - m, size - m)
    d.ellipse(box, fill=col)
    d.line([(size // 2, box[1]), (size // 2, box[3])], fill=tuple(min(255, c + 40) for c in col), width=2)
    if label == "bercak_coklat":
        for _ in range(int(rng.integers(5, 12))):
            x, y, r = rng.integers(20, size - 20), rng.integers(20, size - 20), int(rng.integers(2, 6))
            d.ellipse((x - r, y - r, x + r, y + r), fill=(110 + int(rng.integers(0, 30)), 65, 30))
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32)
    arr += rng.normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_leaves")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for label in CLASSES:
        folder = Path(a.out) / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_class):
            make_leaf(label, rng).save(folder / f"{label}_{i:03d}.png")
    print(f"Dataset sintetis: {a.per_class * len(CLASSES)} gambar di {a.out}")


if __name__ == "__main__":
    main()
