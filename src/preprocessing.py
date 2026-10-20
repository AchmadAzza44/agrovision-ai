"""Pra-pemrosesan gambar dan ekstraksi fitur. Satu jalur yang SAMA dipakai training dan inferensi."""

from __future__ import annotations

import io
from pathlib import Path
from typing import IO

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
FEATURE_SIZE = (64, 64)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageError(ValueError):
    """Gambar tidak valid; pesan aman ditampilkan ke pengguna."""


def open_image(source: str | Path | bytes | IO[bytes], name: str = "") -> Image.Image:
    """Buka dan validasi gambar -> RGB. Menolak format/ukuran tak didukung dan berkas rusak."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = name or path.name
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        name = name or getattr(source, "name", "")
        data = source.read()
    suffix = Path(name).suffix.lower()
    if name and suffix not in SUPPORTED_EXTENSIONS:
        raise ImageError(f"Format '{suffix}' tidak didukung. Gunakan JPG, PNG, atau BMP.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"Ukuran gambar {len(data) / 1_048_576:.1f} MB melebihi batas {MAX_IMAGE_BYTES // 1_048_576} MB.")
    if not data:
        raise ImageError("File gambar kosong.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:  # Pillow hanya memperingatkan di bawah 2x batas
            raise ImageError(f"Resolusi {image.width}x{image.height} melebihi batas {MAX_PIXELS:,} piksel.")
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ImageError(f"Gambar rusak atau tidak dapat dibaca: {exc}") from exc
    return image.convert("RGB")


def augment_pil(image: Image.Image) -> list[Image.Image]:
    """Augmentasi deterministik (flip, rotasi 90). HANYA boleh dipakai pada data training."""
    return [
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        image.transpose(Image.Transpose.ROTATE_90),
    ]


def extract_features(image: Image.Image) -> np.ndarray:
    """Fitur warna + tekstur sederhana (vektor 1-D, panjang tetap).

    RGB histogram 8 bin/kanal, histogram hue 12 bin, mean/std HSV, energi gradien,
    dan proporsi piksel gelap. Bukan CNN: kemampuan membedakan kelas terbatas pada warna/tekstur global.
    """
    small = image.convert("RGB").resize(FEATURE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0

    parts: list[np.ndarray] = []
    for ch in range(3):
        hist, _ = np.histogram(rgb[..., ch], bins=8, range=(0, 1))
        parts.append(hist / hist.sum())
    hue_hist, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 1), weights=hsv[..., 1])
    parts.append(hue_hist / max(hue_hist.sum(), 1e-9))
    parts.append(np.array([hsv[..., i].mean() for i in range(3)] + [hsv[..., i].std() for i in range(3)]))
    gray = np.asarray(small.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    parts.append(np.array([gray.mean(), gray.std()]))
    parts.append(np.array([(rgb.mean(axis=2) < 0.25).mean()]))
    return np.concatenate(parts).astype(np.float32)


FEATURE_DIM = 24 + 12 + 6 + 2 + 1
