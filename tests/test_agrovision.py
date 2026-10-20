import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.dataset import audit_dataset, clean_frame, scan_dataset, split_dataset
from src.evaluation import evaluate_predictions
from src.experiment_store import list_experiments, load_experiment
from src.inference import load_checkpoint, predict_image
from src.models import create_model
from src.preprocessing import FEATURE_DIM, ImageError, augment_pil, extract_features, open_image
from src.training import build_features, run_experiment


def png_bytes(color=(0, 200, 0), size=(40, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_scan_labels_match_folders(synthetic_dir):
    frame = scan_dataset(synthetic_dir)
    assert len(frame) == 72 and set(frame["label"]) == {"sehat", "bercak_coklat", "menguning"}
    assert all(label in path for path, label in zip(frame["path"], frame["label"]))


def test_audit_flags_corrupt_duplicates_and_small_classes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for i in range(12):
        Image.new("RGB", (20, 20), (i * 10, 0, 100)).save(tmp_path / "a" / f"{i}.png")
        Image.new("RGB", (20, 20), (0, i * 10, 200)).save(tmp_path / "b" / f"{i}.png")
    (tmp_path / "a" / "rusak.png").write_bytes(b"bukan gambar")
    (tmp_path / "b" / "dup.png").write_bytes((tmp_path / "b" / "3.png").read_bytes())
    audit = audit_dataset(scan_dataset(tmp_path))
    assert len(audit["corrupt"]) == 1 and audit["duplicates"] == 1 and not audit["errors"]
    assert len(clean_frame(scan_dataset(tmp_path))) == 24


def test_audit_flags_conflicting_labels(tmp_path):
    for cls in "ab":
        (tmp_path / cls).mkdir()
        for i in range(11):
            Image.new("RGB", (9, 9), (i * 9, 5, ord(cls))).save(tmp_path / cls / f"{i}.png")
    Image.new("RGB", (9, 9), (1, 2, 3)).save(tmp_path / "a" / "same.png")
    Image.new("RGB", (9, 9), (1, 2, 3)).save(tmp_path / "b" / "same.png")
    assert any("label berbeda" in e for e in audit_dataset(scan_dataset(tmp_path))["errors"])


def test_audit_errors_on_empty_and_single_class(tmp_path):
    assert audit_dataset(scan_dataset(tmp_path))["errors"]
    (tmp_path / "x").mkdir()
    for i in range(12):
        Image.new("RGB", (8, 8), (i, i, i)).save(tmp_path / "x" / f"{i}.png")
    assert any("2 kelas" in e for e in audit_dataset(scan_dataset(tmp_path))["errors"])


def test_split_reproducible_stratified_no_leak(synthetic_dir):
    frame = clean_frame(scan_dataset(synthetic_dir))
    a_train, a_test = split_dataset(frame, 0.25, 7)
    b_train, b_test = split_dataset(frame, 0.25, 7)
    assert a_test["path"].tolist() == b_test["path"].tolist()
    assert not set(a_train["path"]) & set(a_test["path"])
    assert a_test["label"].value_counts().nunique() == 1  # stratified, kelas seimbang
    with pytest.raises(ValueError):
        split_dataset(frame, 0.9)


def test_features_fixed_length_and_deterministic():
    img = open_image(png_bytes())
    f1, f2 = extract_features(img), extract_features(img)
    assert f1.shape == (FEATURE_DIM,) and np.array_equal(f1, f2) and np.isfinite(f1).all()


def test_augmentation_only_in_train(synthetic_dir):
    frame = clean_frame(scan_dataset(synthetic_dir)).head(6)
    plain, _ = build_features(frame, augment=False)
    aug, y = build_features(frame, augment=True)
    assert len(plain) == 6 and len(aug) == 6 * (1 + len(augment_pil(open_image(frame['path'][0]))))
    assert len(y) == len(aug)


@pytest.mark.parametrize("data,name", [(b"", "a.png"), (b"xx", "a.png"), (png_bytes(), "a.gif"), (b"x" * (11 * 1024 * 1024), "a.png")])
def test_invalid_images_rejected(data, name):
    with pytest.raises(ImageError):
        open_image(data, name)


def test_metrics_match_hand_computation():
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "b", "a"]
    m = evaluate_predictions(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["a"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["a"]["recall"] == pytest.approx(2 / 3)
    assert m["per_class"]["c"]["recall"] == 0.0
    assert m["recall_macro"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert m["confusion_matrix"] == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_cnn_not_available_as_feature_model():
    with pytest.raises(NotImplementedError):
        create_model("cnn")
    with pytest.raises(ValueError):
        create_model("xyz")


def test_end_to_end_experiment_and_inference(synthetic_dir, tmp_path):
    out = run_experiment(synthetic_dir, ["random_forest", "svm"], seed=3, test_size=0.25, out_dir=tmp_path)
    for name, res in out["results"].items():
        m = res["metrics"]
        assert 0 <= m["accuracy"] <= 1 and m["n_test"] == 18 and m["inference_ms_per_image"] > 0
    names = list_experiments(tmp_path / "experiments")
    assert len(names) == 2
    saved = load_experiment(tmp_path / "experiments", names[0])
    assert saved["config"]["seed"] == 3 and saved["config"]["augment_train_only"] is True
    assert len(saved["config"]["test_sha256"]) == 18 and "scikit-learn" in saved["config"]["versions"]

    bundle = load_checkpoint(out["results"]["random_forest"]["checkpoint"])
    sample = next(synthetic_dir.glob("sehat/*.png"))
    pred = predict_image(bundle, sample.read_bytes(), sample.name)
    assert pred["predicted_class"] in out["classes"] and 0 <= pred["confidence"] <= 1
    assert "bukan diagnosis" in pred["disclaimer"]
    with pytest.raises(ImageError):
        predict_image(bundle, b"bukan gambar", "x.png")


def test_experiment_is_reproducible(synthetic_dir, tmp_path):
    a = run_experiment(synthetic_dir, ["random_forest"], seed=5, test_size=0.25, out_dir=tmp_path / "a")
    b = run_experiment(synthetic_dir, ["random_forest"], seed=5, test_size=0.25, out_dir=tmp_path / "b")
    ma, mb = a["results"]["random_forest"]["metrics"], b["results"]["random_forest"]["metrics"]
    assert ma["confusion_matrix"] == mb["confusion_matrix"] and ma["f1_macro"] == mb["f1_macro"]


def test_experiment_refuses_bad_dataset(tmp_path):
    with pytest.raises(ValueError):
        run_experiment(tmp_path, ["svm"], out_dir=tmp_path / "o")
    with pytest.raises(FileNotFoundError):
        run_experiment(tmp_path / "nope", ["svm"], out_dir=tmp_path / "o")


# ---------- robustness input (PRD P2) ----------
@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_non_rgb_modes_are_converted_and_give_fixed_features(mode):
    buf = io.BytesIO()
    img = Image.new(mode, (30, 20))
    img.save(buf, "PNG" if mode != "CMYK" else "TIFF") if mode == "CMYK" else img.save(buf, "PNG")
    if mode == "CMYK":
        buf = io.BytesIO()
        Image.new("CMYK", (30, 20)).convert("RGB").save(buf, "PNG")
    assert extract_features(open_image(buf.getvalue(), "x.png")).shape == (FEATURE_DIM,)


@pytest.mark.parametrize("size", [(1, 1), (3, 500), (2000, 2)])
def test_extreme_sizes_are_handled(size):
    assert np.isfinite(extract_features(open_image(png_bytes(size=size), "x.png"))).all()


def test_decompression_bomb_pixels_rejected():
    big = Image.new("L", (6000, 6000))
    buf = io.BytesIO()
    big.save(buf, "PNG", optimize=False, compress_level=9)
    assert buf.tell() < 10 * 1024 * 1024  # lolos batas byte, harus ditolak oleh batas piksel
    with pytest.raises(ImageError):
        open_image(buf.getvalue(), "bomb.png")


def test_report_contains_measured_numbers_and_limitations(synthetic_dir, tmp_path):
    from src.reporting import build_report
    run_experiment(synthetic_dir, ["random_forest"], seed=3, test_size=0.25, out_dir=tmp_path)
    exp = load_experiment(tmp_path / "experiments", list_experiments(tmp_path / "experiments")[0])
    md = build_report({"rf": exp})
    assert f"{exp['metrics']['accuracy']:.3f}" in md and "Keterbatasan" in md and "Confusion matrix" in md
    with pytest.raises(ValueError):
        build_report({})
    assert (tmp_path / "experiments" / list_experiments(tmp_path / "experiments")[0] / "predictions.csv").exists()


# ---------- CNN & Grad-CAM (dilewati bila torch tidak ada) ----------
torch = pytest.importorskip("torch")


def test_cnn_trains_evaluates_saves_and_predicts(synthetic_dir, tmp_path):
    from src.inference import load_checkpoint
    out = run_experiment(synthetic_dir, ["cnn"], seed=3, test_size=0.25, out_dir=tmp_path, cnn_epochs=8)
    m = out["results"]["cnn"]["metrics"]
    assert m["n_test"] == 18 and 0 <= m["accuracy"] <= 1 and m["epochs_run"] <= 8 and m["best_epoch"] >= 1
    assert m["accuracy"] > 0.5  # kelas sintetis mudah; acak = 0.33
    bundle = load_checkpoint(out["results"]["cnn"]["checkpoint"])
    sample = next(synthetic_dir.glob("sehat/*.png"))
    pred = predict_image(bundle, sample.read_bytes(), sample.name)
    assert pred["predicted_class"] in out["classes"] and abs(sum(t["skor"] for t in pred["top"]) - 1) < 1.01
    saved = load_experiment(tmp_path / "experiments", list_experiments(tmp_path / "experiments")[0])
    assert "torch" in saved["config"]["versions"]


def test_cnn_is_reproducible_with_same_seed(synthetic_dir, tmp_path):
    a = run_experiment(synthetic_dir, ["cnn"], seed=5, test_size=0.25, out_dir=tmp_path / "a", cnn_epochs=4)
    b = run_experiment(synthetic_dir, ["cnn"], seed=5, test_size=0.25, out_dir=tmp_path / "b", cnn_epochs=4)
    assert a["results"]["cnn"]["metrics"]["confusion_matrix"] == b["results"]["cnn"]["metrics"]["confusion_matrix"]


def test_gradcam_shape_range_and_target(synthetic_dir):
    from src.cnn import CNNClassifier, grad_cam, image_to_array, overlay_cam, load_arrays
    frame = clean_frame(scan_dataset(synthetic_dir))
    X, y = load_arrays(frame)
    model = CNNClassifier(sorted(set(y)), seed=1, epochs=2).fit(X, y)
    img = open_image(frame["path"][0])
    cam, cls = grad_cam(model, img)
    assert cam.shape == (96, 96) and 0 <= cam.min() and cam.max() <= 1 and 0 <= cls < 3
    assert overlay_cam(img, cam).size == (192, 192)
    cam2, cls2 = grad_cam(model, img, target=(cls + 1) % 3)
    assert cls2 == (cls + 1) % 3


def test_list_checkpoints_includes_cnn_and_classic(tmp_path):
    from src.inference import list_checkpoints
    for name in ("svm_20260101T000000Z.joblib", "cnn_20260102T000000Z.pt", "mlp_20260101T000000Z.joblib", "catatan.txt"):
        (tmp_path / name).write_bytes(b"x")
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names[0] == "cnn_20260102T000000Z.pt" and len(names) == 3 and "catatan.txt" not in names
    assert list_checkpoints(tmp_path / "tidak_ada") == []
