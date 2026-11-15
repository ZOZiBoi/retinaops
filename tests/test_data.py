import numpy as np
import pandas as pd
import torch
from PIL import Image

from retinaops.data.dataset import FundusDataset, build_transforms
from retinaops.data.preprocess import crop_black_border, split
from retinaops.data.validate import validate

CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def _make_raw(tmp_path, n_per_class=4, corrupt=False):
    (tmp_path / "images").mkdir()
    rows = []
    for c in range(5):
        for i in range(n_per_class):
            id_code = f"img{c}_{i}"
            Image.new("RGB", (200, 200), (c * 40, 80, 80)).save(tmp_path / "images" / f"{id_code}.png")
            rows.append({"id_code": id_code, "diagnosis": c})
    if corrupt:
        (tmp_path / "images" / "img0_0.png").write_bytes(b"not a png")
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    return tmp_path


def test_validate_passes_on_clean_data(tmp_path):
    report = validate(_make_raw(tmp_path), CLASSES, min_side=128, max_ratio=15)
    assert report["passed"]
    assert report["num_images"] == 20
    assert report["imbalance_ratio"] == 1.0


def test_validate_flags_corrupt_image(tmp_path):
    report = validate(_make_raw(tmp_path, corrupt=True), CLASSES, min_side=128, max_ratio=15)
    assert not report["passed"]
    assert report["errors"]["bad_images"][0]["id"] == "img0_0"


def test_validate_flags_small_images(tmp_path):
    report = validate(_make_raw(tmp_path), CLASSES, min_side=256, max_ratio=15)
    assert not report["passed"]
    assert len(report["errors"]["bad_images"]) == 20


def test_crop_black_border():
    img = Image.new("RGB", (100, 100))
    img.paste((200, 100, 50), (20, 30, 80, 70))
    assert crop_black_border(img).size == (60, 40)


def test_split_is_stratified_and_disjoint():
    df = pd.DataFrame({"id_code": [f"x{i}" for i in range(200)], "diagnosis": np.repeat(range(5), 40)})
    train, val, test = split(df, 0.15, 0.15, seed=0)
    assert len(train) + len(val) + len(test) == 200
    assert not set(train.id_code) & set(test.id_code)
    assert not set(val.id_code) & set(test.id_code)
    assert test.diagnosis.value_counts().nunique() == 1  # equal per class


def test_dataset_and_transforms(tmp_path):
    raw = _make_raw(tmp_path)
    ds = FundusDataset(raw / "train.csv", raw / "images", build_transforms(224, train=True))
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert y == 0
    assert ds.class_counts(5).tolist() == [4] * 5
