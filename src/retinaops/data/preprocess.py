"""Preprocess raw images and create a stratified train/val/test split.

- crops the black background around the fundus
- resizes to a fixed square size
- writes data/processed/images/*.png and data/processed/{train,val,test}.csv
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from retinaops.common import ROOT, load_params, write_json


def crop_black_border(img: Image.Image, threshold: int = 10) -> Image.Image:
    """Crop to the bounding box of pixels brighter than `threshold`."""
    gray = np.asarray(img.convert("L"))
    mask = gray > threshold
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def preprocess_image(src: Path, dst: Path, size: int, crop: bool) -> None:
    img = Image.open(src).convert("RGB")
    if crop:
        img = crop_black_border(img)
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    img.save(dst)


def split(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df.diagnosis, random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val.diagnosis,
        random_state=seed,
    )
    return train, val, test


def main() -> None:
    params = load_params()
    pp = params["preprocess"]
    raw = ROOT / params["data"]["raw_dir"]
    out = ROOT / params["data"]["processed_dir"]

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)

    df = pd.read_csv(raw / "train.csv")
    for id_code in df.id_code:
        preprocess_image(
            raw / "images" / f"{id_code}.png",
            out / "images" / f"{id_code}.png",
            pp["image_size"],
            pp["crop_black_border"],
        )

    train, val, test = split(df, pp["val_size"], pp["test_size"], params["seed"])
    summary = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)
        summary[name] = {
            "n": len(part),
            "per_class": part.diagnosis.value_counts().sort_index().tolist(),
        }
        print(f"{name:5s} n={len(part):5d} per-class={summary[name]['per_class']}")
    write_json(summary, ROOT / "reports" / "split_summary.json")


if __name__ == "__main__":
    main()
