"""Download APTOS 2019 from Kaggle into data/raw (then tracked with `dvc add data/raw`).

Layout produced:
    data/raw/train.csv                      id_code, diagnosis (0-4)
    data/raw/images/<id_code>.png
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from retinaops.common import ROOT, load_params


def main() -> None:
    params = load_params()
    classes = params["data"]["classes"]
    raw = ROOT / params["data"]["raw_dir"]

    src = Path(kagglehub.dataset_download(params["data"]["kaggle_dataset"]))
    print(f"Downloaded to kagglehub cache: {src}")

    df = pd.read_csv(next(src.rglob("train.csv")))
    img_root = next(p for p in src.rglob("colored_images") if p.is_dir())

    if raw.exists():
        shutil.rmtree(raw)
    (raw / "images").mkdir(parents=True)

    found = {}
    for cls in classes:
        for img in (img_root / cls).glob("*.png"):
            found[img.stem] = img

    rows = []
    for r in df.itertuples():
        img = found.get(r.id_code)
        if img is None:
            print(f"  missing image for {r.id_code}, skipping")
            continue
        shutil.copy2(img, raw / "images" / f"{r.id_code}.png")
        rows.append({"id_code": r.id_code, "diagnosis": int(r.diagnosis)})

    pd.DataFrame(rows).to_csv(raw / "train.csv", index=False)
    print(f"Wrote {len(rows)} images to {raw}")


if __name__ == "__main__":
    main()
