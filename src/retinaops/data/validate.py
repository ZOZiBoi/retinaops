"""Data validation: corrupt files, image sizes, label sanity and class balance.

Writes reports/data_validation.json and exits non-zero if a check fails, which
stops `dvc repro` before bad data reaches training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from retinaops.common import ROOT, load_params, write_json


def check_image(path: Path, min_side: int) -> str | None:
    """Return an error string, or None if the image is fine."""
    try:
        with Image.open(path) as im:
            im.verify()  # detects truncated / corrupt files
        with Image.open(path) as im:
            w, h = im.size
            if min(w, h) < min_side:
                return f"too small ({w}x{h})"
    except Exception as e:  # noqa: BLE001
        return f"corrupt: {e.__class__.__name__}"
    return None


def validate(raw: Path, classes: list[str], min_side: int, max_ratio: float) -> dict:
    df = pd.read_csv(raw / "train.csv")
    errors: dict[str, list] = {"missing": [], "bad_images": [], "bad_labels": [], "duplicates": []}

    errors["duplicates"] = df[df.id_code.duplicated()].id_code.tolist()
    errors["bad_labels"] = df[~df.diagnosis.isin(range(len(classes)))].id_code.tolist()

    for id_code in df.id_code:
        p = raw / "images" / f"{id_code}.png"
        if not p.exists():
            errors["missing"].append(id_code)
        elif (err := check_image(p, min_side)) is not None:
            errors["bad_images"].append({"id": id_code, "error": err})

    counts = df.diagnosis.value_counts().reindex(range(len(classes)), fill_value=0)
    ratio = float(counts.max() / max(counts.min(), 1))
    distribution = {
        classes[i]: {"count": int(c), "pct": round(100 * c / len(df), 2)} for i, c in counts.items()
    }

    passed = not any(errors.values()) and ratio <= max_ratio
    return {
        "passed": passed,
        "num_images": int(len(df)),
        "class_distribution": distribution,
        "imbalance_ratio": round(ratio, 2),
        "max_imbalance_ratio": max_ratio,
        "errors": errors,
    }


def main() -> None:
    params = load_params()
    report = validate(
        ROOT / params["data"]["raw_dir"],
        params["data"]["classes"],
        params["validate"]["min_side"],
        params["validate"]["max_imbalance_ratio"],
    )
    write_json(report, ROOT / "reports" / "data_validation.json")

    print(f"Images: {report['num_images']}  imbalance ratio: {report['imbalance_ratio']}")
    for cls, d in report["class_distribution"].items():
        print(f"  {cls:15s} {d['count']:5d}  ({d['pct']}%)")
    if not report["passed"]:
        print("Data validation FAILED, see reports/data_validation.json")
        sys.exit(1)
    print("Data validation passed")


if __name__ == "__main__":
    main()
