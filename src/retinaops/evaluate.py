"""Evaluate the best checkpoint on the held-out test set and log to the training run.

Usage: python -m retinaops.evaluate --model efficientnet_b0
Outputs: reports/<arch>/metrics.json, reports/<arch>/confusion_matrix.png
Also logs the model to MLflow (as the `model` artifact) so it can be registered.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from retinaops.common import ROOT, get_device, load_params, setup_mlflow, write_json
from retinaops.data.dataset import FundusDataset, build_transforms
from retinaops.metrics import compute_metrics, confusion
from retinaops.model import load_checkpoint


@torch.no_grad()
def predict(model, loader, device):
    ys, probs = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).float().softmax(1).cpu())
        ys.append(y)
    prob = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), prob.argmax(1), prob


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    params = load_params()
    classes = params["data"]["classes"]
    device = get_device()
    processed = ROOT / params["data"]["processed_dir"]
    model_dir, report_dir = ROOT / "models" / args.model, ROOT / "reports" / args.model
    report_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(model_dir / "model.pt", device)
    test_ds = FundusDataset(processed / "test.csv", processed / "images",
                            build_transforms(ckpt["image_size"], train=False))
    y, pred, prob = predict(model, DataLoader(test_ds, batch_size=64), device)

    m = compute_metrics(y, pred, prob, classes)
    cm = confusion(y, pred, len(classes))
    write_json({k: round(v, 4) for k, v in m.items()}, report_dir / "metrics.json")
    plot_confusion(cm, classes, f"{args.model}: test set (n={len(y)})", report_dir / "confusion_matrix.png")

    print(f"[{args.model}] test: " + "  ".join(
        f"{k}={m[k]:.4f}" for k in ["accuracy", "qwk", "macro_f1", "auc_macro_ovr",
                                    "referable_sensitivity", "referable_specificity"]))

    run_id = json.loads((model_dir / "run.json").read_text())["run_id"]
    mlflow = setup_mlflow(params)
    import mlflow.pytorch  # noqa: F401

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
        mlflow.log_artifact(str(report_dir / "confusion_matrix.png"))
        mlflow.log_artifact(str(report_dir / "metrics.json"))
        example = torch.zeros(1, 3, ckpt["image_size"], ckpt["image_size"])
        mlflow.pytorch.log_model(
            model.cpu(),
            name="model",
            input_example=example.numpy(),
            pip_requirements=["torch", "torchvision", "timm"],
        )
    print(f"Logged test metrics and model to MLflow run {run_id}")


if __name__ == "__main__":
    main()
