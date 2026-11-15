"""Fine-tune a pretrained timm model on APTOS and track the run in MLflow.

Usage: python -m retinaops.train --model efficientnet_b0
Outputs: models/<arch>/model.pt, models/<arch>/run.json, reports/<arch>/history.csv
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from retinaops.common import (
    ROOT,
    dvc_data_hash,
    get_device,
    git_commit,
    load_params,
    seed_everything,
    setup_mlflow,
    write_json,
)
from retinaops.data.dataset import FundusDataset, build_transforms
from retinaops.metrics import compute_metrics
from retinaops.model import build_model, save_checkpoint


def make_loaders(params: dict, batch_size: int):
    tp = params["train"]
    processed = ROOT / params["data"]["processed_dir"]
    size = params["preprocess"]["image_size"]
    k = len(params["data"]["classes"])

    train_ds = FundusDataset(processed / "train.csv", processed / "images", build_transforms(size, True))
    val_ds = FundusDataset(processed / "val.csv", processed / "images", build_transforms(size, False))

    counts = train_ds.class_counts(k)
    sampler = None
    if tp["weighted_sampler"]:
        w = (1.0 / counts)[train_ds.labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)

    kw = dict(
        batch_size=batch_size,
        num_workers=tp["num_workers"],
        persistent_workers=tp["num_workers"] > 0,
    )
    train_dl = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    return train_dl, val_dl, counts


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    autocast = torch.autocast(device.type, dtype=torch.float16) if amp else nullcontext()
    total_loss, ys, preds, probs = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast:
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(y)
            p = logits.float().softmax(1)
            ys.append(y.cpu())
            preds.append(p.argmax(1).cpu())
            probs.append(p.detach().cpu())

    y, yp, pr = torch.cat(ys).numpy(), torch.cat(preds).numpy(), torch.cat(probs).numpy()
    return total_loss / len(y), y, yp, pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    params = load_params()
    tp, mp = params["train"], params["models"][args.model]
    classes = params["data"]["classes"]
    seed_everything(params["seed"])
    device = get_device()
    amp = tp["amp"] and device.type in ("cuda", "mps")

    train_dl, val_dl, counts = make_loaders(params, mp["batch_size"])
    class_weights = None
    if tp["class_weights"]:
        class_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=tp["label_smoothing"],
    )

    model = build_model(args.model, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mp["lr"], weight_decay=tp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=mp["lr"], epochs=tp["epochs"], steps_per_epoch=len(train_dl), pct_start=0.2
    )
    scaler = torch.amp.GradScaler(device.type) if amp else None

    out_dir = ROOT / "models" / args.model
    report_dir = ROOT / "reports" / args.model
    report_dir.mkdir(parents=True, exist_ok=True)

    mlflow = setup_mlflow(params)
    with mlflow.start_run(run_name=f"{args.model}-{time.strftime('%Y%m%d-%H%M')}") as run:
        mlflow.set_tags(
            {
                "model_arch": args.model,
                "role": "baseline" if args.model == "resnet50" else "main",
                "git_commit": git_commit(),
                "data_version_md5": dvc_data_hash(),
                "dataset": params["data"]["kaggle_dataset"],
                "device": device.type,
            }
        )
        mlflow.log_params(
            {
                "arch": args.model,
                "seed": params["seed"],
                "image_size": params["preprocess"]["image_size"],
                **{f"model.{k}": v for k, v in mp.items()},
                **{f"train.{k}": v for k, v in tp.items()},
                "n_train": int(counts.sum()),
                "train_class_counts": counts.tolist(),
                "class_weights": [round(w, 3) for w in class_weights.tolist()] if class_weights is not None else None,
            }
        )

        best_qwk, best_epoch, bad_epochs, history = -1.0, -1, 0, []
        for epoch in range(1, tp["epochs"] + 1):
            t0 = time.time()
            tr_loss, ty, tpred, _ = run_epoch(model, train_dl, device, criterion, optimizer, scheduler, scaler, amp)
            va_loss, vy, vpred, vprob = run_epoch(model, val_dl, device, criterion, amp=amp)
            vm = compute_metrics(vy, vpred, vprob, classes)
            row = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_accuracy": float((ty == tpred).mean()),
                "val_loss": va_loss,
                "val_accuracy": vm["accuracy"],
                "val_qwk": vm["qwk"],
                "val_macro_f1": vm["macro_f1"],
                "val_auc_macro_ovr": vm["auc_macro_ovr"],
                "lr": scheduler.get_last_lr()[0],
                "epoch_seconds": time.time() - t0,
            }
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(
                f"[{args.model}] epoch {epoch:2d}/{tp['epochs']}  "
                f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={vm['accuracy']:.4f}  "
                f"val_qwk={vm['qwk']:.4f}  val_f1={vm['macro_f1']:.4f}  ({row['epoch_seconds']:.0f}s)",
                flush=True,
            )

            if vm["qwk"] > best_qwk:
                best_qwk, best_epoch, bad_epochs = vm["qwk"], epoch, 0
                save_checkpoint(
                    model, out_dir / "model.pt", arch=args.model, classes=classes,
                    image_size=params["preprocess"]["image_size"], epoch=epoch, val_qwk=best_qwk,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= tp["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                    break

        mlflow.log_metrics({"best_val_qwk": best_qwk, "best_epoch": best_epoch})
        pd.DataFrame(history).to_csv(report_dir / "history.csv", index=False)
        mlflow.log_artifact(str(report_dir / "history.csv"))
        write_json({"run_id": run.info.run_id, "best_epoch": best_epoch, "best_val_qwk": best_qwk},
                   out_dir / "run.json")
        print(f"Best val QWK {best_qwk:.4f} at epoch {best_epoch}. MLflow run {run.info.run_id}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
