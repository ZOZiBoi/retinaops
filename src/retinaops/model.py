"""Model factory and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def save_checkpoint(model: nn.Module, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
