import numpy as np
import torch

from retinaops.metrics import compute_metrics
from retinaops.model import build_model, load_checkpoint, save_checkpoint

CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 0, 2])
    prob = np.eye(5)[y]
    m = compute_metrics(y, y, prob, CLASSES)
    assert m["accuracy"] == 1.0 and m["qwk"] == 1.0 and m["macro_f1"] == 1.0
    assert m["referable_sensitivity"] == 1.0 and m["referable_specificity"] == 1.0
    assert m["auc_macro_ovr"] == 1.0


def test_qwk_penalises_distant_errors_more():
    y = np.array([0, 0, 2, 2, 4, 4])
    near = compute_metrics(y, np.array([0, 1, 2, 3, 4, 3]), None, CLASSES)["qwk"]
    far = compute_metrics(y, np.array([0, 4, 2, 0, 4, 0]), None, CLASSES)["qwk"]
    assert near > far


def test_always_no_dr_has_zero_qwk():
    y = np.array([0, 0, 0, 1, 2, 3, 4])
    m = compute_metrics(y, np.zeros_like(y), None, CLASSES)
    assert m["qwk"] == 0.0
    assert m["referable_sensitivity"] == 0.0


def test_model_forward_and_checkpoint_roundtrip(tmp_path):
    model = build_model("efficientnet_b0", 5, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5)

    path = tmp_path / "model.pt"
    save_checkpoint(model, path, arch="efficientnet_b0", classes=CLASSES, image_size=224)
    loaded, ckpt = load_checkpoint(path)
    assert ckpt["classes"] == CLASSES
    torch.testing.assert_close(loaded(x), out)
