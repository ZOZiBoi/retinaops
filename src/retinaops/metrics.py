"""Evaluation metrics suited to imbalanced, ordinal DR grading.

Plain accuracy is misleading on APTOS (~49% of images are grade 0), so we also
report quadratic weighted kappa (the APTOS competition metric), macro-F1,
per-class sensitivity/specificity, macro AUC-ROC and referable-DR (grade >= 2)
sensitivity/specificity, the clinically relevant screening decision.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _sens_spec(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> tuple[float, float]:
    tp = np.sum(y_true_bin & y_pred_bin)
    fn = np.sum(y_true_bin & ~y_pred_bin)
    tn = np.sum(~y_true_bin & ~y_pred_bin)
    fp = np.sum(~y_true_bin & y_pred_bin)
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return float(sens), float(spec)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, classes: list[str]
) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(classes)
    m: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=range(k))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(k), zero_division=0)),
    }

    sens_all, spec_all = [], []
    for c, name in enumerate(classes):
        s, p = _sens_spec(y_true == c, y_pred == c)
        m[f"sensitivity_{name}"], m[f"specificity_{name}"] = s, p
        sens_all.append(s)
        spec_all.append(p)
    m["macro_sensitivity"] = float(np.mean(sens_all))
    m["macro_specificity"] = float(np.mean(spec_all))

    # Referable DR = moderate or worse (grade >= 2): the screening referral decision.
    s, p = _sens_spec(y_true >= 2, y_pred >= 2)
    m["referable_sensitivity"], m["referable_specificity"] = s, p

    if y_prob is not None:
        try:
            m["auc_macro_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=range(k))
            )
        except ValueError:  # a class missing from y_true (tiny batches/tests)
            m["auc_macro_ovr"] = float("nan")
    return m


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=range(num_classes))
