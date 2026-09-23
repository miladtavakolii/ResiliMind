from __future__ import annotations

from typing import Any


def classification_report(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Calculate multiclass classification metrics.

    Args:
        pairs: List of tuples containing (gold_label, predicted_label).

    Returns:
        dict[str, Any]: Dictionary containing accuracy, macro-averaged metrics,
            and per-class precision, recall, and F1 scores.
    """
    if not pairs:
        return {
            "accuracy": 0.0,
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "per_class": {},
        }

    labels = sorted({label for pair in pairs for label in pair})
    total = len(pairs)
    correct = sum(1 for gold, pred in pairs if gold == pred)
    per_class = {}
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": correct / total if total else 0.0,
        "macro": {
            metric: sum(value[metric] for value in per_class.values()) / len(per_class)
            for metric in ("precision", "recall", "f1")
        },
        "per_class": per_class,
    }
