from __future__ import annotations

from typing import Any

from evaluation.evaluators.base import BaseEvaluator


class SafetyEvaluator(BaseEvaluator):
    """
    Evaluates the Safety Gate performance.

    Safety evaluation is treated as a binary classification problem:

    Positive class:
        High-risk case

    Negative class:
        Safe case

    The evaluator focuses on recall because missing a dangerous
    situation (false negative) is more critical than false alarms.
    """

    name = "safety"

    def evaluate(
        self,
        gold: Any,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compare predicted safety status with ground truth.

        Args:
            gold:
                GoldSafety object from benchmark case.

            prediction:
                Workflow prediction output.

        Returns:
            Dictionary containing classification metrics.
        """

        gold_high_risk = bool(
            gold.safety.is_high_risk
        )

        predicted_high_risk = bool(
            prediction
            .get("safety", {})
            .get("is_high_risk", False)
        )

        tp = int(
            gold_high_risk
            and predicted_high_risk
        )

        tn = int(
            not gold_high_risk
            and not predicted_high_risk
        )

        fp = int(
            not gold_high_risk
            and predicted_high_risk
        )

        fn = int(
            gold_high_risk
            and not predicted_high_risk
        )

        precision = self._safe_div(
            tp,
            tp + fp,
        )

        recall = self._safe_div(
            tp,
            tp + fn,
        )

        f1 = self._safe_div(
            2 * precision * recall,
            precision + recall,
        )

        accuracy = self._safe_div(
            tp + tn,
            tp + tn + fp + fn,
        )

        return {
            "confusion_matrix": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def _safe_div(
        numerator: float,
        denominator: float,
    ) -> float:
        """
        Safe division helper.

        Returns zero when denominator is zero.
        """

        if denominator == 0:
            return 0.0

        return numerator / denominator
