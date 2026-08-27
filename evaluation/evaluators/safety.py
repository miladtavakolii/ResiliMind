from __future__ import annotations

from typing import Any

from evaluation.evaluators.base import BaseEvaluator


class SafetyEvaluator(BaseEvaluator):
    """Evaluates the Safety Gate performance.

    Safety evaluation is treated as a binary classification problem:
        - Positive class: High-risk case
        - Negative class: Safe case

    The evaluator focuses on recall because missing a dangerous
    situation (false negative) is more critical than false alarms.

    Attributes:
        name (str): Identifier name for the evaluator.
    """

    name: str = "safety"

    def evaluate(self, gold: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """Compare predicted safety status with ground truth.

        Args:
            gold: EvaluationGold object containing ground truth safety labels.
            prediction: Workflow prediction dictionary containing safety outputs.

        Returns:
            dict[str, Any]: Dictionary containing confusion matrix elements
                (tp, tn, fp, fn) and classification metrics (accuracy, precision, recall, f1).
        """
        gold_high_risk = bool(gold.safety.is_high_risk)
        predicted_high_risk = bool(
            prediction.get("safety", {}).get("is_high_risk", False)
        )

        gold_category = gold.safety.risk_category
        predicted_category = (
            prediction.get("safety", {}).get("risk_category", "SAFE")
        )

        tp = int(gold_high_risk and predicted_high_risk)
        tn = int(not gold_high_risk and not predicted_high_risk)
        fp = int(not gold_high_risk and predicted_high_risk)
        fn = int(gold_high_risk and not predicted_high_risk)

        category_correct = gold_category == predicted_category

        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * precision * recall, precision + recall)
        accuracy = self._safe_div(tp + tn, tp + tn + fp + fn)

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
            "category": {
                "expected": gold_category,
                "predicted": predicted_category,
                "correct": category_correct,
            }
        }

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        """Perform division safely, returning 0.0 when the denominator is zero.

        Args:
            numerator: The dividend.
            denominator: The divisor.

        Returns:
            float: The division result, or 0.0 if denominator is 0.
        """
        if denominator == 0:
            return 0.0
        return numerator / denominator
