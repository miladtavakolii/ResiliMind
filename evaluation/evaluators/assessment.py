from __future__ import annotations

import math
from typing import Any

from evaluation.evaluators.base import BaseEvaluator


class AssessmentEvaluator(BaseEvaluator):
    """
    Evaluates the Assessor Agent score prediction quality.

    The Assessor predicts resilience dimensions:
    - severity
    - frequency
    - functional
    - coping

    This evaluator treats assessment as a regression problem and
    measures distance between predicted and ground-truth scores.
    """

    name = "assessment"

    DIMENSIONS = ("severity", "frequency", "functional", "coping")

    def evaluate(self, gold: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """
        Compare predicted assessment scores with ground truth.
        """
        gold_assessments = {
            item.node_id: item.rubric for item in gold.assessment.assessments
        }

        predicted_assessments = {
            item.get("node_id"): item.get("rubric", {})
            for item in prediction.get("assessment", {}).get("assessments", [])
        }

        errors = {dimension: [] for dimension in self.DIMENSIONS}
        matched_nodes = gold_assessments.keys() & predicted_assessments.keys()

        for node_id in matched_nodes:
            gold_rubric = gold_assessments[node_id]
            pred_rubric = predicted_assessments[node_id]

            for dimension in self.DIMENSIONS:
                if dimension in pred_rubric:
                    errors[dimension].append(
                        abs(getattr(gold_rubric, dimension) - pred_rubric[dimension])
                    )

        metrics = {
            dimension: self._calculate_metrics(values)
            for dimension, values in errors.items()
        }

        all_errors = [value for values in errors.values() for value in values]
        metrics["overall"] = self._calculate_metrics(all_errors)
        metrics["matched_nodes"] = len(matched_nodes)

        return metrics

    def _calculate_metrics(self, errors: list[float]) -> dict[str, float]:
        """Calculate regression error metrics."""
        if not errors:
            return {"mae": 0.0, "rmse": 0.0}

        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))

        return {"mae": mae, "rmse": rmse}
