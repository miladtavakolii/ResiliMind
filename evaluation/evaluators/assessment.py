from __future__ import annotations

import math
from typing import Any, Sequence

from evaluation.evaluators.base import BaseEvaluator


class AssessmentEvaluator(BaseEvaluator):
    """Evaluates the Assessor Agent score prediction quality.

    The Assessor predicts resilience dimensions:
        - severity
        - frequency
        - functional
        - coping

    This evaluator treats assessment as a regression problem and
    measures distance between predicted and ground-truth scores.

    Attributes:
        name (str): Identifier name for the evaluator.
        DIMENSIONS (tuple[str, ...]): Target resilience evaluation dimensions.
    """

    name: str = "assessment"
    DIMENSIONS: tuple[str, ...] = ("severity", "frequency", "functional", "coping")

    def evaluate(self, gold: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """Compare predicted assessment scores with ground truth.

        Args:
            gold: EvaluationGold object containing ground truth assessments.
            prediction: Workflow output dictionary containing predicted assessments.

        Returns:
            dict[str, Any]: Regression evaluation metrics (MAE, RMSE) broken down
                per dimension and overall, along with the count of matched nodes.
        """
        gold_assessments = {
            item.node_id: item.rubric for item in gold.assessment.assessments
        }
        predicted_assessments = {
            item.get("node_id"): item.get("rubric", {})
            for item in prediction.get("assessment", {}).get("assessments", [])
            if item.get("node_id")
        }

        gold_nodes = set(gold_assessments)
        predicted_nodes = set(predicted_assessments)
        matched_nodes = gold_nodes & predicted_nodes
        missing_nodes = sorted(gold_nodes - predicted_nodes)
        unexpected_nodes = sorted(predicted_nodes - gold_nodes)

        errors = {dimension: [] for dimension in self.DIMENSIONS}
        for node_id in matched_nodes:
            gold_rubric = gold_assessments[node_id]
            pred_rubric = predicted_assessments[node_id]

            for dimension in self.DIMENSIONS:
                if dimension not in pred_rubric:
                    errors[dimension].append(25)
                    continue

                predicted_value = pred_rubric[dimension]

                if not isinstance(predicted_value, (int, float)):
                    errors[dimension].append(25)
                    continue

                errors[dimension].append(
                    abs(getattr(gold_rubric, dimension) - predicted_value)
                )

        status_correct = 0
        status_total = len(matched_nodes)
        prediction_items = {
            item.get("node_id"): item
            for item in prediction.get("assessment", {}).get("assessments", [])
        }

        for node_id in matched_nodes:
            gold_status = gold_assessments[node_id].status
            predicted_item = prediction_items.get(node_id)

            if predicted_item is None:
                continue

            predicted_rubric = predicted_item.get("rubric", {})
            predicted_score = sum(
                predicted_rubric.get(dimension, 0)
                for dimension in self.DIMENSIONS
            )

            predicted_status = (
                "GREEN"
                if predicted_score >= 70
                else "YELLOW"
                if predicted_score >= 40
                else "RED"
            )

            if predicted_status == gold_status:
                status_correct += 1

        missing_dimensions = {
            node_id: [
                dimension
                for dimension in self.DIMENSIONS
                if dimension not in predicted_assessments[node_id]
            ]
            for node_id in matched_nodes
            if any(
                dimension not in predicted_assessments[node_id]
                for dimension in self.DIMENSIONS
            )
        }

        metrics = {
            dimension: self._calculate_metrics(values)
            for dimension, values in errors.items()
        }
        all_errors = [value for values in errors.values() for value in values]

        metrics["overall"] = self._calculate_metrics(all_errors)
        metrics["matched_nodes"] = len(matched_nodes)
        metrics["missing_nodes"] = missing_nodes
        metrics["unexpected_nodes"] = unexpected_nodes
        metrics["coverage"] = len(matched_nodes) / len(gold_nodes) if gold_nodes else 1.0
        metrics["status"] = {
            "accuracy": status_correct / status_total if status_total else 1.0,
            "correct": status_correct,
            "total": status_total,
        }
        metrics["missing_dimensions"] = missing_dimensions

        return metrics

    def _calculate_metrics(self, errors: Sequence[float]) -> dict[str, float]:
        """Calculate regression error metrics from absolute error values.

        Args:
            errors: Sequence of absolute error values.

        Returns:
            dict[str, float]: Dictionary containing Mean Absolute Error ('mae')
                and Root Mean Squared Error ('rmse').
        """
        if not errors:
            return {"mae": 0.0, "rmse": 0.0}

        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))

        return {"mae": mae, "rmse": rmse}
