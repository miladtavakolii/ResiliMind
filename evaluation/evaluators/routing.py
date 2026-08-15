from __future__ import annotations

from typing import Any

from evaluation.evaluators.base import BaseEvaluator


class RoutingEvaluator(BaseEvaluator):
    """
    Evaluates workflow routing decisions.

    The evaluator measures whether LangGraph selected the correct
    next agent based on the case characteristics.

    Supported routes:
    - advisor
    - questioner
    - emergency_response
    """

    name = "routing"

    ROUTES = ("advisor", "questioner", "emergency_response")

    def evaluate(self, gold: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """
        Compare expected workflow route with predicted route.

        Args:
            gold: EvaluationGold object.
            prediction: Workflow execution output.

        Returns:
            Classification metrics.
        """
        expected = gold.routing.expected_route
        predicted = prediction.get("routing", {}).get("route")

        if predicted not in self.ROUTES:
            predicted = "unknown"

        labels = list(self.ROUTES)
        matrix = {label: {other: 0 for other in labels} for label in labels}

        if expected in labels and predicted in labels:
            matrix[expected][predicted] += 1

        correct = int(expected == predicted)

        return {
            "correct": correct,
            "expected": expected,
            "predicted": predicted,
            "confusion_matrix": matrix,
        }
