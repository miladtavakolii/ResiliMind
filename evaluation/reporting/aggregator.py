from __future__ import annotations

from statistics import mean
from typing import Any

from evaluation.schemas import CaseEvaluationResult


class EvaluationAggregator:
    """Aggregates per-case evaluation results into dataset-level metrics."""

    def aggregate(self, results: list[CaseEvaluationResult]) -> dict[str, Any]:
        """Generate overall evaluation statistics across all evaluated cases.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, Any]: Aggregated metrics for safety, assessment, routing,
                and response evaluation alongside dataset size.
        """
        return {
            "dataset_size": len(results),
            "safety": self._aggregate_safety(results),
            "assessment": self._aggregate_assessment(results),
            "routing": self._aggregate_routing(results),
            "response": self._aggregate_response(results),
        }

    def _aggregate_safety(self, results: list[CaseEvaluationResult]) -> dict[str, float]:
        """Calculate aggregate safety metrics across evaluation results.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, float]: Dictionary containing mean safety classification accuracy.
        """
        values = [
            metric.get("accuracy", 0)
            for result in results
            if (metric := result.metrics.get("safety"))
        ]
        return {"accuracy": mean(values) if values else 0}

    def _aggregate_assessment(self, results: list[CaseEvaluationResult]) -> dict[str, float]:
        """Calculate aggregate assessment regression metrics across evaluation results.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, float]: Dictionary containing mean MAE across assessment dimensions.
        """
        maes = [
            metric.get("overall", {}).get("mae", 0)
            for result in results
            if (metric := result.metrics.get("assessment"))
        ]
        return {"mean_mae": mean(maes) if maes else 0}

    def _aggregate_routing(self, results: list[CaseEvaluationResult]) -> dict[str, float]:
        """Calculate aggregate workflow routing accuracy across evaluation results.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, float]: Dictionary containing overall routing accuracy.
        """
        correct = 0
        total = 0
        for result in results:
            if metric := result.metrics.get("routing"):
                total += 1
                correct += metric.get("correct", 0)
        return {"accuracy": correct / total if total else 0}

    def _aggregate_response(self, results: list[CaseEvaluationResult]) -> dict[str, float]:
        """Calculate aggregate qualitative response scores from LLM judge outputs.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, float]: Dictionary containing average judge evaluation score.
        """
        scores = [
            metric.get("overall", 0)
            for result in results
            if (metric := result.metrics.get("response"))
        ]
        return {"average_score": mean(scores) if scores else 0}
