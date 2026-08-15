from __future__ import annotations

from statistics import mean
from typing import Any

from evaluation.schemas.evaluation_result import (
    CaseEvaluationResult,
)


class EvaluationAggregator:
    """
    Aggregates per-case evaluation results into
    dataset-level metrics.
    """


    def aggregate(
        self,
        results: list[CaseEvaluationResult],
    ) -> dict[str, Any]:
        """
        Generate overall evaluation statistics.
        """

        return {
            "dataset_size": len(results),
            "safety": self._aggregate_safety(results),
            "assessment": self._aggregate_assessment(results),
            "routing": self._aggregate_routing(results),
            "response": self._aggregate_response(results),
        }


    def _aggregate_safety(
        self,
        results,
    ):
        values = []

        for result in results:

            metric = (
                result.metrics
                .get("safety")
            )

            if metric:
                values.append(
                    metric.get(
                        "accuracy",
                        0,
                    )
                )

        return {
            "accuracy": mean(values)
            if values else 0
        }


    def _aggregate_assessment(
        self,
        results,
    ):
        maes = []

        for result in results:

            metric = (
                result.metrics
                .get("assessment")
            )

            if metric:

                overall = metric.get(
                    "overall",
                    {}
                )

                maes.append(
                    overall.get(
                        "mae",
                        0,
                    )
                )

        return {
            "mean_mae": mean(maes)
            if maes else 0
        }


    def _aggregate_routing(
        self,
        results,
    ):
        correct = 0
        total = 0

        for result in results:

            metric = (
                result.metrics
                .get("routing")
            )

            if metric:

                total += 1
                correct += metric.get(
                    "correct",
                    0,
                )

        return {
            "accuracy": (
                correct / total
                if total
                else 0
            )
        }


    def _aggregate_response(
        self,
        results,
    ):
        scores = []

        for result in results:

            metric = (
                result.metrics
                .get("response")
            )

            if metric:

                scores.append(
                    metric.get(
                        "overall",
                        0,
                    )
                )

        return {
            "average_score": (
                mean(scores)
                if scores
                else 0
            )
        }
