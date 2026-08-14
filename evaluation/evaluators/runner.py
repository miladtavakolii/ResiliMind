from __future__ import annotations

from collections.abc import Iterable

from evaluation.evaluators.base import BaseEvaluator
from evaluation.schemas import EvaluationCase
from evaluation.schemas.evaluation_result import (
    CaseEvaluationResult,
    EvaluationSummary,
)


class EvaluationRunner:
    """
    Coordinates execution of multiple evaluation modules.

    The runner itself does not implement metrics.
    It delegates evaluation logic to registered evaluators.
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
    ) -> None:
        self.evaluators = evaluators

    def evaluate_case(
        self,
        case: EvaluationCase,
        prediction: dict,
    ) -> CaseEvaluationResult:
        """
        Evaluate a single benchmark case.

        Args:
            case:
                Benchmark ground truth case.

            prediction:
                ResiliMind generated output.

        Returns:
            Evaluation result for the case.
        """

        metrics = {}

        for evaluator in self.evaluators:
            metrics[evaluator.name] = evaluator.evaluate(
                gold=case.gold,
                prediction=prediction,
            )

        return CaseEvaluationResult(
            case_id=case.case_id,
            metrics=metrics,
        )

    def evaluate_dataset(
        self,
        cases: Iterable[EvaluationCase],
        predictions: dict[str, dict],
    ) -> list[CaseEvaluationResult]:
        """
        Evaluate all benchmark cases.

        Args:
            cases:
                Iterable of benchmark cases.

            predictions:
                Mapping from case_id to model output.

        Returns:
            List of per-case evaluation results.
        """

        results = []

        for case in cases:
            prediction = predictions.get(
                case.case_id,
                {},
            )

            results.append(
                self.evaluate_case(
                    case,
                    prediction,
                )
            )

        return results

    def summarize(
        self,
        results: list[CaseEvaluationResult],
    ) -> EvaluationSummary:
        """
        Aggregate per-case metrics.

        Detailed aggregation will be implemented by individual
        metric modules.
        """

        grouped = {}

        for result in results:
            for evaluator, metrics in result.metrics.items():
                grouped.setdefault(
                    evaluator,
                    [],
                ).append(metrics)

        return EvaluationSummary(
            dataset_size=len(results),
            evaluators=grouped,
        )
