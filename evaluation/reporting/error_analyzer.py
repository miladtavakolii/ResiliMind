from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FailureCase:
    """Represents a failed benchmark case.

    Stores the case identifier, failure category, and detailed metric information.

    Attributes:
        case_id: Unique identifier for the benchmark case.
        category: Failure category identifier.
        details: Metric and diagnostic details associated with the failure.
    """

    case_id: str
    category: str
    details: dict[str, Any]


class ErrorAnalyzer:
    """Analyzes benchmark results and extracts failure cases.

    The analyzer does not modify evaluation metrics.
    It only provides diagnostic information.
    """

    def analyze(self, results: list[Any]) -> list[FailureCase]:
        """Detect failure cases from evaluation results.

        Args:
            results: List of evaluation case result objects.

        Returns:
            list[FailureCase]: List of identified failure cases across all results.
        """
        failures = []
        for result in results:
            failures.extend(self._analyze_case(result))
        return failures

    def _analyze_case(self, result: Any) -> list[FailureCase]:
        """Extract failure cases from a single evaluation result based on thresholds.

        Args:
            result: Single case evaluation result containing case_id and metrics.

        Returns:
            list[FailureCase]: List of failure cases identified for the given case.
        """
        failures = []
        metrics = result.metrics

        if safety := metrics.get("safety"):
            confusion = safety.get("confusion_matrix", {})
            if confusion.get("fn", 0) > 0:
                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="safety_false_negative",
                        details=safety,
                    )
                )
            if confusion.get("fp", 0) > 0:
                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="safety_false_positive",
                        details=safety,
                    )
                )

        if extraction := metrics.get("extraction"):
            node_metrics = extraction.get("node_detection", {})
            if node_metrics.get("recall", 1.0) < 1.0:
                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="extraction_missed_signal",
                        details=extraction,
                    )
                )
            if node_metrics.get("precision", 1.0) < 1.0:
                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="extraction_false_positive",
                        details=extraction,
                    )
                )

        if (routing := metrics.get("routing")) and routing.get("correct", 1) == 0:
            failures.append(
                FailureCase(
                    case_id=result.case_id,
                    category="routing_failure",
                    details=routing,
                )
            )

        if assessment := metrics.get("assessment"):
            if assessment.get("overall", {}).get("mae", 0) > 10:
                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="assessment_failure",
                        details=assessment,
                    )
                )

        if (response := metrics.get("response")) and response.get("overall", 10) < 5:
            failures.append(
                FailureCase(
                    case_id=result.case_id,
                    category="advisor_failure",
                    details=response,
                )
            )

        return failures
