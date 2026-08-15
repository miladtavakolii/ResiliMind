from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FailureCase:
    """
    Represents a failed benchmark case.

    Stores the case identifier, failure category,
    and detailed metric information.
    """

    case_id: str
    category: str
    details: dict[str, Any]


class ErrorAnalyzer:
    """
    Analyzes benchmark results and extracts failure cases.

    The analyzer does not modify evaluation metrics.
    It only provides diagnostic information.
    """

    def analyze(
        self,
        results: list,
    ) -> list[FailureCase]:
        """
        Detect failure cases from evaluation results.
        """

        failures = []

        for result in results:

            failures.extend(
                self._analyze_case(result)
            )

        return failures


    def _analyze_case(
        self,
        result,
    ) -> list[FailureCase]:

        failures = []

        metrics = result.metrics


        safety = metrics.get(
            "safety"
        )

        if safety:

            if safety.get(
                "correct"
            ) is False:

                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="safety_failure",
                        details=safety,
                    )
                )


        routing = metrics.get(
            "routing"
        )

        if routing:

            if routing.get(
                "correct"
            ) is False:

                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="routing_failure",
                        details=routing,
                    )
                )


        assessment = metrics.get(
            "assessment"
        )

        if assessment:

            if (
                assessment
                .get("mae", 0)
                > 10
            ):

                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="assessment_failure",
                        details=assessment,
                    )
                )


        response = metrics.get(
            "response"
        )

        if response:

            if (
                response.get(
                    "overall",
                    10,
                )
                < 5
            ):

                failures.append(
                    FailureCase(
                        case_id=result.case_id,
                        category="advisor_failure",
                        details=response,
                    )
                )


        return failures
