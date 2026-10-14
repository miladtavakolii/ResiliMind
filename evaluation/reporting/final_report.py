from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class FinalReportGenerator:
    """Generates final benchmark summary report."""

    def generate(
        self, summary: dict[str, Any], failures: Sequence[Any], output: Path
    ) -> None:
        """Generate and save the final evaluation JSON report including summary and failure stats.

        Args:
            summary: Dictionary containing aggregated benchmark metrics.
            failures: Sequence of failure case objects.
            output: Destination file path for the JSON report.
        """
        output.parent.mkdir(parents=True, exist_ok=True)

        failure_events = len(failures)
        unique_cases = {
            getattr(f, "case_id", f.get("case_id") if isinstance(f, dict) else None)
            for f in failures
        }
        unique_cases.discard(None)
        unique_failed_cases = len(unique_cases)

        report = {
            **summary,
            "error_statistics": {
                "failure_events": failure_events,
                "unique_failed_cases": unique_failed_cases,
                "total_failures": failure_events,
                "failure_types": self._count_failures(failures),
            },
        }

        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _count_failures(self, failures: Sequence[Any]) -> dict[str, int]:
        """Count occurrences of each failure category.

        Args:
            failures: Sequence of failure case objects with a category attribute.

        Returns:
            dict[str, int]: Mapping of failure categories to their occurrence counts.
        """
        counter = {}
        for failure in failures:
            counter[failure.category] = counter.get(failure.category, 0) + 1
        return counter
