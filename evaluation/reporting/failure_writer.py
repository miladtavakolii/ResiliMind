from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class FailureCSVWriter:
    """Writes evaluation failures into CSV format for manual analysis."""

    def write(self, failures: Iterable[Any], output: Path) -> None:
        """Write a collection of failure cases to a CSV file.

        Args:
            failures: Iterable of failure case objects containing case_id, category, and details.
            output: Destination file path for the output CSV.
        """
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["case_id", "category", "details"])

            for failure in failures:
                writer.writerow([failure.case_id, failure.category, failure.details])
