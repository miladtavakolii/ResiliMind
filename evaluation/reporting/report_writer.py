from __future__ import annotations

from pathlib import Path
from typing import Any


class MarkdownReportWriter:
    """Writes human-readable evaluation reports."""

    def write(self, report: dict[str, Any], path: Path) -> None:
        """Write evaluation metrics to a formatted Markdown report file.

        Args:
            report: Dictionary containing aggregated evaluation statistics across
                dataset size, safety, assessment, routing, and response quality.
            path: Destination file path for the Markdown report.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        content = f"""# ResiliMind Evaluation Report

## Dataset
Size: {report["dataset_size"]}

## Safety
Accuracy: {report["safety"]["accuracy"]:.3f}

## Assessment
Mean MAE: {report["assessment"]["mean_mae"]:.3f}

## Routing
Accuracy: {report["routing"]["accuracy"]:.3f}

## Advisor Response Quality
Average Score: {report["response"]["average_score"]:.2f}/10
"""

        path.write_text(content, encoding="utf-8")
