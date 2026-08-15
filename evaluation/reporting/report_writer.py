from pathlib import Path


class MarkdownReportWriter:
    """Writes human-readable evaluation reports."""

    def write(self, report: dict, path: Path) -> None:
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
