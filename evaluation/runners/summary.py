from __future__ import annotations

from typing import Any
from collections.abc import Sequence

from evaluation.schemas import CasePrediction


def build_execution_summary(
    predictions: Sequence[CasePrediction],
) -> dict[str, int]:
    """Build execution-level success statistics.

    Args:
        predictions: Benchmark predictions.

    Returns:
        Total, successful, and failed execution counts.
    """
    successful = sum(prediction.successful for prediction in predictions)
    return {
        "total": len(predictions),
        "successful": successful,
        "failed": len(predictions) - successful,
    }


def build_final_summary(
    summary: dict[str, Any],
    predictions: Sequence[CasePrediction],
) -> dict[str, Any]:
    """Add benchmark execution statistics to aggregated evaluation metrics.

    Args:
        summary: Dataset-level evaluator metrics.
        predictions: Raw benchmark predictions.

    Returns:
        Combined evaluation summary.
    """
    return {
        **summary,
        "execution": build_execution_summary(predictions),
    }
