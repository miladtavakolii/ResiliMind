from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CaseEvaluationResult(BaseModel):
    """
    Stores evaluation results for a single benchmark case.

    Each evaluator contributes its own metric namespace.
    """

    case_id: str

    metrics: dict[str, Any] = Field(
        default_factory=dict
    )


class EvaluationSummary(BaseModel):
    """
    Aggregated benchmark evaluation report.

    Contains metrics calculated across the complete dataset.
    """

    dataset_size: int

    evaluators: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )
