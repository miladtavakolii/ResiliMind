from __future__ import annotations

from typing import Any

from evaluation.evaluators.base import BaseEvaluator
from evaluation.judges.gemini import GeminiJudge


class ResponseEvaluator(BaseEvaluator):
    """Evaluates final advisor responses using an LLM judge.

    This evaluator measures qualitative properties that cannot
    be captured by deterministic metrics.

    Attributes:
        name (str): Identifier name for the evaluator.
        judge (GeminiJudge): LLM-based judge instance used for response evaluation.
    """

    name: str = "response"

    def __init__(self, judge: GeminiJudge) -> None:
        """Initialize the response evaluator.

        Args:
            judge: GeminiJudge instance configured for LLM-based evaluation.
        """
        self.judge = judge

    def evaluate(self, gold: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the advisor response against ground truth data using the LLM judge.

        Args:
            gold: EvaluationGold object containing ground truth annotations.
            prediction: Workflow output dictionary containing the advisor response.

        Returns:
            dict[str, Any]: Dictionary containing evaluation metrics from the judge,
                or an error dict if the response is missing.
        """
        response = prediction.get("advisor_response")
        if not response:
            return {"error": "missing response"}

        prompt = self._build_prompt(gold, response)
        return self.judge.evaluate(prompt)

    def _build_prompt(self, gold: Any, response: str) -> str:
        """Construct the prompt string for the LLM judge evaluation.

        Args:
            gold: EvaluationGold object containing ground truth signals and assessment.
            response: The generated advisor response text.

        Returns:
            str: Formatted prompt string for the LLM judge.
        """
        signals = [
            {"node_id": item.node_id, "signal": item.detected_signal}
            for item in gold.extraction.active_signals
        ]

        return f"""Evaluate this AI advisor response.

Ground truth signals:
{signals}

Assessment:
{gold.assessment.model_dump()}

Advisor response:
{response}

Return JSON only."""
