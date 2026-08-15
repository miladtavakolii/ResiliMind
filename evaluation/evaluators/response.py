from __future__ import annotations

from typing import Any

from evaluation.evaluators.base import BaseEvaluator
from evaluation.judges.gemini import GeminiJudge


class ResponseEvaluator(BaseEvaluator):
    """
    Evaluates final advisor responses using an LLM judge.

    This evaluator measures qualitative properties that cannot
    be captured by deterministic metrics.
    """

    name = "response"


    def __init__(
        self,
        judge: GeminiJudge,
    ) -> None:

        self.judge = judge


    def evaluate(
        self,
        gold: Any,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:

        response = (
            prediction
            .get("advisor_response")
        )

        if not response:
            return {
                "error": "missing response"
            }


        prompt = self._build_prompt(
            gold,
            response,
        )


        return self.judge.evaluate(
            prompt
        )


    def _build_prompt(
        self,
        gold,
        response: str,
    ) -> str:

        signals = [
            {
                "node_id": item.node_id,
                "signal": item.detected_signal,
            }
            for item in gold.extraction.active_signals
        ]


        return f"""
Evaluate this AI advisor response.

Ground truth signals:
{signals}


Assessment:
{gold.assessment.model_dump()}


Advisor response:

{response}


Return JSON only.
"""
