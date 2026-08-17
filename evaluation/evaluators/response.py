from __future__ import annotations

from typing import Any
from pathlib import Path

from evaluation.evaluators.base import BaseEvaluator
from evaluation.judges.gemini import GeminiJudge

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT_PATH = PROJECT_ROOT / "prompts" / "response_judge.txt"

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
        self.prompt_template = (PROMPT_PATH.read_text(encoding="utf-8"))

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
        user_context = prediction.get("user_context","")
        if not response:
            return {"error": "missing response"}
        signals = prediction.get("extraction", {}).get("signals", [])

        prompt = self._build_prompt(user_context=user_context,signals=signals,assessment=gold.assessment.model_dump(),response=response)
        return self.judge.evaluate(prompt)

    def _build_prompt(
        self,
        *,
        user_context: str,
        signals: list[dict[str, Any]],
        assessment: dict[str, Any],
        response: str,
    ) -> str:
        """Build the prompt from the repository judge template.

        Args:
            user_context: Aggregated text or conversation history from the user.
            signals: List of extracted resilience signal dictionaries.
            assessment: Assessment dictionary containing resilience scores and rubrics.
            response: Generated advisor response text to be evaluated.

        Returns:
            str: Formatted prompt string ready for LLM judge evaluation.
        """
        return (
            f"{self.prompt_template}\n\n"
            f"=== USER CONTEXT ===\n{user_context}\n\n"
            f"=== EXTRACTED SIGNALS ===\n{signals}\n\n"
            f"=== ASSESSMENT ===\n{assessment}\n\n"
            f"=== ADVISOR RESPONSE ===\n{response}\n\n"
            "Return ONLY the JSON object defined by the output schema."
        )
