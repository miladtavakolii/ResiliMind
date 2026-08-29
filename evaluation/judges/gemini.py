from __future__ import annotations

import json
import logging
import time
from typing import Any

from evaluation.schemas import ResponseJudgeResult

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiJudge:
    """Gemini based evaluator for qualitative response assessment.

    Uses a separate LLM call to judge advisor responses.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        retries: int = 5,
        delay: float = 3.0,
    ) -> None:
        """Initialize the Gemini judge.

        Args:
            api_key: Google Gemini API key.
            model: Gemini model identifier used for evaluation.
            retries: Maximum number of retry attempts upon failure.
            delay: Base delay in seconds between retries.
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.retries = retries
        self.delay = delay


    def _is_retryable_error(self, exc: Exception) -> bool:
        """Return whether an exception represents a transient API failure.

        Args:
            exc: The exception instance raised during API invocation.

        Returns:
            bool: True if the exception matches known transient or rate-limit
                error patterns, False otherwise.
        """
        message = str(exc).lower()
        retryable_markers = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "rate limit",
            "resource exhausted",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "connection",
        )
        return any(marker in message for marker in retryable_markers)


    def evaluate(self, prompt: str) -> dict[str, Any]:
        """Evaluate a prompt using the Gemini model and return the parsed JSON response.

        Args:
            prompt: Formatted evaluation prompt string.

        Returns:
            Parsed JSON evaluation results as a dictionary.

        Raises:
            RuntimeError: If the request fails across all retry attempts.
        """
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=ResponseJudgeResult,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini returned an empty response")
                result = ResponseJudgeResult.model_validate_json(response.text)
                return result.model_dump()

            except Exception as exc:
                logger.warning("Judge failed attempt %s/%s: %s", attempt, self.retries, exc)

                if not self._is_retryable_error(exc):
                    raise

                if attempt < self.retries:
                    delay = self.delay * (2 ** (attempt - 1))
                    time.sleep(delay)

        raise RuntimeError("Gemini judge failed after retries")
