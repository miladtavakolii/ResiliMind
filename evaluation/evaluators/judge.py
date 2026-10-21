from __future__ import annotations

import logging
import time
from typing import Any

from evaluation.schemas import ResponseJudgeResult
from src.resilimind.core.config import settings
from src.resilimind.llm.providers.base import BaseLLMProvider
from src.resilimind.llm.providers.factory import LLMProviderFactory

logger = logging.getLogger(__name__)


class ResponseJudge:
    """Multi-provider LLM evaluator for qualitative response assessment.

    Uses an LLM call via ResiliMind's BaseLLMProvider to judge advisor responses
    with exponential backoff retries and structured output validation.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
        request_delay: float = 2.0,
    ) -> None:
        """Initialize the response judge.

        Args:
            provider: LLM provider name ('gemini', 'openai', 'anthropic', 'ollama').
                If omitted, falls back to settings.RESILIMIND_LLM_PROVIDER.
            model: Model identifier used for evaluation.
            api_key: Optional provider API key.
            base_url: Optional provider base URL.
            max_retries: Maximum number of retry attempts upon failure.
            retry_delay: Base delay in seconds between retries.
            request_delay: Delay between requests.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        if request_delay < 0:
            raise ValueError("request_delay must be >= 0")

        self.provider_name = (provider or settings.RESILIMIND_LLM_PROVIDER).strip().lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay

        factory_kwargs: dict[str, Any] = {}
        if model:
            factory_kwargs["model_name"] = model
        if api_key:
            factory_kwargs["api_key"] = api_key
        if base_url:
            factory_kwargs["base_url"] = base_url

        self.provider: BaseLLMProvider = LLMProviderFactory.create(
            self.provider_name,
            **factory_kwargs,
        )

        self.runner = self.provider.get_structured_model(
            schema=ResponseJudgeResult,
            temperature=0.0,
        )
        logger.info(
            "[ResponseJudge] Initialized judge with provider '%s'%s.",
            self.provider_name,
            f" (model='{self.model}')" if self.model else "",
        )

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
            "ratelimit",
            "resource exhausted",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "connection",
            "overloaded",
        )
        return any(marker in message for marker in retryable_markers)

    def evaluate(self, prompt: str) -> dict[str, Any]:
        """Evaluate a prompt using the configured model and return the parsed JSON response.

        Args:
            prompt: Formatted evaluation prompt string.

        Returns:
            Parsed JSON evaluation results as a dictionary.

        Raises:
            RuntimeError: If the request fails across all retry attempts.
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0 and self.request_delay > 0:
                    logger.info(
                        "Waiting %.1f seconds before %s request...",
                        self.request_delay,
                        self.provider_name,
                    )
                    time.sleep(self.request_delay)

                raw_result = self.runner.invoke(prompt)

                if isinstance(raw_result, dict):
                    if raw_result.get("parsed") is not None:
                        result: ResponseJudgeResult = raw_result["parsed"]
                    elif "raw" in raw_result:
                        raise ValueError(f"Judge model returned invalid structured output: {raw_result.get('raw')!r}")
                    else:
                        result = ResponseJudgeResult.model_validate(raw_result)
                elif isinstance(raw_result, ResponseJudgeResult):
                    result = raw_result
                else:
                    raise ValueError(f"Unexpected response format from judge model: {type(raw_result)}")

                return result.model_dump()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Judge failed attempt %d/%d: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

                if not self._is_retryable_error(exc):
                    raise

                if attempt >= self.max_retries:
                    break

                delay = self.retry_delay * (2**attempt)
                logger.info("Retrying in %.1f seconds...", delay)
                time.sleep(delay)

        raise RuntimeError(
            f"{self.provider_name} judge failed after {self.max_retries + 1} attempts"
        ) from last_error
