import logging
import os
from typing import Any, Type
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

from ...core.config import settings
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    LLM provider adapter for Google Gemini models via ChatGoogleGenerativeAI.

    Supports multimodal/chat generation and strict structured Pydantic output
    for ResiliMind's multi-stage safety, extraction, and assessment workflow.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Gemini provider.

        Args:
            model_name: Name of the Gemini model. Defaults to settings.RESILIMIND_LLM_MODEL.
                If set to Ollama's default ('gemma4:e2b'), falls back to 'gemini-2.5-flash'.
            api_key: Google Gemini API key. Defaults to settings.GEMINI_API_KEY
                or GEMINI_API_KEY / GOOGLE_API_KEY environment variables.
            **kwargs: Extra parameters reserved for future extension.

        Raises:
            ValueError: If no valid Gemini API key is found.
        """
        resolved_model = model_name or settings.RESILIMIND_LLM_MODEL
        if resolved_model == "gemma4:e2b":
            resolved_model = "gemini-2.5-flash"
            logger.info(
                "[GeminiProvider] RESILIMIND_LLM_MODEL is set to Ollama default ('gemma4:e2b'); "
                "falling back to '%s' for Gemini.",
                resolved_model,
            )

        self.model_name: str = resolved_model
        self.api_key: str | None = (
            api_key
            or settings.GEMINI_API_KEY
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required when using the 'gemini' provider. "
                "Please configure GEMINI_API_KEY in your .env file or environment variables."
            )

        self.extra_kwargs = kwargs
        logger.info(
            "[GeminiProvider] Initialized successfully with model '%s'.",
            self.model_name,
        )

    def get_chat_model(
        self,
        temperature: float,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Builds a ChatGoogleGenerativeAI instance configured for conversational flow and streaming.
        """
        params: dict[str, Any] = {
            "model": self.model_name,
            "google_api_key": self.api_key,
            "temperature": temperature,
        }
        if num_predict is not None and num_predict > 0:
            params["max_output_tokens"] = num_predict

        params.update(self.extra_kwargs)
        params.update(kwargs)

        return ChatGoogleGenerativeAI(**params)

    def get_structured_model(
        self,
        schema: Type[BaseModel],
        temperature: float = 0.0,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        method: str = "json_schema",
        include_raw: bool = True,
        **kwargs: Any,
    ) -> RunnableSerializable[Any, Any]:
        """
        Builds a ChatGoogleGenerativeAI instance bound to a Pydantic schema using structured output.
        """
        params: dict[str, Any] = {
            "model": self.model_name,
            "google_api_key": self.api_key,
            "temperature": temperature,
        }
        if num_predict is not None and num_predict > 0:
            params["max_output_tokens"] = num_predict

        params.update(self.extra_kwargs)
        params.update(kwargs)

        chat_model = ChatGoogleGenerativeAI(**params)
        return chat_model.with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
        )
