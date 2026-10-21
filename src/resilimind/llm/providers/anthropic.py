import logging
from typing import Any, Type
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

from ...core.config import settings
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    LLM provider adapter for Anthropic Claude models via ChatAnthropic.

    Provides high-empathy, psychologically nuanced conversational responses
    and robust structured Pydantic output validation for ResiliMind.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Anthropic provider.

        Args:
            model_name: Name of the Claude model. Defaults to settings.RESILIMIND_LLM_MODEL.
                If set to Ollama's default ('gemma4:e2b'), falls back to 'claude-3-5-sonnet-latest'.
            api_key: Anthropic API key. Defaults to settings.RESILIMIND_LLM_API_KEY.
            base_url: Optional API endpoint base URL (defaults to settings.RESILIMIND_LLM_BASE_URL).
            **kwargs: Extra parameters reserved for future extension.

        Raises:
            ValueError: If no valid API key is found.
        """
        resolved_model = model_name or settings.RESILIMIND_LLM_MODEL
        if resolved_model == "gemma4:e2b":
            resolved_model = "claude-3-5-sonnet-latest"
            logger.info(
                "[AnthropicProvider] RESILIMIND_LLM_MODEL is set to Ollama default ('gemma4:e2b'); "
                "falling back to '%s' for Anthropic.",
                resolved_model,
            )

        self.model_name: str = resolved_model
        self.api_key: str | None = api_key or settings.RESILIMIND_LLM_API_KEY
        self.base_url: str | None = base_url or settings.RESILIMIND_LLM_BASE_URL

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required when using the 'anthropic' provider. "
                "Please configure RESILIMIND_LLM_API_KEY in your .env file or environment variables."
            )

        self.extra_kwargs = kwargs
        logger.info(
            "[AnthropicProvider] Initialized successfully with model '%s'%s.",
            self.model_name,
            f" at base_url '{self.base_url}'" if self.base_url else "",
        )

    def get_chat_model(
        self,
        temperature: float,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Builds a ChatAnthropic instance configured for conversational flow and streaming.
        """
        params: dict[str, Any] = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": temperature,
            "max_tokens": num_predict if (num_predict is not None and num_predict > 0) else 4096,
        }
        if self.base_url:
            params["anthropic_api_url"] = self.base_url

        params.update(self.extra_kwargs)
        params.update(kwargs)

        return ChatAnthropic(**params)

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
        Builds a ChatAnthropic instance bound to a Pydantic schema using structured output.
        """
        params: dict[str, Any] = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": temperature,
            "max_tokens": num_predict if (num_predict is not None and num_predict > 0) else 4096,
        }
        if self.base_url:
            params["anthropic_api_url"] = self.base_url

        params.update(self.extra_kwargs)
        params.update(kwargs)

        chat_model = ChatAnthropic(**params)
        return chat_model.with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
        )
