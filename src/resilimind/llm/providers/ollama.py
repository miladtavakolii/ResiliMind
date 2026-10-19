import logging
from typing import Any, Type
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

from ...core.config import settings
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    LLM provider adapter for local inference using Ollama and ChatOllama.

    Encapsulates Ollama-specific configuration parameters (such as context window size,
    prediction token limits, and reasoning flags) while exposing the unified BaseLLMProvider contract.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Ollama provider.

        Args:
            model_name: Name of the Ollama model (defaults to settings.RESILIMIND_LLM_MODEL).
            base_url: Endpoint URL of the Ollama service (defaults to settings.OLLAMA_BASE_URL).
            **kwargs: Extra parameters reserved for future extension.
        """
        self.model_name = model_name or settings.RESILIMIND_LLM_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.extra_kwargs = kwargs
        logger.info(
            "[OllamaProvider] Initialized with model '%s' at '%s'.",
            self.model_name,
            self.base_url,
        )

    def get_chat_model(
        self,
        temperature: float,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Builds a ChatOllama instance configured for conversational text generation and streaming.
        """
        params: dict[str, Any] = {
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": temperature,
        }
        if num_ctx is not None:
            params["num_ctx"] = num_ctx
        if num_predict is not None:
            params["num_predict"] = num_predict

        params.update(self.extra_kwargs)
        params.update(kwargs)

        return ChatOllama(**params)

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
        Builds a ChatOllama instance bound to a Pydantic schema using JSON Schema enforcement.
        """
        params: dict[str, Any] = {
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": temperature,
            "reasoning": False,
        }
        if num_ctx is not None:
            params["num_ctx"] = num_ctx
        if num_predict is not None:
            params["num_predict"] = num_predict

        params.update(self.extra_kwargs)
        params.update(kwargs)

        chat_model = ChatOllama(**params)
        return chat_model.with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
        )
