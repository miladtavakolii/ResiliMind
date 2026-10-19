import logging
from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class establishing the contract for all LLM providers in ResiliMind.

    Concrete implementations (e.g. OllamaProvider, OpenAIProvider, GeminiProvider)
    must implement methods to produce LangChain chat models for free-form conversation
    and structured Pydantic runners for downstream analytical and classification agents.
    """

    @abstractmethod
    def get_chat_model(
        self,
        temperature: float,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Creates and returns a LangChain BaseChatModel for conversation and streaming.

        Args:
            temperature: Sampling temperature for generative freedom.
            num_ctx: Optional context window token limit.
            num_predict: Optional maximum generated tokens limit.
            **kwargs: Additional provider-specific keyword arguments.

        Returns:
            BaseChatModel: Initialized LangChain chat model.
        """
        pass

    @abstractmethod
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
        Creates and returns a LangChain runnable configured with structured Pydantic output.

        Args:
            schema: The target Pydantic BaseModel class for validation.
            temperature: Sampling temperature (default 0.0 for deterministic triage).
            num_ctx: Optional context window token limit.
            num_predict: Optional maximum tokens limit.
            method: Structured output method (e.g., 'json_schema', 'function_calling').
            include_raw: Whether to include the raw LLM response in the output dict.
            **kwargs: Additional provider-specific keyword arguments.

        Returns:
            RunnableSerializable: Runnable pipeline outputting validated schema data.
        """
        pass
