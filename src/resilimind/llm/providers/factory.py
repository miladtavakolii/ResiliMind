import logging
from typing import Any, Dict, Type

from ...core.config import settings
from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .gemini import GeminiProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory and Registry for instantiating concrete BaseLLMProvider implementations.

    Adheres to the Open-Closed Principle (OCP) by allowing runtime registration of new
    providers (e.g. OpenAI, Gemini, Anthropic) without altering the calling code or core workflow.
    """

    _registry: Dict[str, Type[BaseLLMProvider]] = {
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: Type[BaseLLMProvider]) -> None:
        """
        Registers a new LLM provider class in the factory registry.

        Args:
            name: Case-insensitive unique name identifier (e.g. 'openai', 'gemini').
            provider_cls: Subclass implementing BaseLLMProvider.
        """
        key = name.strip().lower()
        cls._registry[key] = provider_cls
        logger.info("[LLMProviderFactory] Registered provider '%s' (%s).", key, provider_cls.__name__)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Returns the list of currently registered provider names."""
        return sorted(cls._registry.keys())

    @classmethod
    def create(
        cls,
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """
        Instantiates a BaseLLMProvider according to the requested or configured provider name.

        Args:
            provider_name: The name of the provider to instantiate. Defaults to settings.RESILIMIND_LLM_PROVIDER.
            **kwargs: Extra parameters forwarded directly to the provider's constructor.

        Returns:
            BaseLLMProvider: The initialized provider instance.

        Raises:
            ValueError: If the requested provider name is not found in the registry.
        """
        target_name = (provider_name or settings.RESILIMIND_LLM_PROVIDER).strip().lower()
        provider_cls = cls._registry.get(target_name)

        if provider_cls is None:
            supported = cls.get_supported_providers()
            logger.error(
                "[LLMProviderFactory] Unsupported provider '%s'. Registered providers: %s",
                target_name,
                supported,
            )
            raise ValueError(
                f"Unsupported LLM provider '{target_name}'. Available providers: {supported}"
            )

        logger.debug("[LLMProviderFactory] Creating provider instance for '%s'.", target_name)
        return provider_cls(**kwargs)
