import logging
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableSerializable

from ..schemas.models import (
    ExtractionOutput,
    AssessmentOutput,
    SafetyOutput,
    NodeSelectionOutput,
    SignalResolutionOutput,
    PolarityEvidenceIndexOutput,
)
from ..core.config import settings
from .providers.base import BaseLLMProvider
from .providers.factory import LLMProviderFactory

# Initialize module logger
logger = logging.getLogger(__name__)


class LLMEngine:
    """
    True Singleton facade class for orchestrating LLM interactions across ResiliMind.

    Manages the lifecycle of LLM instances to prevent unnecessary object instantiation,
    memory leaks, and initialization overhead. It delegates LLM creation to a configured
    BaseLLMProvider (resolved via LLMProviderFactory) and provides cached, task-specific
    runnable chains (e.g., extraction, assessment, safety, conversational) tailored with
    appropriate temperatures and structured output schemas.

    Attributes:
        provider_name (str): The active LLM provider name (e.g. 'ollama', 'openai', 'gemini').
        provider (BaseLLMProvider): The active provider instance.
        model_name (str): The name of the underlying model instance.
        base_url (str): The endpoint URL (for providers that use an HTTP endpoint).
        conversational_temp (float): Temperature parameter for the conversational LLM.
        is_initialized (bool): Flag indicating if the Singleton instance has been set up.
    """
    
    _instance: Optional["LLMEngine"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "LLMEngine":
        """
        Enforces the Singleton pattern by creating the instance only if it does not exist.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            LLMEngine: The single, shared instance of the class.
        """
        if cls._instance is None:
            cls._instance = super(LLMEngine, cls).__new__(cls)
            logger.debug("[LLMEngine] Creating new singleton instance of LLMEngine.")
        return cls._instance

    def __init__(self) -> None:
        """
        Initializes the LLMEngine using centralized environment settings and the provider factory.
        Safely prevents re-initialization if the Singleton instance has already been configured.
        """
        if not hasattr(self, 'is_initialized'):
            self.provider_name: str = settings.RESILIMIND_LLM_PROVIDER
            self.model_name: str = settings.RESILIMIND_LLM_MODEL
            self.base_url: str = settings.OLLAMA_BASE_URL
            self.conversational_temp: float = settings.RESILIMIND_LLM_TEMPERATURE
            
            # Instantiate provider via factory
            self.provider: BaseLLMProvider = LLMProviderFactory.create(self.provider_name)
            
            # Lazy-loaded, cached LLM runnable instances
            self._extractor_llm: Optional[Any] = None
            self._assessor_llm: Optional[Any] = None
            self._safety_llm: Optional[Any] = None
            self._conversational_llm: Optional[BaseChatModel] = None
            self._node_selector_llm: Optional[Any] = None
            self._signal_resolver_llm: Optional[Any] = None
            self._polarity_evidence_llm: Optional[Any] = None
            self.is_initialized: bool = True
            logger.info(
                "[LLMEngine] Initialized engine (provider='%s', model='%s').",
                self.provider_name,
                self.model_name,
            )

    def get_node_selector_runner(self, system_prompt: str) -> RunnableSerializable[dict[str, Any], Any]:
        """Build and cache the LangChain runnable pipeline for candidate node selection.

        Initializes a structured-output LLM configured with the `NodeSelectionOutput`
        JSON schema and binds it to a chat prompt template accepting `user_message`.

        Args:
            system_prompt: System instructions and knowledge graph context for node selection.

        Returns:
            RunnableSerializable: Composed LCEL chain mapping prompt variables to structured
                node selection results.
        """
        if self._node_selector_llm is None:
            self._node_selector_llm = self.provider.get_structured_model(
                schema=NodeSelectionOutput,
                temperature=0.0,
                num_ctx=8192,
                num_predict=512,
                method="json_schema",
                include_raw=True,
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("human", "User input: {user_message}"),
        ])
        return prompt | self._node_selector_llm

    def get_signal_resolver_runner(self, system_prompt: str) -> RunnableSerializable[dict[str, Any], Any]:
        """Build and cache the LangChain runnable pipeline for signal resolution.

        Initializes a structured-output LLM configured with the `SignalResolutionOutput`
        JSON schema and binds it to a two-part chat prompt template accepting `user_message`
        and `selected_nodes`.

        Args:
            system_prompt: System instructions detailing polarity extraction and evidence alignment.

        Returns:
            RunnableSerializable: Composed LCEL chain resolving candidate nodes into verified
                active signals with evidence spans and polarities.
        """
        if self._signal_resolver_llm is None:
            self._signal_resolver_llm = self.provider.get_structured_model(
                schema=SignalResolutionOutput,
                temperature=0.0,
                num_ctx=8192,
                num_predict=1024,
                method="json_schema",
                include_raw=True,
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            (
                "human",
                "=== USER MESSAGE ===\n{user_message}\n\n"
                "=== SELECTED NODES ===\n{selected_nodes}",
            ),
        ])
        return prompt | self._signal_resolver_llm

    def get_polarity_evidence_runner(
        self,
        system_prompt: str,
    ) -> RunnableSerializable[dict[str, Any], Any]:
        """Build and cache the LangChain runnable pipeline for polarity evidence index resolution."""
        if self._polarity_evidence_llm is None:
            self._polarity_evidence_llm = self.provider.get_structured_model(
                schema=PolarityEvidenceIndexOutput,
                temperature=0.0,
                num_ctx=8192,
                num_predict=256,
                method="json_schema",
                include_raw=True,
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            (
                "human",
                "=== USER MESSAGE ===\n{user_message}\n\n"
                "=== TARGET NODE ===\n{selected_nodes}\n\n"
                "=== REQUESTED POLARITY ===\n{polarity}\n\n"
                "=== EXCLUDED INDEX ===\n{excluded_index}\n\n"
                "=== EVIDENCE CANDIDATES ===\n{evidence_candidates}",
            ),
        ])

        return prompt | self._polarity_evidence_llm

    def get_safety_runner(self, system_prompt: str) -> Any:
        """
        Creates or retrieves a cached runnable chain for zero-tolerance safety classification.
        Uses temperature 0.0 for deterministic, highly reproducible triage.

        Args:
            system_prompt (str): The system prompt instructions defining safety protocols.

        Returns:
            Any (Runnable): A LangChain runnable chain that outputs a structured 
                            `SafetyOutput` Pydantic object.
        """
        if self._safety_llm is None:
            logger.debug("[LLMEngine] Initializing cached safety LLM runner (temp=0.0)...")
            self._safety_llm = self.provider.get_structured_model(
                schema=SafetyOutput,
                temperature=0.0,
                method="json_schema",
                include_raw=True,
            )
            
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("human", "User input: {user_message}")
        ])
        return prompt | self._safety_llm

    def get_extractor_runner(self, system_prompt: str) -> Any:
        """
        Creates or retrieves a cached runnable chain for entity and signal extraction.
        Uses temperature 0.0 to ensure factual extraction without hallucination.

        Args:
            system_prompt (str): The system prompt instructions defining extraction targets.

        Returns:
            Any (Runnable): A LangChain runnable chain that outputs a structured 
                            `ExtractionOutput` Pydantic object.
        """
        if self._extractor_llm is None:
            logger.debug("[LLMEngine] Initializing cached extractor LLM runner (temp=0.0)...")
            self._extractor_llm = self.provider.get_structured_model(
                schema=ExtractionOutput,
                temperature=0.0,
                num_ctx=8192,
                num_predict=1024,
                method="json_schema",
                include_raw=True,
            )
            
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("human", "User input: {user_message}")
        ])
        return prompt | self._extractor_llm

    def get_assessor_runner(self, system_prompt: str) -> Any:
        """
        Creates or retrieves a cached runnable chain for psychological resilience assessment.
        Uses a low temperature (0.0) to maintain strict schema compliance and reproducibility.

        Args:
            system_prompt (str): The system prompt instructions for graph-grounded assessment.

        Returns:
            Any (Runnable): A LangChain runnable chain that outputs a structured 
                            `AssessmentOutput` Pydantic object.
        """
        if self._assessor_llm is None:
            logger.debug("[LLMEngine] Initializing cached assessor LLM runner (temp=0.0)...")
            self._assessor_llm = self.provider.get_structured_model(
                schema=AssessmentOutput,
                temperature=0.0,
                num_ctx=8192,
                num_predict=2048,
                method="json_schema",
                include_raw=True,
            )
            
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("human", "User message: {user_message}\n\nRetrieved Graph Knowledge:\n{subgraph_context}")
        ])
        return prompt | self._assessor_llm

    def get_conversational_llm(self) -> BaseChatModel:
        """
        Retrieves a cached BaseChatModel instance tuned for natural, empathetic conversational responses.
        Uses the temperature and model settings defined in the centralized environment configuration.

        Returns:
            BaseChatModel: An initialized LLM instance configured for conversational flow.
        """
        if self._conversational_llm is None:
            logger.debug("[LLMEngine] Initializing cached conversational LLM instance (temp=%s)...", self.conversational_temp)
            self._conversational_llm = self.provider.get_chat_model(
                temperature=self.conversational_temp,
                num_predict=-1,
                num_ctx=10240,
            )
            
        return self._conversational_llm
