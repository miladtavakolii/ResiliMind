from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.models import ExtractionOutput, AssessmentOutput


class LLMEngine:
    """
    Singleton wrapper class for interacting with the local Gemma model via Ollama.
    Handles structured outputs and conversational chains.
    """

    def __init__(self, model_name: str = "gemma4:e2b") -> None:
        """
        Initializes the LLMEngine with the specified Ollama model.

        Args:
            model_name (str): Name of the Ollama model instance (default: 'gemma4:e2b').
        """
        self.model_name: str = model_name

    def get_extractor_runner(self, system_prompt: str) -> Any:
        """
        Creates a runnable chain for extraction enforced with ExtractionOutput schema.

        Args:
            system_prompt (str): System prompt instructions for extraction.

        Returns:
            Any (Runnable): Execution chain outputting ExtractionOutput Pydantic object.
        """
        llm = ChatOllama(model=self.model_name, temperature=0.0)
        structured_llm = llm.with_structured_output(ExtractionOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User input: {user_message}")
        ])
        return prompt | structured_llm

    def get_assessor_runner(self, system_prompt: str) -> Any:
        """
        Creates a runnable chain for resilience assessment enforced with AssessmentOutput schema.

        Args:
            system_prompt (str): System prompt instructions for assessment.

        Returns:
            Any (Runnable): Execution chain outputting AssessmentOutput Pydantic object.
        """
        llm = ChatOllama(model=self.model_name, temperature=0.1)
        structured_llm = llm.with_structured_output(AssessmentOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User message: {user_message}\n\nRetrieved Graph Knowledge:\n{subgraph_context}")
        ])
        return prompt | structured_llm

    def get_conversational_llm(self) -> ChatOllama:
        """
        Returns a ChatOllama instance tuned for natural conversational responses.

        Returns:
            ChatOllama: LLM instance with temperature set for empathy and fluid advice.
        """
        return ChatOllama(model=self.model_name, temperature=0.6)
