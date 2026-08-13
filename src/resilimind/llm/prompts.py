import logging
import importlib.resources as pkg_resources
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize module logger
logger = logging.getLogger(__name__)

def load_prompt_text(filename: str) -> str:
    """
    Reads and returns the content of a prompt text file.

    Args:
        filename (str): The name of the text file in the prompts directory.

    Returns:
        str: Raw prompt text string.
    """
    try:
        logger.debug(f"[Prompts] Attempting to load prompt file: {filename}")
        # Load directly from the packaged resources (PEP 592/302 compatible)
        prompt_content = pkg_resources.files("resilimind.assets.prompts").joinpath(filename).read_text(encoding="utf-8")
        logger.debug(f"[Prompts] Successfully loaded prompt '{filename}' ({len(prompt_content)} chars).")
        return prompt_content.strip()
    except Exception as e:
        logger.exception(f"[Prompts] Failed to load prompt '{filename}' from package resources: {e}")
        raise FileNotFoundError(f"Failed to load prompt '{filename}' from package resources: {e}")
    
# System Prompt Strings loaded from files
EXTRACTOR_SYSTEM_PROMPT: str = load_prompt_text("extractor.txt")
ASSESSOR_SYSTEM_PROMPT: str = load_prompt_text("assessor.txt")
SAFETY_CLASSIFIER_PROMPT: str = load_prompt_text("safety_classifier.txt")
EMERGENCY_RESPONSE_TEMPLATE: str = load_prompt_text("emergency_response.txt")

def get_questioner_prompt() -> ChatPromptTemplate:
    """
    Constructs the ChatPromptTemplate for the Questioner Agent using file context.
    """
    logger.debug("[Prompts] Constructing Questioner ChatPromptTemplate.")
    system_text = load_prompt_text("questioner.txt")
    return ChatPromptTemplate.from_messages([
        ("system", f"{system_text}\n\n=== GRAPH CONTEXT ===\n{{subgraph_context}}"),
        MessagesPlaceholder(variable_name="messages")
    ])

def get_advisor_prompt() -> ChatPromptTemplate:
    """
    Constructs the ChatPromptTemplate for the Advisor Agent using file context.
    """
    logger.debug("[Prompts] Constructing Advisor ChatPromptTemplate.")
    system_text = load_prompt_text("advisor.txt")
    return ChatPromptTemplate.from_messages([
        ("system", f"{system_text}\n\n{{subgraph_context}}\n\n=== ASSESSMENTS ===\n{{assessments}}"),
        MessagesPlaceholder(variable_name="messages")
    ])
