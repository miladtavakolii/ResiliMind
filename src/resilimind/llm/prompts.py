from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Base path for prompt files
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "prompts"

def load_prompt_text(filename: str) -> str:
    """
    Reads and returns the content of a prompt text file.

    Args:
        filename (str): The name of the text file in the prompts directory.

    Returns:
        str: Raw prompt text string.
    """
    file_path = PROMPTS_DIR / filename
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {file_path}")

# System Prompt Strings loaded from files for Extractor, Assessor, and Safety modules
EXTRACTOR_SYSTEM_PROMPT: str = load_prompt_text("extractor.txt")
ASSESSOR_SYSTEM_PROMPT: str = load_prompt_text("assessor.txt")
SAFETY_CLASSIFIER_PROMPT: str = load_prompt_text("safety_classifier.txt")
EMERGENCY_RESPONSE_TEMPLATE: str = load_prompt_text("emergency_response.txt")

def get_questioner_prompt() -> ChatPromptTemplate:
    """
    Constructs the ChatPromptTemplate for the Questioner Agent using file context.
    """
    system_text = load_prompt_text("questioner.txt")
    return ChatPromptTemplate.from_messages([
        ("system", f"{system_text}\n\n=== GRAPH CONTEXT ===\n{{subgraph_context}}"),
        MessagesPlaceholder(variable_name="messages")
    ])

def get_advisor_prompt() -> ChatPromptTemplate:
    """
    Constructs the ChatPromptTemplate for the Advisor Agent using file context.
    """
    system_text = load_prompt_text("advisor.txt")
    return ChatPromptTemplate.from_messages([
        ("system", f"{system_text}\n\n{{subgraph_context}}\n\n=== ASSESSMENTS ===\n{{assessments}}"),
        MessagesPlaceholder(variable_name="messages")
    ])
