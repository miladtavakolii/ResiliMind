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

# System Prompt Strings loaded from files for Extractor and Assessor
EXTRACTOR_SYSTEM_PROMPT: str = load_prompt_text("extractor.txt")
ASSESSOR_SYSTEM_PROMPT: str = load_prompt_text("assessor.txt")

def get_questioner_prompt() -> ChatPromptTemplate:
    """
    Constructs the ChatPromptTemplate for the Questioner Agent using file context.
    """
    system_text = load_prompt_text("questioner.txt")
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "User message: {user_message}\n\nContext:\n{subgraph_context}")
    ])


def get_advisor_prompt() -> ChatPromptTemplate:
    """
    Constructs the ChatPromptTemplate for the Advisor Agent using file context.
    """
    system_text = load_prompt_text("advisor.txt")
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "User message: {user_message}\n\nAssessed Status:\n{assessments}\n\nGraph Interventions:\n{subgraph_context}")
    ])
