from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """
    Centralized configuration management for the ResiliMind application.
    Automatically loads variables from the environment or a .env file.

    Attributes:
        RESILIMIND_LLM_MODEL (str): The specific LLM model name to use via Ollama.
        RESILIMIND_LLM_TEMPERATURE (float): The temperature setting for conversational LLM responses.
        OLLAMA_BASE_URL (str): The base URL for the Ollama API connection.
        DATA_DIR (Path): The root directory for storing mutable runtime data (e.g., SQLite DBs).
        model_config (SettingsConfigDict): Pydantic settings configuration object.
    """
    # LLM Settings
    RESILIMIND_LLM_MODEL: str = "gemma4:e2b"
    RESILIMIND_LLM_TEMPERATURE: float = 0.6
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Centralized Storage Directory (Defaults to 'data' in the current working directory)
    DATA_DIR: Path = Path("data")

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def user_db_path(self) -> Path:
        """
        Dynamically computes the absolute path for the main user SQLite database.
        
        Returns:
            Path: The resolved path to 'resilimind.db'.
        """
        return (self.DATA_DIR / "resilimind.db").resolve()

    @property
    def checkpoint_db_path(self) -> Path:
        """
        Dynamically computes the absolute path for the LangGraph persistent checkpoint database.
        
        Returns:
            Path: The resolved path to 'checkpoints.db'.
        """
        return (self.DATA_DIR / "checkpoints.db").resolve()

# Singleton instance of the configuration
settings: AppConfig = AppConfig()
