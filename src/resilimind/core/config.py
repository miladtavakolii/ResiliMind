import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from typing_extensions import Self

# Initialize module logger
logger = logging.getLogger(__name__)

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
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: Path = Path("resilimind.log")
    
    # Centralized Storage Directory (Defaults to 'data' in the current working directory)
    DATA_DIR: Path = Path("data")

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def _log_config_initialization(self) -> Self:
        """Logs configuration details at INFO level upon successful initialization."""
        logger.info(f"[Config] AppConfig successfully loaded with model '{self.RESILIMIND_LLM_MODEL}' and data directory '{self.DATA_DIR}'.")
        return self

    @property
    def resolved_log_file_path(self) -> Path:
        """
        Dynamically computes the absolute path for the log file, ensuring parent directories exist.
        """
        path: Path = self.LOG_FILE_PATH
        if not path.is_absolute():
            path = (self.DATA_DIR / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def user_db_path(self) -> Path:
        """
        Dynamically computes the absolute path for the main user SQLite database.
        
        Returns:
            Path: The resolved path to 'resilimind.db'.
        """
        path: Path = (self.DATA_DIR / "resilimind.db").resolve()
        logger.debug(f"[Config] Resolved user database path: {path}")
        return path

    @property
    def checkpoint_db_path(self) -> Path:
        """
        Dynamically computes the absolute path for the LangGraph persistent checkpoint database.
        
        Returns:
            Path: The resolved path to 'checkpoints.db'.
        """
        path: Path = (self.DATA_DIR / "checkpoints.db").resolve()
        logger.debug(f"[Config] Resolved checkpoint database path: {path}")
        return path

# Singleton instance of the configuration
settings: AppConfig = AppConfig()
