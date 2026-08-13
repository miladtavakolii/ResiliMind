from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """
    Centralized configuration management for the ResiliMind application.
    Automatically loads variables from the environment or a .env file.

    Attributes:
        RESILIMIND_LLM_MODEL (str): The specific LLM model name to use via Ollama.
        RESILIMIND_LLM_TEMPERATURE (float): The temperature setting for conversational LLM responses.
        OLLAMA_BASE_URL (str): The base URL for the Ollama API connection.
        DB_PATH_DIR (str): The directory path for storing the local SQLite database.
        model_config (SettingsConfigDict): Pydantic settings configuration object.
    """
    RESILIMIND_LLM_MODEL: str = "gemma4:e2b"
    RESILIMIND_LLM_TEMPERATURE: float = 0.6
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DB_PATH_DIR: str = "data"

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton instance of the configuration
settings: AppConfig = AppConfig()
