from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    test_database_url: str | None = None
    # AI provider: "groq" (cloud, free) or "ollama" (local). Default groq.
    ai_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    gemini_api_key: str = "changeme"
    storage_dir: str = "./data/files"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
