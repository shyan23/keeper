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
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollama_embed_model: str = "nomic-embed-text"
    rag_top_k: int = 5
    rag_confidence_threshold: float = 0.5
    gemini_api_key: str = "changeme"
    gemini_model: str = "gemini-2.0-flash"
    gemini_vision_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "models/text-embedding-004"
    storage_dir: str = "./data/files"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
