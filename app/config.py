from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    test_database_url: str | None = None
    # AI chat provider:
    #   "groq"     — Groq primary (fast, ~3-4s structured), Ollama fallback. DEFAULT.
    #   "ollama"   — local only, no cloud (slow structured extraction on CPU).
    #   "fallback" — Gemini -> Groq -> Ollama.
    # OCR is always Tesseract; embeddings always Ollama (768-dim, pinned).
    ai_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Structured-output model: strict json_schema (constrained decoding) is only
    # available on Groq for the gpt-oss family — llama-3.3 does json_object only.
    # Free tier: 30 RPM / 8000 TPM. Used only by structured(); complete() stays on groq_model.
    groq_structured_model: str = "openai/gpt-oss-120b"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollama_embed_model: str = "nomic-embed-text"
    # Small vision model: llama3.2-vision needs ~11 GiB RAM; moondream fits in ~2.
    ollama_vision_model: str = "moondream"
    # MedGemma 4B multimodal (Ollama-served) for imaging/radiology narrative OCR.
    # Off by default: Gemma/HAI-DEF terms are field-restricted (not OSI/Apache), and
    # a 4B model is slow per page. When enabled it joins the FallbackVision chain.
    medgemma_enabled: bool = False
    medgemma_model: str = "alibayram/medgemma"
    rag_top_k: int = 5
    rag_confidence_threshold: float = 0.5
    gemini_api_key: str = "changeme"
    gemini_model: str = "gemini-2.5-flash"
    gemini_vision_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "models/text-embedding-004"
    redis_url: str = "redis://localhost:6379/0"
    storage_dir: str = "./data/files"
    app_version: str = "0.1.0"
    # Self-hosted Langfuse tracing (opt-in). Tracing is OFF unless BOTH keys are
    # set. host MUST stay local — never cloud.langfuse.com (medical prompts).
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
