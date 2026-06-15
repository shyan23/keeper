from app.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key123")
    monkeypatch.setenv("STORAGE_DIR", "/tmp/files")
    s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@h:5432/db"
    assert s.gemini_api_key == "key123"
    assert s.storage_dir == "/tmp/files"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.delenv("STORAGE_DIR", raising=False)
    s = Settings()
    assert s.storage_dir == "./data/files"
    assert s.app_version == "0.1.0"


def test_agent_config_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x")
    from app.config import Settings
    s = Settings()
    assert s.groq_vision_model == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert s.ollama_embed_model == "nomic-embed-text"
    assert s.rag_top_k == 5
    assert s.rag_confidence_threshold == 0.5
