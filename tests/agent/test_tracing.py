from app.config import Settings


def test_langfuse_settings_default_off(monkeypatch):
    # Construct Settings ignoring any real .env so defaults are deterministic.
    monkeypatch.setattr(Settings, "model_config",
                        {"env_file": None, "extra": "ignore"}, raising=True)
    s = Settings(database_url="postgresql://x/y")
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "http://localhost:3000"
