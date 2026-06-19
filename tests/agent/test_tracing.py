from app.config import Settings


def test_langfuse_settings_default_off(monkeypatch):
    # Construct Settings ignoring any real .env so defaults are deterministic.
    monkeypatch.setattr(Settings, "model_config",
                        {"env_file": None, "extra": "ignore"}, raising=True)
    s = Settings(database_url="postgresql://x/y")
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "http://localhost:3000"


from app.agent import tracing
from app.config import get_settings


def test_tracing_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr(get_settings(), "langfuse_public_key", None, raising=True)
    monkeypatch.setattr(get_settings(), "langfuse_secret_key", None, raising=True)
    assert tracing.tracing_enabled() is False
    assert tracing.get_handler("thread-1") is None


def test_tracing_enabled_when_both_keys_set(monkeypatch):
    monkeypatch.setattr(get_settings(), "langfuse_public_key", "pk-test", raising=True)
    monkeypatch.setattr(get_settings(), "langfuse_secret_key", "sk-test", raising=True)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    assert tracing.tracing_enabled() is True
    handler = tracing.get_handler("thread-1")
    assert handler is not None


from app.api import runtime


def test_cfg_omits_callbacks_when_disabled(monkeypatch):
    monkeypatch.setattr(runtime, "get_handler", lambda session_id=None: None)
    c = runtime.cfg("thread-1", deps=object())
    assert "callbacks" not in c


def test_cfg_attaches_callbacks_when_enabled(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runtime, "get_handler", lambda session_id=None: sentinel)
    c = runtime.cfg("thread-1", deps=object())
    assert c["callbacks"] == [sentinel]
