from app.agent import tracing
from app.agent.llm import GroqChat
from app.agent.nodes.rag import generate_answer_node
from app.agent.providers import FallbackChat
from app.api import runtime
from app.config import Settings, get_settings


def test_langfuse_settings_default_off(monkeypatch):
    # Construct Settings ignoring any real .env so defaults are deterministic.
    monkeypatch.setattr(Settings, "model_config",
                        {"env_file": None, "extra": "ignore"}, raising=True)
    s = Settings(database_url="postgresql://x/y")
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "http://localhost:3000"


def test_tracing_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr(get_settings(), "langfuse_public_key", None, raising=True)
    monkeypatch.setattr(get_settings(), "langfuse_secret_key", None, raising=True)
    assert tracing.tracing_enabled() is False
    assert tracing.get_handler("thread-1") is None


def test_tracing_enabled_when_both_keys_set(monkeypatch):
    # Keys come from Settings (pydantic reads .env); no os.environ needed now
    # that get_handler passes them explicitly to CallbackHandler.
    monkeypatch.setattr(get_settings(), "langfuse_public_key", "pk-test", raising=True)
    monkeypatch.setattr(get_settings(), "langfuse_secret_key", "sk-test", raising=True)
    monkeypatch.setattr(get_settings(), "langfuse_host", "http://localhost:3000", raising=True)
    assert tracing.tracing_enabled() is True
    handler = tracing.get_handler("thread-1")
    assert handler is not None


def test_cfg_omits_callbacks_when_disabled(monkeypatch):
    monkeypatch.setattr(runtime, "get_handler", lambda session_id=None: None)
    c = runtime.cfg("thread-1", deps=object())
    assert "callbacks" not in c


def test_cfg_attaches_callbacks_when_enabled(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runtime, "get_handler", lambda session_id=None: sentinel)
    c = runtime.cfg("thread-1", deps=object())
    assert c["callbacks"] == [sentinel]


class _RecordingInner:
    """Stands in for a langchain chat client; records the config it was invoked with."""
    def __init__(self):
        self.seen_config = "UNSET"

    def invoke(self, prompt, config=None):
        self.seen_config = config
        class _R:
            content = "ok"
        return _R()


def test_groqchat_complete_forwards_config():
    inner = _RecordingInner()
    chat = GroqChat(inner=inner)
    chat.complete("hi", config={"callbacks": ["h"]})
    assert inner.seen_config == {"callbacks": ["h"]}


class _RecordingProvider:
    def __init__(self):
        self.seen_config = "UNSET"

    def complete(self, prompt, config=None):
        self.seen_config = config
        return "ok"


def test_fallbackchat_complete_forwards_config():
    p = _RecordingProvider()
    chat = FallbackChat([p])
    chat.complete("hi", config={"callbacks": ["h"]})
    assert p.seen_config == {"callbacks": ["h"]}


class _RecordingChat:
    def __init__(self):
        self.seen_config = "UNSET"

    def complete(self, prompt, config=None):
        self.seen_config = config
        return "an answer"


def test_generate_answer_node_forwards_config():
    chat = _RecordingChat()
    deps = type("Deps", (), {"chat": chat})()
    config = {"configurable": {"deps": deps}, "callbacks": ["h"]}
    state = {
        "messages": [{"role": "user", "content": "what is my RBC?"}],
        "retrieved": [{"text": "RBC 5.1", "chunk_id": 1, "doc_type": "lab",
                       "report_date": "2026-01-01"}],
    }
    generate_answer_node(state, config)
    assert chat.seen_config == config
