from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    """True only when both Langfuse keys are configured."""
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def get_handler(session_id: str | None = None):
    """A Langfuse langchain CallbackHandler, or None when tracing is disabled.

    session_id groups a conversation's turns under one Langfuse session. The
    handler reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from
    the environment (populated from .env).
    """
    if not tracing_enabled():
        return None
    try:
        from langfuse.callback import CallbackHandler
        s = get_settings()
        return CallbackHandler(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
            session_id=session_id,
        )
    except Exception:  # noqa: BLE001 - tracing must never break a request
        logger.warning("Langfuse handler unavailable; continuing untraced.",
                       exc_info=True)
        return None
