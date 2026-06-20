from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter

from app.agent.tracing import tracing_enabled
from app.config import get_settings

router = APIRouter(prefix="/api/tracer")
log = logging.getLogger(__name__)

_TIMEOUT = 5.0


def _auth_header() -> str:
    s = get_settings()
    token = base64.b64encode(
        f"{s.langfuse_public_key}:{s.langfuse_secret_key}".encode()
    ).decode()
    return f"Basic {token}"


def _langfuse_url(path: str) -> str:
    return f"{get_settings().langfuse_host}{path}"


def _extract_question(trace_input: dict | None) -> str:
    """Pull the user's message out of the LangGraph state dict."""
    if not isinstance(trace_input, dict):
        return "Unknown"
    msgs = trace_input.get("messages", [])
    if msgs and isinstance(msgs[-1], dict):
        content = msgs[-1].get("content", "")
        if content and content != "Read this and arrange it.":
            return str(content)[:200]
    if "file_path" in trace_input or trace_input.get("original_name"):
        name = trace_input.get("original_name") or "a file"
        return f"Uploaded: {name}"
    return "Uploaded a document"


def _kind(trace_input: dict | None) -> str:
    if not isinstance(trace_input, dict):
        return "question"
    if "file_path" in trace_input:
        return "upload"
    return "question"


def _speed_label(latency_ms: float | None) -> str:
    if latency_ms is None:
        return "unknown"
    s = latency_ms / 1000
    if s < 3:
        return "fast"
    if s < 8:
        return "normal"
    return "slow"


def _fmt_tokens(n: int | None) -> str | None:
    if n is None or n == 0:
        return None
    if n >= 1000:
        return f"{n // 1000},{n % 1000:03d}"
    return str(n)


def _rel_day(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = (now.date() - dt.date()).days
        if delta == 0:
            return "Today"
        if delta == 1:
            return "Yesterday"
        return dt.strftime("%b %d")
    except Exception:
        return "Earlier"


@router.get("/activity")
def get_activity(limit: int = 50):
    if not tracing_enabled():
        return {
            "enabled": False,
            "conversations": [],
            "stats": {"total": 0, "avg_duration_s": None, "total_tokens": 0},
        }

    try:
        resp = httpx.get(
            _langfuse_url(f"/api/public/traces?limit={limit}&orderBy=timestamp.desc"),
            headers={"Authorization": _auth_header()},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
    except Exception as exc:
        log.warning("Langfuse fetch failed: %s", exc)
        return {
            "enabled": True,
            "error": "Could not reach Langfuse. Is the stack running?",
            "conversations": [],
            "stats": {"total": 0, "avg_duration_s": None, "total_tokens": 0},
        }

    conversations = []
    total_ms = 0
    ms_count = 0
    total_tokens = 0

    for t in raw:
        latency_ms: float | None = t.get("latency")
        tokens: int | None = t.get("totalTokens") or None
        tip = t.get("input")
        ts = t.get("timestamp", "")

        if latency_ms:
            total_ms += latency_ms
            ms_count += 1
        if tokens:
            total_tokens += tokens

        conversations.append({
            "id": t.get("id"),
            "session": t.get("sessionId"),
            "question": _extract_question(tip),
            "kind": _kind(tip),
            "duration_s": round(latency_ms / 1000, 1) if latency_ms else None,
            "speed": _speed_label(latency_ms),
            "tokens": _fmt_tokens(tokens),
            "timestamp": ts,
            "day": _rel_day(ts),
            "time": _fmt_time(ts),
        })

    avg = round(total_ms / ms_count / 1000, 1) if ms_count else None

    return {
        "enabled": True,
        "conversations": conversations,
        "stats": {
            "total": len(conversations),
            "avg_duration_s": avg,
            "total_tokens": _fmt_tokens(total_tokens),
        },
    }


def _fmt_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%-I:%M %p")
    except Exception:
        return ""
