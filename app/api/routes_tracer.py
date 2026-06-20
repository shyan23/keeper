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


# ---- Cost per 1M tokens (USD). Groq/Ollama = free. ----
# Gemini 2.5 Flash: $0.075 input / $0.30 output per 1M tokens.
_COST_PER_M: dict[str, dict[str, float]] = {
    "gemini-2.5-flash":        {"input": 0.075,  "output": 0.30},
    "gemini-2.0-flash":        {"input": 0.075,  "output": 0.30},
    "gemini-1.5-flash":        {"input": 0.075,  "output": 0.30},
}
_FREE_PREFIXES = ("llama", "qwen", "mixtral", "openai/gpt-oss", "moondream",
                  "nomic", "llava", "mistral", "deepseek")


def _model_cost_usd(model: str, input_tok: int, output_tok: int) -> float:
    name = (model or "").lower()
    for prefix in _FREE_PREFIXES:
        if prefix in name:
            return 0.0
    rates = _COST_PER_M.get(name)
    if rates is None:
        # Unknown model — fall back to Gemini Flash rates as a conservative estimate
        rates = {"input": 0.075, "output": 0.30}
    return (input_tok * rates["input"] + output_tok * rates["output"]) / 1_000_000


def _provider_label(model: str) -> str:
    name = (model or "").lower()
    if "gemini" in name:
        return "Gemini (paid)"
    if any(p in name for p in ("llama", "qwen", "mixtral", "openai/gpt-oss", "deepseek")):
        return "Groq (free)"
    if any(p in name for p in ("moondream", "llava", "nomic", "ollama")):
        return "Ollama (local)"
    return model or "Unknown"


@router.get("/cost")
def get_cost(limit: int = 100):
    if not tracing_enabled():
        return {"enabled": False, "models": [], "total_usd": "0.0000"}

    try:
        resp = httpx.get(
            _langfuse_url(f"/api/public/observations?type=GENERATION&limit={limit}"),
            headers={"Authorization": _auth_header()},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
    except Exception as exc:
        log.warning("Langfuse cost fetch failed: %s", exc)
        return {"enabled": True, "error": "Could not reach Langfuse.", "models": [], "total_usd": "0.0000"}

    # Aggregate per model
    agg: dict[str, dict] = {}
    for obs in raw:
        model = obs.get("model") or "unknown"
        usage = obs.get("usage") or {}
        inp = int(usage.get("input") or usage.get("promptTokens") or 0)
        out = int(usage.get("output") or usage.get("completionTokens") or 0)
        cost = _model_cost_usd(model, inp, out)
        if model not in agg:
            agg[model] = {"model": model, "label": _provider_label(model),
                          "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        agg[model]["calls"] += 1
        agg[model]["input_tokens"] += inp
        agg[model]["output_tokens"] += out
        agg[model]["cost_usd"] += cost

    models = sorted(agg.values(), key=lambda x: x["cost_usd"], reverse=True)
    total = sum(m["cost_usd"] for m in models)

    for m in models:
        m["cost_usd"] = f"{m['cost_usd']:.4f}"
        m["input_tokens"] = _fmt_tokens(m["input_tokens"])
        m["output_tokens"] = _fmt_tokens(m["output_tokens"])

    return {
        "enabled": True,
        "models": models,
        "total_usd": f"{total:.4f}",
    }
