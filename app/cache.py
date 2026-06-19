"""Redis cache for expensive, deterministic work (OCR, embeddings, extraction).

All helpers degrade to a no-op if Redis is unreachable: a cache miss just runs
the real function, so a missing/broken Redis never breaks the app — it only
removes the speed-up. Values are JSON, keyed by a SHA-256 of the inputs.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
from typing import Any, Callable

from app.config import get_settings

log = logging.getLogger(__name__)

# 30 days: OCR/embeddings/extraction for identical input never change.
_DEFAULT_TTL = 60 * 60 * 24 * 365 * 5


@functools.lru_cache(maxsize=1)
def _client():
    """Connect once. Returns a live client, or None if Redis is unreachable."""
    try:
        import redis

        c = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
        c.ping()
        return c
    except Exception as e:  # noqa: BLE001 - any failure ⇒ caching disabled
        log.warning("Redis unavailable, caching disabled: %s", e)
        return None


def make_key(namespace: str, *parts: Any) -> str:
    """Stable key: namespace + SHA-256 over the parts (bytes hashed directly)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
    return f"{namespace}:{h.hexdigest()}"


def get_or_set(key: str, compute: Callable[[], Any], ttl: int = _DEFAULT_TTL) -> Any:
    """Return cached JSON for `key`, else run `compute()`, cache it, and return it."""
    c = _client()
    if c is None:
        return compute()
    try:
        hit = c.get(key)
        if hit is not None:
            return json.loads(hit)
    except Exception as e:  # noqa: BLE001 - read failure ⇒ just recompute
        log.warning("cache read failed (%s): %s", key, e)

    value = compute()
    try:
        c.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception as e:  # noqa: BLE001 - write failure ⇒ value still returned
        log.warning("cache write failed (%s): %s", key, e)
    return value
