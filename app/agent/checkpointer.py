from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def pg_conninfo() -> str:
    """Plain libpq conninfo from settings.database_url.

    Strips the SQLAlchemy '+psycopg' dialect tag so psycopg/langgraph accept
    it: 'postgresql+psycopg://...' -> 'postgresql://...'.
    """
    url = get_settings().database_url
    return url.replace("postgresql+psycopg://", "postgresql://", 1)
