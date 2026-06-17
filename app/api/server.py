from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure app.* INFO logs (e.g. per-node SSE timing in app.api.sse) reach the
# console. uvicorn only configures its own loggers, not the root logger.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from app.api.routes_browse import router as browse_router
from app.api.routes_chat import router as chat_router

app = FastAPI(title="MedAgentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: Vite on :3000; no credentials used
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(browse_router)
app.include_router(chat_router)
