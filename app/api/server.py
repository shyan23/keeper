from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_browse import router as browse_router

app = FastAPI(title="MedAgentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: Vite on :3000; no credentials used
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(browse_router)
