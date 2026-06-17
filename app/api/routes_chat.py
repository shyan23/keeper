from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

import app.storage as storage
from app.api import runtime, sse

router = APIRouter(prefix="/api/chat")

_ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "pdf", "txt"}


def _ext(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower() or "bin"


class StreamIn(BaseModel):
    thread_id: str
    message: str | None = None
    patient_id: int | None = None
    staged_path: str | None = None
    mime: str | None = None
    ext: str | None = None
    original_name: str | None = None


class ResumeIn(BaseModel):
    thread_id: str
    resume: dict


@router.post("/upload")
def upload(file: UploadFile):
    ext = _ext(file.filename or "")
    if ext not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {ext}")
    data = file.file.read()
    staged = storage.save_staging(ext, data)
    mime = file.content_type or (
        "application/pdf" if ext == "pdf"
        else "text/plain" if ext == "txt" else f"image/{ext}")
    return {"staged_path": staged, "mime": mime, "ext": ext}


def _sse_response(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.post("/stream")
def stream(body: StreamIn):
    messages = [{"role": "user", "content": body.message or "Read this and arrange it."}]

    if body.staged_path:
        payload = {
            "messages": messages,
            "file_path": body.staged_path,
            "mime_type": body.mime,
            "file_ext": body.ext,
            "source_type": "pdf" if body.ext == "pdf" else "image",
            "original_name": body.original_name,
        }
    else:
        if body.patient_id is None:
            def err():
                yield sse.sse_event("error", {
                    "message": "Pick a patient to ask about their records, "
                               "or attach a document for me to read."})
                yield sse.sse_event("done", {})
            return _sse_response(err())
        payload = {"messages": messages, "patient_id": body.patient_id}

    graph = runtime.get_graph()
    return _sse_response(
        sse.run_graph_sse(graph, payload, body.thread_id, deps=runtime.get_deps()))


@router.post("/resume")
def resume(body: ResumeIn):
    graph = runtime.get_graph()
    return _sse_response(
        sse.run_graph_sse(graph, Command(resume=body.resume), body.thread_id,
                          deps=runtime.get_deps()))
