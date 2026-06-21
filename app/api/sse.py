from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Iterator

from app.api import runtime

log = logging.getLogger("app.sse")

_DONE = object()  # sentinel: graph thread finished, stop draining


def sse_event(event: str, data: dict | None = None) -> str:
    return f"event: {event}\ndata: {json.dumps(data or {}, default=str, ensure_ascii=False)}\n\n"


def run_graph_sse(graph, payload: Any, thread_id: str,
                  deps: Any = None) -> Iterator[str]:
    """Drive graph.stream in a background thread; yield SSE lines.

    `payload` is either an initial state dict or a langgraph Command (resume).
    """
    q: queue.Queue = queue.Queue()
    done_meta: dict[str, Any] = {}  # patient_id/document_id, attached to the done event

    def progress(msg: str) -> None:
        log.info("[sse %s] progress: %s", thread_id, msg)
        q.put(("progress", {"msg": msg}))

    def worker() -> None:
        cfg = runtime.cfg(thread_id, deps=deps, progress=progress)
        interrupted = False
        if isinstance(payload, dict):
            log.info("[sse %s] start: keys=%s file=%s patient=%s",
                     thread_id, sorted(payload.keys()),
                     payload.get("file_path"), payload.get("patient_id"))
        else:
            log.info("[sse %s] start: resume command", thread_id)
        t0 = time.monotonic()
        t_prev = t0
        try:
            for chunk in graph.stream(payload, cfg, stream_mode="updates"):
                for node in chunk:
                    now = time.monotonic()
                    if node == "__interrupt__":
                        interrupted = True
                        log.info("[sse %s] INTERRUPT after %.2fs", thread_id, now - t_prev)
                        q.put(("interrupt", chunk["__interrupt__"][0].value))
                        t_prev = now
                        continue
                    log.info("[sse %s] node done: %s (+%.2fs, total %.2fs)",
                             thread_id, node, now - t_prev, now - t0)
                    t_prev = now
                    q.put(("node", {"label": runtime.NODE_LABELS.get(node, f"… {node}")}))
            if not interrupted:
                snap = graph.get_state(cfg)
                # Surface the resolved patient/doc so the UI can refresh its cohort
                # and select the patient ingestion just created.
                pid = snap.values.get("patient_id")
                if pid is not None:
                    done_meta["patient_id"] = pid
                did = snap.values.get("document_id")
                if did is not None:
                    done_meta["document_id"] = did
                messages = snap.values.get("messages", [])
                last = messages[-1] if messages else None
                if last is not None:
                    q.put(("message", {
                        "role": last.get("role", "assistant"),
                        "content": last.get("content", ""),
                        "sources": last.get("sources"),
                        "subgraph": last.get("subgraph"),
                    }))
        except Exception as e:  # noqa: BLE001 - surface to the client as an error event
            log.exception("[sse %s] graph error after %.2fs", thread_id, time.monotonic() - t0)
            q.put(("error", {"message": "Something went wrong. Please try again."}))
        finally:
            log.info("[sse %s] stream end (total %.2fs)", thread_id, time.monotonic() - t0)
            q.put((_DONE, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, data = q.get()
        if kind is _DONE:
            yield sse_event("done", done_meta)
            return
        yield sse_event(kind, data)
