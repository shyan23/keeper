from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterator

from app.api import runtime

_DONE = object()  # sentinel: graph thread finished, stop draining


def sse_event(event: str, data: dict | None = None) -> str:
    return f"event: {event}\ndata: {json.dumps(data or {}, default=str, ensure_ascii=False)}\n\n"


def run_graph_sse(graph, payload: Any, thread_id: str,
                  deps: Any = None) -> Iterator[str]:
    """Drive graph.stream in a background thread; yield SSE lines.

    `payload` is either an initial state dict or a langgraph Command (resume).
    """
    q: queue.Queue = queue.Queue()

    def progress(msg: str) -> None:
        q.put(("progress", {"msg": msg}))

    def worker() -> None:
        cfg = runtime.cfg(thread_id, deps=deps, progress=progress)
        interrupted = False
        try:
            for chunk in graph.stream(payload, cfg, stream_mode="updates"):
                for node in chunk:
                    if node == "__interrupt__":
                        interrupted = True
                        q.put(("interrupt", chunk["__interrupt__"][0].value))
                        continue
                    q.put(("node", {"label": runtime.NODE_LABELS.get(node, f"… {node}")}))
            if not interrupted:
                snap = graph.get_state(cfg)
                messages = snap.values.get("messages", [])
                last = messages[-1] if messages else None
                if last is not None:
                    q.put(("message", {
                        "role": last.get("role", "assistant"),
                        "content": last.get("content", ""),
                        "sources": last.get("sources"),
                    }))
        except Exception as e:  # noqa: BLE001 - surface to the client as an error event
            q.put(("error", {"message": str(e)}))
        finally:
            q.put((_DONE, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, data = q.get()
        if kind is _DONE:
            yield sse_event("done", {})
            return
        yield sse_event(kind, data)
