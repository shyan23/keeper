.PHONY: run ui dev eval eval-retrieval

# Backend API (single worker: in-process agent checkpointer; --reload picks up edits).
run:
	uvicorn app.api.server:app --port 8000 --workers 1 --reload

# Frontend dev server (vanilla TS, Vite).
ui:
	cd medagentic-dashboard && npm run dev

# Both at once (backend backgrounded; Ctrl-C stops the UI).
dev:
	uvicorn app.api.server:app --port 8000 --workers 1 --reload & \
	cd medagentic-dashboard && npm run dev

# Deterministic eval (extraction quality). Uses the free chat model, no DB.
eval:
	python -m app.eval

# Also run retrieval recall@k (needs TEST_DATABASE_URL + Ollama embedder).
eval-retrieval:
	python -m app.eval --retrieval
