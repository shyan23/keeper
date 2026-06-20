.PHONY: run ui dev tracing tracing-down eval eval-retrieval mcp

# MCP server (stdio) — exposes records/search as tools for any MCP client.
mcp:
	python -m app.mcp.server

# Backend API (single worker: in-process agent checkpointer; --reload picks up edits).
run:
	uvicorn app.api.server:app --port 8000 --workers 1 --reload

# Frontend dev server (vanilla TS, Vite).
ui:
	cd medagentic-dashboard && npm run dev

# Self-hosted Langfuse v2 tracing stack (Langfuse + its own Postgres).
# After first start: open http://localhost:3000, create a project, copy keys into .env.
tracing:
	docker compose -f docker-compose.langfuse.yml up -d

# Tear down the Langfuse stack (data volume is preserved).
tracing-down:
	docker compose -f docker-compose.langfuse.yml down

# Full dev environment: Langfuse + backend + frontend.
# First run: `make tracing` then set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env.
# Tracing is a no-op when keys are blank — safe to run without them.
dev:
	$(MAKE) tracing
	uvicorn app.api.server:app --port 8000 --workers 1 --reload & \
	cd medagentic-dashboard && npm run dev

# Deterministic eval (extraction quality). Uses the free chat model, no DB.
eval:
	python -m app.eval

# Also run retrieval recall@k (needs TEST_DATABASE_URL + Ollama embedder).
eval-retrieval:
	python -m app.eval --retrieval
