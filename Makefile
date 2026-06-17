.PHONY: run ui dev

# Backend API (single worker: in-process agent checkpointer).
run:
	uvicorn app.api.server:app --port 8000 --workers 1

# Frontend dev server (vanilla TS, Vite).
ui:
	cd medagentic-dashboard && npm run dev

# Both at once (backend backgrounded; Ctrl-C stops the UI).
dev:
	uvicorn app.api.server:app --port 8000 --workers 1 & \
	cd medagentic-dashboard && npm run dev
