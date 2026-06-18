"""Deterministic, zero-cost evaluation harness for the medical-doc agent.

No LLM-as-judge: every metric is scored by code (exact / numeric / set-overlap),
so a run is reproducible, free, and unbiased. The model under test (the free
Groq/Ollama chat for extraction; the local Ollama embedder for retrieval) GENERATES;
this package only SCORES. See `eval/README.md`.
"""
