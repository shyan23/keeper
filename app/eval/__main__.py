"""CLI: `python -m app.eval [--retrieval]`.

Runs the deterministic scoring suites and prints a scorecard. Exit code is 0 if the
run completed (a low score is still a successful run — gating thresholds belong in CI,
not here, so the number is always visible).
"""
from __future__ import annotations

import argparse

from app.eval.harness import run, format_scorecard


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m app.eval")
    ap.add_argument("--retrieval", action="store_true",
                    help="also run the retrieval suite (needs TEST_DATABASE_URL + Ollama embedder)")
    args = ap.parse_args()
    out = run(with_retrieval=args.retrieval)
    print(format_scorecard(out))


if __name__ == "__main__":
    main()
