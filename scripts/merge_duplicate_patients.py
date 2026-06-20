#!/usr/bin/env python3
"""
One-shot script: merge all near-duplicate patient rows into one canonical row.
Groups patients by normalized name (honorifics stripped), keeps the oldest
(lowest id) as canonical, re-points all documents + chunks, deletes ghosts.

Usage:
  cd /home/shyan/Desktop/Code/Keeper
  .venv/bin/python scripts/merge_duplicate_patients.py [--dry-run]
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import defaultdict

# Bootstrap the app so we get settings + models without duplicating config.
sys.path.insert(0, "/home/shyan/Desktop/Code/Keeper")

from sqlalchemy import create_engine, text
from app.config import get_settings

_TITLE_TOKENS = {
    "mr", "mrs", "ms", "miss", "mst", "mds", "master", "md", "dr", "prof",
    "professor", "mister", "sir", "madam", "smt", "begum",
    "mr.", "mrs.", "dr.", "mst.", "mds.",
}


def _normalize(name: str | None) -> str:
    n = re.sub(r"[.,]", " ", (name or "").lower())
    toks = [t for t in n.split() if t and t not in _TITLE_TOKENS]
    return " ".join(toks)


def main(dry_run: bool) -> None:
    url = get_settings().database_url
    engine = create_engine(url)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, name, age, gender FROM patient ORDER BY id")
        ).fetchall()

    # Group by normalized name
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[_normalize(r.name)].append(r)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    if not dupes:
        print("No duplicates found.")
        return

    for norm, patients in dupes.items():
        canonical = patients[0]  # lowest id = oldest
        ghosts = patients[1:]
        ghost_ids = [g.id for g in ghosts]
        print(f"\nGroup '{norm}'")
        print(f"  canonical: id={canonical.id}  name={canonical.name!r}")
        for g in ghosts:
            print(f"  ghost:     id={g.id}  name={g.name!r}")

        if dry_run:
            continue

        with engine.begin() as conn:
            # Merge demographics: fill nulls on canonical from ghosts
            age = canonical.age
            gender = canonical.gender
            for g in ghosts:
                if age is None and g.age:
                    age = g.age
                if gender is None and g.gender:
                    gender = g.gender
            conn.execute(
                text("UPDATE patient SET age=:a, gender=:g WHERE id=:id"),
                {"a": age, "g": gender, "id": canonical.id},
            )

            # Re-point documents
            conn.execute(
                text("UPDATE document SET patient_id=:cid WHERE patient_id = ANY(:gids)"),
                {"cid": canonical.id, "gids": ghost_ids},
            )
            # Re-point chunks (denormalized patient_id)
            conn.execute(
                text("UPDATE chunk SET patient_id=:cid WHERE patient_id = ANY(:gids)"),
                {"cid": canonical.id, "gids": ghost_ids},
            )
            # Delete ghosts (no documents left, CASCADE is a no-op now)
            conn.execute(
                text("DELETE FROM patient WHERE id = ANY(:gids)"),
                {"gids": ghost_ids},
            )
            print(f"  → merged {len(ghosts)} ghost(s) into id={canonical.id}, deleted ghost ids={ghost_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("=== DRY RUN — no changes written ===")
    main(args.dry_run)
