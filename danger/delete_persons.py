#!/usr/bin/env python3
"""
delete_persons.py — hard-delete contaminated PersonIdentity rows and their tracks.

Deletes:
  - track_sessions (observations CASCADE)
  - person_face_embeddings / person_embeddings (CASCADE or explicit)
  - person_identities

Before delete, NULLs FKs that lack ON DELETE (events, billing_interactions,
storage_objects). Does NOT hard-delete events/billing rows — only clears
person/track links.

MinIO crops are left for the periodic sweep (no immediate remove_object).

Usage (dry-run default):
    PYTHONPATH=/gmr/gmr venv/bin/python danger/delete_persons.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/delete_persons.py --apply
    PYTHONPATH=/gmr/gmr venv/bin/python danger/delete_persons.py --ids UUID1 UUID2 --apply

Edit PERSON_IDS below, or pass --ids on the CLI (CLI overrides the list).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal

# ── Edit this list when reusing the script ──────────────────────────────────
PERSON_IDS: list[str] = [
    # "3ed0970c-edbf-4e48-ad37-66dc108e71cf",
    # "e1858517-41fa-40b7-9c21-797f92e0a0f8",
]


async def preview(db, pids: list[str]) -> list[str]:
    """Print person summary; return track_session ids linked to these persons."""
    print("\n=== PERSONS ===")
    for pid in pids:
        r = await db.execute(
            text(
                """
                SELECT pi.id::text, pi.gender, pi.visit_count,
                       pi.first_seen_at, pi.last_seen_at, pi.is_staff,
                       (SELECT COUNT(*) FROM track_sessions
                          WHERE person_identity_id = pi.id) AS tracks,
                       (SELECT COUNT(*) FROM person_face_embeddings
                          WHERE person_identity_id = pi.id) AS faces,
                       (SELECT COUNT(*) FROM person_embeddings
                          WHERE person_identity_id = pi.id) AS bodies
                FROM person_identities pi
                WHERE pi.id::text = :pid
                """
            ),
            {"pid": pid},
        )
        row = r.fetchone()
        if not row:
            print(f"  {pid}  NOT FOUND")
            continue
        print(
            f"  {row[0]}  g={row[1]} visits={row[2]} staff={row[5]} "
            f"tracks={row[6]} faces={row[7]} bodies={row[8]} "
            f"first={row[3]} last={row[4]}"
        )

    r = await db.execute(
        text(
            """
            SELECT id::text FROM track_sessions
            WHERE person_identity_id::text = ANY(:pids)
            """
        ),
        {"pids": pids},
    )
    track_ids = [row[0] for row in r.fetchall()]
    print(f"\nTrack sessions linked: {len(track_ids)}")
    return track_ids


async def delete_persons(pids: list[str], apply: bool) -> None:
    pids = [p.strip() for p in pids if p and p.strip()]
    if not pids:
        print("No person IDs provided. Edit PERSON_IDS or pass --ids.")
        sys.exit(1)

    print("=" * 80)
    print(f"  DELETE PERSONS  {'— APPLY' if apply else '— DRY RUN'}")
    print(f"  IDs ({len(pids)}):")
    for p in pids:
        print(f"    {p}")
    print("=" * 80)

    async with AsyncSessionLocal() as db:
        track_ids = await preview(db, pids)

        if not apply:
            print("\nDry run only. Pass --apply to execute.")
            return

        try:
            # 1) Break FKs that block track / person deletion
            if track_ids:
                for table, col in (
                    ("events", "track_session_id"),
                    ("billing_interactions", "track_session_id"),
                ):
                    res = await db.execute(
                        text(
                            f"UPDATE {table} SET {col} = NULL "
                            f"WHERE {col}::text = ANY(:tids)"
                        ),
                        {"tids": track_ids},
                    )
                    print(f"NULL {table}.{col}: {res.rowcount}")

            for table, col in (
                ("events", "person_identity_id"),
                ("billing_interactions", "person_identity_id"),
                ("storage_objects", "person_identity_id"),
                ("track_sessions", "person_identity_id"),
            ):
                res = await db.execute(
                    text(
                        f"UPDATE {table} SET {col} = NULL "
                        f"WHERE {col}::text = ANY(:pids)"
                    ),
                    {"pids": pids},
                )
                print(f"NULL {table}.{col}: {res.rowcount}")

            # 2) Delete tracks (observations CASCADE)
            if track_ids:
                res = await db.execute(
                    text("DELETE FROM track_sessions WHERE id::text = ANY(:tids)"),
                    {"tids": track_ids},
                )
                print(f"DELETE track_sessions: {res.rowcount}")

            res = await db.execute(
                text(
                    "DELETE FROM track_sessions "
                    "WHERE person_identity_id::text = ANY(:pids)"
                ),
                {"pids": pids},
            )
            if res.rowcount:
                print(f"DELETE leftover tracks: {res.rowcount}")

            # 3) Explicit embedding delete (also CASCADE from person)
            res = await db.execute(
                text(
                    "DELETE FROM person_face_embeddings "
                    "WHERE person_identity_id::text = ANY(:pids)"
                ),
                {"pids": pids},
            )
            print(f"DELETE person_face_embeddings: {res.rowcount}")

            res = await db.execute(
                text(
                    "DELETE FROM person_embeddings "
                    "WHERE person_identity_id::text = ANY(:pids)"
                ),
                {"pids": pids},
            )
            print(f"DELETE person_embeddings: {res.rowcount}")

            # 4) Delete person rows
            res = await db.execute(
                text("DELETE FROM person_identities WHERE id::text = ANY(:pids)"),
                {"pids": pids},
            )
            print(f"DELETE person_identities: {res.rowcount}")

            await db.commit()
            print("\nCOMMITTED")
        except Exception as e:
            await db.rollback()
            print(f"\nFAILED — rolled back: {e}")
            raise

    # Verify
    async with AsyncSessionLocal() as db:
        r = await db.execute(
            text("SELECT id::text FROM person_identities WHERE id::text = ANY(:pids)"),
            {"pids": pids},
        )
        left = [row[0] for row in r.fetchall()]
        n_tracks = 0
        if track_ids:
            r2 = await db.execute(
                text("SELECT COUNT(*) FROM track_sessions WHERE id::text = ANY(:tids)"),
                {"tids": track_ids},
            )
            n_tracks = r2.scalar() or 0
        print(f"AFTER: persons remaining={len(left)} prior_tracks remaining={n_tracks}")
        if left:
            print("WARNING: these IDs still exist:")
            for p in left:
                print(f"  {p}")
            sys.exit(2)
        print("Done. MinIO crops left for periodic sweep.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hard-delete PersonIdentity rows + related tracks"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Person UUIDs (overrides PERSON_IDS in the script)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute deletes (default is dry-run)",
    )
    args = parser.parse_args()
    pids = args.ids if args.ids else list(PERSON_IDS)
    asyncio.run(delete_persons(pids, apply=args.apply))


if __name__ == "__main__":
    main()
