#!/usr/bin/env python3
"""
reset_tracking_data.py
----------------------
Wipes all tracking, person identity, embedding, event, analytics and
billing data from the database AND deletes all crop/snapshot files
from MinIO, while preserving configuration tables (users, roles,
cameras, zones, stores, rules, areas, feature_requests).

Usage:
    .venv/bin/python reset_tracking_data.py [--yes]

    --yes   Skip the confirmation prompt (for scripted use).
"""
import asyncio
import sys
import argparse

from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX


# Tables to clear — in FK-safe order (children before parents)
TABLES_IN_ORDER = [
    "track_observations",           # → track_sessions
    "billing_interactions",         # → track_sessions, person_identities
    "events",                       # → track_sessions, person_identities
    "track_sessions",               # → person_identities
    "person_embeddings",            # → person_identities
    "person_face_embeddings",       # → person_identities
    "person_identities",            # root person table
    "daily_analytics_summary",      # standalone analytics
    "storage_objects",              # file references
]

# MinIO object name prefixes to wipe
MINIO_PREFIXES = ["crops/", "snapshots/"]


def wipe_minio():
    """Delete all objects under the crop and snapshot prefixes in MinIO."""
    client = get_client()
    total = 0
    for prefix in MINIO_PREFIXES:
        objects = client.list_objects(BUCKET_PREFIX, prefix=prefix, recursive=True)
        batch = [obj.object_name for obj in objects]
        for name in batch:
            try:
                client.remove_object(BUCKET_PREFIX, name)
                total += 1
            except Exception as e:
                print(f"   ⚠️  Could not delete {name}: {e}")
        print(f"   ✓ minio/{BUCKET_PREFIX}/{prefix}: {len(batch)} objects deleted")
    return total


async def reset():
    parser = argparse.ArgumentParser(description="Wipe all tracking/person data + MinIO crops.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    if not args.yes:
        print("\n⚠️  WARNING: This will permanently delete:")
        print("\n  PostgreSQL tables:")
        for t in TABLES_IN_ORDER:
            print(f"    • {t}")
        print("\n  MinIO objects (all files under):")
        for p in MINIO_PREFIXES:
            print(f"    • {BUCKET_PREFIX}/{p}")
        print("\nConfiguration tables (users, cameras, stores, zones, rules, etc.) will NOT be touched.")
        confirm = input("\nType 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    # ── Database ──────────────────────────────────────────────────────────────
    print("\n🗑️  Clearing database tables...\n")
    async with AsyncSessionLocal() as db:
        for table in TABLES_IN_ORDER:
            result = await db.execute(text(f"DELETE FROM {table}"))
            print(f"   ✓ {table}: {result.rowcount} rows deleted")
        await db.commit()

    # ── MinIO ─────────────────────────────────────────────────────────────────
    print("\n🗑️  Clearing MinIO objects...\n")
    try:
        total = wipe_minio()
        print(f"\n   Total MinIO objects deleted: {total}")
    except Exception as e:
        print(f"\n   ⚠️  MinIO cleanup failed: {e}")
        print("   Database was already cleared. Clean MinIO manually if needed.")

    print("\n✅ Done. All tracking and person data has been cleared.\n")


if __name__ == "__main__":
    asyncio.run(reset())
