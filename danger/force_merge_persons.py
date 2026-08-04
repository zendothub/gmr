#!/usr/bin/env python3
"""
force_merge_persons.py — merge loser person_identity into winner (ops override).

Use when dedup is blocked (e.g. historical same-cam split of one person into two
IDs) but operators confirmed they are the same person.

Takes pg_advisory_xact_lock(1001). Contamination-gated absorb. Updates
last_seen_at / first_seen_at / visit_count. Deletes loser.

Usage:
  PYTHONPATH=/gmr/gmr venv/bin/python danger/force_merge_persons.py \\
    --winner 3fc4be2c-... --loser 69f37e21-...
  PYTHONPATH=/gmr/gmr venv/bin/python danger/force_merge_persons.py \\
    --winner ... --loser ... --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from loguru import logger
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.jobs.tasks import (
    _absorb_body_embeddings,
    _absorb_face_embeddings,
    _revote_person_gender,
)
from app.modules.reid.identity_decision_engine import IDENTITY_ADVISORY_LOCK_KEY
from app.config import get_settings


async def run(winner_id: str, loser_id: str, apply: bool) -> None:
    settings = get_settings()
    winner_id = str(uuid.UUID(winner_id))
    loser_id = str(uuid.UUID(loser_id))
    if winner_id == loser_id:
        raise SystemExit("winner and loser must differ")

    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                text("""
                    SELECT id::text, is_staff, gender, estimated_age, visit_count,
                           first_seen_at, last_seen_at, best_face_score, face_crop_path
                    FROM person_identities
                    WHERE id::text = ANY(:ids)
                """),
                {"ids": [winner_id, loser_id]},
            )
        ).mappings().all()
        by = {r["id"]: dict(r) for r in rows}
        if winner_id not in by:
            raise SystemExit(f"winner {winner_id} not found")
        if loser_id not in by:
            raise SystemExit(f"loser {loser_id} not found")

        w, l = by[winner_id], by[loser_id]
        n_ts = (
            await db.execute(
                text("""
                    SELECT person_identity_id::text, COUNT(*)::int
                    FROM track_sessions
                    WHERE person_identity_id::text = ANY(:ids)
                    GROUP BY 1
                """),
                {"ids": [winner_id, loser_id]},
            )
        ).fetchall()
        ts = {r[0]: r[1] for r in n_ts}

        print("=" * 72)
        print(f"  FORCE MERGE  {loser_id}  →  {winner_id}")
        print(
            f"  winner: staff={w['is_staff']} visits={w['visit_count']} "
            f"tracks={ts.get(winner_id, 0)} age={w['estimated_age']} g={w['gender']}"
        )
        print(
            f"  loser:  staff={l['is_staff']} visits={l['visit_count']} "
            f"tracks={ts.get(loser_id, 0)} age={l['estimated_age']} g={l['gender']}"
        )
        print("=" * 72)

        if not apply:
            print("  Dry run. Pass --apply to execute.")
            return

        await db.execute(
            text(f"SELECT pg_advisory_xact_lock({int(IDENTITY_ADVISORY_LOCK_KEY)})")
        )

        max_faces = settings.MAX_FACE_EMBEDDINGS_PER_PERSON
        max_bodies = 10

        async with db.begin_nested():
            for tbl, col in (
                ("track_sessions", "person_identity_id"),
                ("events", "person_identity_id"),
                ("billing_interactions", "person_identity_id"),
                ("storage_objects", "person_identity_id"),
            ):
                await db.execute(
                    text(
                        f"UPDATE {tbl} SET {col} = :winner WHERE {col}::text = :loser"
                    ),
                    {"winner": winner_id, "loser": loser_id},
                )

            update_parts = ["visit_count = visit_count + :extra"]
            params: dict = {"extra": int(l["visit_count"] or 0), "winner": winner_id}
            if l["first_seen_at"] and (
                w["first_seen_at"] is None or l["first_seen_at"] < w["first_seen_at"]
            ):
                update_parts.append("first_seen_at = :lf")
                params["lf"] = l["first_seen_at"]
            if l["last_seen_at"] and (
                w["last_seen_at"] is None or l["last_seen_at"] > w["last_seen_at"]
            ):
                update_parts.append("last_seen_at = :ll")
                params["ll"] = l["last_seen_at"]

            await db.execute(
                text(
                    f"UPDATE person_identities SET {', '.join(update_parts)} "
                    f"WHERE id::text = :winner"
                ),
                params,
            )

            await _absorb_face_embeddings(db, winner_id, loser_id, max_faces)
            await _absorb_body_embeddings(db, winner_id, loser_id, max_bodies)
            await _revote_person_gender(db, winner_id)

            await db.execute(
                text("""
                    INSERT INTO identity_merge_events (
                      merged_at, source, job_run_id, job_run_at,
                      winner_person_id, loser_person_id,
                      face_similarity, winner_face_score, loser_face_score,
                      winner_first_seen_at, loser_first_seen_at,
                      loser_visit_count, loser_track_count,
                      winner_visit_count_before,
                      winner_face_crop_path, loser_face_crop_path,
                      metadata_json, created_at, updated_at
                    ) VALUES (
                      now(), 'force_merge', NULL, now(),
                      CAST(:winner AS uuid), CAST(:loser AS uuid),
                      NULL, :w_score, :l_score,
                      :w_first, :l_first,
                      :l_visits, :l_tracks,
                      :w_visits,
                      :w_crop, :l_crop,
                      CAST(:meta AS jsonb), now(), now()
                    )
                """),
                {
                    "winner": winner_id,
                    "loser": loser_id,
                    "w_score": w.get("best_face_score"),
                    "l_score": l.get("best_face_score"),
                    "w_first": w.get("first_seen_at"),
                    "l_first": l.get("first_seen_at"),
                    "l_visits": int(l.get("visit_count") or 0),
                    "l_tracks": int(ts.get(loser_id, 0)),
                    "w_visits": int(w.get("visit_count") or 0),
                    "w_crop": w.get("face_crop_path"),
                    "l_crop": l.get("face_crop_path"),
                    "meta": json.dumps({"source": "force_merge_persons.py"}),
                },
            )

            await db.execute(
                text("DELETE FROM person_identities WHERE id::text = :loser"),
                {"loser": loser_id},
            )

        await db.commit()
        logger.info(f"force_merge OK {loser_id[:8]} → {winner_id[:8]}")
        print(f"  OK merged {loser_id[:8]} → {winner_id[:8]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Force-merge person identities")
    p.add_argument("--winner", required=True, help="Surviving person UUID")
    p.add_argument("--loser", required=True, help="Person UUID to absorb and delete")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args.winner, args.loser, args.apply))


if __name__ == "__main__":
    main()
