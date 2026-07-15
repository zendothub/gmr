#!/usr/bin/env python3
"""
backfill_billing_dwell.py — insert BillingInteractions for counter dwell ≥ thr
that never got a row under a higher historical threshold.

Mirrors the live path: one BI per (track_session_id, zone_id).
Candidates from Counter-zone events (zone_exit / zone_dwell_milestone /
billing_interaction) with metadata dwell_seconds ≥ --min-dwell.

Usage (dry-run default):
  PYTHONPATH=/gmr/gmr venv/bin/python danger/backfill_billing_dwell.py
  PYTHONPATH=/gmr/gmr venv/bin/python danger/backfill_billing_dwell.py \\
      --min-dwell 50 --days 7 --cleanup-prior-backfill --update-rule --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from loguru import logger
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal

DEFAULT_ZONE_ID = "57250117-4363-44e9-ad02-c404fd349724"  # Counter / Apollo counter
DEFAULT_RULE_ID = "4522f8dd-1c4b-4075-9b7d-d060bef6927d"  # Billing Counter Interaction


async def _window(db, days: int):
    r = await db.execute(
        text(
            """
            SELECT
              (date_trunc('day', now() AT TIME ZONE 'Asia/Kolkata')
               - (CAST(:days AS int) * interval '1 day'))
              AT TIME ZONE 'Asia/Kolkata' AS start_ts,
              now() AS end_ts,
              now() AT TIME ZONE 'Asia/Kolkata' AS now_ist
            """
        ),
        {"days": days},
    )
    row = r.fetchone()
    return row.start_ts, row.end_ts, row.now_ist


async def cleanup_prior_backfill(db, days: int | None) -> int:
    """Delete previous script inserts tagged metadata.backfill=true."""
    if days is None:
        r = await db.execute(
            text(
                """
                DELETE FROM billing_interactions
                WHERE metadata_json->>'backfill' = 'true'
                RETURNING id
                """
            )
        )
    else:
        start_ts, _, _ = await _window(db, days)
        r = await db.execute(
            text(
                """
                DELETE FROM billing_interactions
                WHERE metadata_json->>'backfill' = 'true'
                  AND entered_at >= :start_ts
                RETURNING id
                """
            ),
            {"start_ts": start_ts},
        )
    n = len(r.fetchall())
    await db.commit()
    return n


async def fetch_candidates(db, days: int, min_dwell: float, zone_id: str):
    start_ts, end_ts, now_ist = await _window(db, days)
    logger.info(
        f"Window start={start_ts} end≈{now_ist} (days={days}) "
        f"min_dwell={min_dwell}s zone={zone_id[:8]}…"
    )

    r = await db.execute(
        text(
            """
            WITH hits AS (
              SELECT
                e.track_session_id,
                e.person_identity_id,
                e.camera_id,
                e.zone_id,
                e.occurred_at,
                (e.metadata_json->>'dwell_seconds')::float AS dwell
              FROM events e
              WHERE e.zone_id = CAST(:zid AS uuid)
                AND e.occurred_at >= :start_ts
                AND e.occurred_at <= :end_ts
                AND e.track_session_id IS NOT NULL
                AND e.event_type IN (
                  'zone_exit', 'zone_dwell_milestone', 'billing_interaction'
                )
                AND e.metadata_json ? 'dwell_seconds'
                AND (e.metadata_json->>'dwell_seconds')::float >= :min_dwell
            ),
            per_track AS (
              SELECT
                track_session_id,
                MAX(dwell) AS max_dwell,
                MIN(occurred_at) AS entered_at,
                (ARRAY_AGG(person_identity_id ORDER BY occurred_at DESC)
                  FILTER (WHERE person_identity_id IS NOT NULL))[1] AS person_identity_id,
                (ARRAY_AGG(camera_id ORDER BY occurred_at DESC)
                  FILTER (WHERE camera_id IS NOT NULL))[1] AS camera_id,
                (ARRAY_AGG(zone_id ORDER BY occurred_at DESC))[1] AS zone_id
              FROM hits
              GROUP BY track_session_id
            )
            SELECT
              p.track_session_id::text AS track_session_id,
              p.person_identity_id::text AS person_identity_id,
              p.camera_id::text AS camera_id,
              p.zone_id::text AS zone_id,
              p.max_dwell,
              p.entered_at,
              COALESCE(pi.is_staff, false) AS is_staff
            FROM per_track p
            LEFT JOIN person_identities pi ON pi.id = p.person_identity_id
            WHERE p.person_identity_id IS NOT NULL
              AND p.camera_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM billing_interactions bi
                WHERE bi.track_session_id = p.track_session_id
                  AND bi.zone_id = p.zone_id
              )
            ORDER BY p.entered_at
            """
        ),
        {
            "zid": zone_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "min_dwell": min_dwell,
        },
    )
    return [dict(row._mapping) for row in r.fetchall()], start_ts


async def apply_inserts(db, rows: list[dict], min_dwell: float) -> int:
    n = 0
    for row in rows:
        meta = {
            "dwell_seconds": float(row["max_dwell"]),
            "backfill": True,
            "source": f"dwell_ge_{int(min_dwell)}_events",
            "min_dwell": float(min_dwell),
            "original_threshold": 90,
        }
        await db.execute(
            text(
                """
                INSERT INTO billing_interactions (
                  id, camera_id, person_identity_id, track_session_id, zone_id,
                  entered_at, exited_at, dwell_seconds, interaction_type,
                  metadata_json, created_at, updated_at
                ) VALUES (
                  CAST(:id AS uuid), CAST(:camera_id AS uuid),
                  CAST(:person_identity_id AS uuid),
                  CAST(:track_session_id AS uuid), CAST(:zone_id AS uuid),
                  :entered_at, NULL, :dwell_seconds, 'billing_counter',
                  CAST(:metadata_json AS jsonb), now(), now()
                )
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "camera_id": row["camera_id"],
                "person_identity_id": row["person_identity_id"],
                "track_session_id": row["track_session_id"],
                "zone_id": row["zone_id"],
                "entered_at": row["entered_at"],
                "dwell_seconds": float(row["max_dwell"]),
                "metadata_json": json.dumps(meta),
            },
        )
        n += 1
    await db.commit()
    return n


async def update_rule(db, rule_id: str, thr: int) -> None:
    r = await db.execute(
        text(
            """
            UPDATE rules
            SET dwell_threshold_seconds = :thr, updated_at = now()
            WHERE id = CAST(:rid AS uuid)
            RETURNING name, dwell_threshold_seconds
            """
        ),
        {"thr": thr, "rid": rule_id},
    )
    row = r.fetchone()
    await db.commit()
    if row:
        logger.info(f"Updated rule '{row.name}' dwell → {row.dwell_threshold_seconds}s")
    else:
        logger.warning(f"Rule {rule_id} not found")


async def run(args: argparse.Namespace) -> None:
    async with AsyncSessionLocal() as db:
        r = await db.execute(
            text(
                """
                SELECT id::text, name, dwell_threshold_seconds, cooldown_seconds
                FROM rules WHERE id = CAST(:rid AS uuid)
                """
            ),
            {"rid": args.rule_id},
        )
        rule = r.fetchone()
        if rule:
            logger.info(
                f"Rule '{rule.name}': dwell={rule.dwell_threshold_seconds}s "
                f"cooldown={rule.cooldown_seconds}s"
            )

        if args.cleanup_prior_backfill:
            if args.apply:
                deleted = await cleanup_prior_backfill(
                    db, days=args.days if args.cleanup_windowed else None
                )
                logger.info(f"Deleted {deleted} prior backfill BI rows")
            else:
                if args.cleanup_windowed:
                    start_ts, _, _ = await _window(db, args.days)
                    r = await db.execute(
                        text(
                            """
                            SELECT COUNT(*) FROM billing_interactions
                            WHERE metadata_json->>'backfill' = 'true'
                              AND entered_at >= :start_ts
                            """
                        ),
                        {"start_ts": start_ts},
                    )
                else:
                    r = await db.execute(
                        text(
                            """
                            SELECT COUNT(*) FROM billing_interactions
                            WHERE metadata_json->>'backfill' = 'true'
                            """
                        )
                    )
                logger.info(
                    f"DRY-RUN would delete {r.scalar()} prior backfill BI rows"
                )

        rows, start_ts = await fetch_candidates(
            db, args.days, args.min_dwell, args.zone_id
        )
        staff_n = sum(1 for x in rows if x["is_staff"])
        novis = len(rows) - staff_n
        logger.info(
            f"Candidates missing BI: {len(rows)} tracks "
            f"(non-staff={novis}, staff={staff_n})"
        )
        for x in rows[:12]:
            logger.info(
                f"  track={x['track_session_id'][:8]}… "
                f"person={x['person_identity_id'][:8]}… "
                f"dwell={x['max_dwell']:.1f}s staff={x['is_staff']} "
                f"at={x['entered_at']}"
            )
        if len(rows) > 12:
            logger.info(f"  … +{len(rows) - 12} more")

        if not args.apply:
            logger.info("DRY-RUN — no writes. Re-run with --apply.")
            return

        if args.update_rule:
            await update_rule(db, args.rule_id, int(args.min_dwell))

        if rows:
            n = await apply_inserts(db, rows, args.min_dwell)
            logger.info(f"Inserted {n} billing_interactions")
        else:
            logger.info("Nothing to insert")

        r = await db.execute(
            text(
                """
                SELECT
                  COUNT(DISTINCT bi.person_identity_id) FILTER (
                    WHERE COALESCE(pi.is_staff, false) = false
                  ) AS excl_staff,
                  COUNT(DISTINCT bi.person_identity_id) AS with_staff,
                  COUNT(*) AS rows
                FROM billing_interactions bi
                LEFT JOIN person_identities pi ON pi.id = bi.person_identity_id
                WHERE bi.entered_at >= :start_ts
                  AND bi.person_identity_id IS NOT NULL
                """
            ),
            {"start_ts": start_ts},
        )
        s = r.fetchone()
        logger.info(
            f"BI window after: rows={s.rows} excl_staff={s.excl_staff} "
            f"with_staff={s.with_staff}"
        )

        # today-only excl staff for quick match check
        r = await db.execute(
            text(
                """
                SELECT COUNT(DISTINCT bi.person_identity_id)
                FROM billing_interactions bi
                JOIN person_identities pi ON pi.id = bi.person_identity_id
                WHERE bi.entered_at >=
                  (date_trunc('day', now() AT TIME ZONE 'Asia/Kolkata')
                   AT TIME ZONE 'Asia/Kolkata')
                  AND COALESCE(pi.is_staff, false) = false
                """
            )
        )
        logger.info(f"Today excl-staff distinct purchases: {r.scalar()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-dwell", type=float, default=50.0)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--zone-id", default=DEFAULT_ZONE_ID)
    p.add_argument("--rule-id", default=DEFAULT_RULE_ID)
    p.add_argument(
        "--cleanup-prior-backfill",
        action="store_true",
        help="Delete BI rows with metadata.backfill=true before insert",
    )
    p.add_argument(
        "--cleanup-windowed",
        action="store_true",
        help="Limit cleanup to the --days window (default: all backfills)",
    )
    p.add_argument(
        "--update-rule",
        action="store_true",
        help="Set rules.dwell_threshold_seconds = min-dwell",
    )
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
