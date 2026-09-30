#!/usr/bin/env python3
"""
revalidate_staff_seen_sept.py — revalidate is_staff identities seen in a date range.

Population: person_identities WHERE is_staff = TRUE AND has a track_session whose
IST date falls in [START, END] (default 2026-09-01 .. 2026-09-09).

Rule (identical to live staff_classifier._evaluate_and_apply):
  1. Face required when STAFF_REQUIRE_FACE (default True): >=1 face embedding.
  2. Day list = DISTINCT IST dates of ALL track_sessions (full history, as live).
  3. passes_staff_rule(days, consec=5, window=15, min_days=11):
       max consecutive-day streak >= 5  OR  >= 11 distinct days in any 15-day span.
  4. Fail -> demote (is_staff = FALSE). Pass -> keep.

Dry-run by default. --apply writes demotions.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/revalidate_staff_seen_sept.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/revalidate_staff_seen_sept.py --apply
    PYTHONPATH=/gmr/gmr venv/bin/python danger/revalidate_staff_seen_sept.py \
        --start 2026-09-01 --end 2026-09-09 --apply
"""

from __future__ import annotations

import argparse
import uuid
from datetime import date

from loguru import logger
from sqlalchemy import text

from app.config import get_settings
from app.core.db.session import AsyncSessionLocal
from app.modules.reid.staff_classifier import passes_staff_rule


async def run(start: str, end: str, apply: bool) -> None:
    settings = get_settings()
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    async with AsyncSessionLocal() as db:
        staff_rows = (
            await db.execute(
                text(
                    """
                    SELECT DISTINCT pi.id
                    FROM person_identities pi
                    JOIN track_sessions t ON t.person_identity_id = pi.id
                    WHERE pi.is_staff = TRUE
                      AND (t.started_at AT TIME ZONE 'Asia/Kolkata')::date
                          BETWEEN :start AND :end
                    ORDER BY 1
                    """
                ),
                {"start": start_d, "end": end_d},
            )
        ).fetchall()
        pids = [r[0] for r in staff_rows]
        logger.info(
            f"Found {len(pids)} is_staff identity(ies) seen {start}..{end} (IST)"
        )

        demote: list[tuple[uuid.UUID, int, str]] = []
        keep = 0
        for pid in pids:
            meta = (
                await db.execute(
                    text(
                        """
                        SELECT EXISTS(
                            SELECT 1 FROM person_face_embeddings fe
                            WHERE fe.person_identity_id = :pid
                        ) AS has_face
                        """
                    ),
                    {"pid": pid},
                )
            ).scalar()
            has_face = bool(meta)
            day_rows = (
                await db.execute(
                    text(
                        """
                        SELECT DISTINCT (started_at AT TIME ZONE 'Asia/Kolkata')::date AS d
                        FROM track_sessions
                        WHERE person_identity_id = :pid
                          AND started_at IS NOT NULL
                        ORDER BY 1
                        """
                    ),
                    {"pid": pid},
                )
            ).fetchall()
            days = [r[0] for r in day_rows if r[0] is not None]
            ok = (has_face or not settings.STAFF_REQUIRE_FACE) and passes_staff_rule(
                days,
                consec=settings.STAFF_CONSECUTIVE_DAYS,
                window=settings.STAFF_WINDOW_DAYS,
                min_days=settings.STAFF_WINDOW_MIN_DAYS,
            )
            short = str(pid)[:8]
            if ok:
                keep += 1
                logger.info(
                    f"[KEEP] {short} days={len(days)} face={has_face} "
                    f"first={days[0] if days else None} last={days[-1] if days else None}"
                )
            else:
                reason = "no_face" if not has_face else "rule_fail"
                demote.append((pid, len(days), reason))
                logger.info(
                    f"[DEMOTE] {short} days={len(days)} face={has_face} reason={reason} "
                    f"first={days[0] if days else None} last={days[-1] if days else None}"
                )

        logger.info(
            f"Summary: total={len(pids)} keep={keep} demote={len(demote)} "
            f"(consec>={settings.STAFF_CONSECUTIVE_DAYS} OR "
            f">={settings.STAFF_WINDOW_MIN_DAYS}d in {settings.STAFF_WINDOW_DAYS}d window, "
            f"require_face={settings.STAFF_REQUIRE_FACE})"
        )

        if not apply:
            logger.info("DRY-RUN — no changes written. Re-run with --apply to demote.")
            return

        for pid, ndays, reason in demote:
            await db.execute(
                text("UPDATE person_identities SET is_staff = FALSE WHERE id = :pid"),
                {"pid": pid},
            )
            logger.info(f"[APPLIED] {str(pid)[:8]} demoted (days={ndays}, reason={reason})")
        await db.commit()
        logger.info(f"Done — demoted {len(demote)} identity(ies).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-09-01", help="IST date, inclusive")
    parser.add_argument("--end", default="2026-09-09", help="IST date, inclusive")
    parser.add_argument("--apply", action="store_true", help="Write demotions")
    args = parser.parse_args()
    date.fromisoformat(args.start)
    date.fromisoformat(args.end)
    logger.info(f"revalidate_staff_seen_sept start={args.start} end={args.end} apply={args.apply}")
    import asyncio

    asyncio.run(run(args.start, args.end, args.apply))