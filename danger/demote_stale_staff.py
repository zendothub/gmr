#!/usr/bin/env python3
"""
demote_stale_staff.py — demote auto-classified staff that fail the >10-day rule.

Rule (2026-10-15, user decision):
  - registered staff (metadata_json.staff_registered) → ALWAYS keep
  - auto staff with >10 DISTINCT lifetime IST days with tracks → keep
  - auto staff with <=10 days → DEMOTE (is_staff = FALSE)
  - auto staff with >10 days but not seen since RECENT_CUTOFF → review list
    only; demote only with --include-stale.

Demotion is history-preserving: the identity, tracks, events, billing and
embeddings stay attached; the person simply counts as a customer again.
(See staff/service.py: demotion re-adds the person to per-day counts.)

Dry-run by default. --apply writes demotions.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/demote_stale_staff.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/demote_stale_staff.py --apply
    PYTHONPATH=/gmr/gmr venv/bin/python danger/demote_stale_staff.py --apply --include-stale [--recent-days 5]
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import date, timedelta

from loguru import logger
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal


async def run(min_days: int, recent_days: int, include_stale: bool, apply: bool) -> None:
    cutoff = date.today() - timedelta(days=recent_days)
    async with AsyncSessionLocal() as db:
        staff_rows = (
            await db.execute(
                text(
                    """
                    SELECT pi.id,
                           COALESCE((pi.metadata_json->>'staff_registered')::boolean, false) AS reg,
                           pi.label
                    FROM person_identities pi
                    WHERE pi.is_staff = TRUE
                    ORDER BY 1
                    """
                )
            )
        ).fetchall()

        demote: list[tuple[uuid.UUID, int, str]] = []
        keep_reg = keep_days = 0
        stale: list[tuple[uuid.UUID, int, date | None]] = []
        noface_keep: list[tuple[uuid.UUID, int]] = []

        for pid, reg, label in staff_rows:
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
            ndays = len(days)
            first, last = (days[0], days[-1]) if days else (None, None)
            short = f"{str(pid)[:8]}({label or '-'})"

            if reg:
                keep_reg += 1
                # registered rows are never demoted, but report overlaps
                logger.info(f"[KEEP-reg] {short} days={ndays} last={last}")
                continue

            if ndays > min_days:
                if last is not None and last >= cutoff:
                    keep_days += 1
                    logger.info(f"[KEEP] {short} days={ndays} last={last} (active)")
                else:
                    stale.append((pid, ndays, last))
                    if include_stale:
                        demote.append((pid, ndays, f"stale(>{min_days}d,last={last})"))
                        logger.info(f"[DEMOTE-stale] {short} days={ndays} last={last}")
                    else:
                        logger.info(f"[REVIEW-stale] {short} days={ndays} last={last}")
                continue

            # <= min_days → demote regardless of face (rule is presence-based);
            # note faceless rows for visibility but still demote.
            demote.append((pid, ndays, f"days<={min_days}"))
            logger.info(f"[DEMOTE] {short} days={ndays} last={last} reason=days<={min_days}")

        logger.info(
            f"Summary: total={len(staff_rows)} keep_reg={keep_reg} keep_active={keep_days} "
            f"stale_review={len(stale) if not include_stale else 0} demote={len(demote)}"
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
    parser.add_argument("--min-days", type=int, default=10,
                        help="Keep auto-staff with MORE than this many distinct days (default 10)")
    parser.add_argument("--recent-days", type=int, default=5,
                        help="Recency cutoff for the stale review list (default 5)")
    parser.add_argument("--include-stale", action="store_true",
                        help="Also demote >min-days identities not seen in recent window")
    parser.add_argument("--apply", action="store_true", help="Write demotions")
    args = parser.parse_args()
    logger.info(f"demote_stale_staff min_days={args.min_days} recent_days={args.recent_days} "
                f"include_stale={args.include_stale} apply={args.apply}")
    asyncio.run(run(args.min_days, args.recent_days, args.include_stale, args.apply))
