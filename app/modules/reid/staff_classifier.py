"""Live staff classification: consec-5 or 70% of any 15-day window, lazy cache.

DISABLED (2026-10-05): no longer called from camera_worker. Staff are now registered
explicitly with a photo via POST /api/staff/register (app/modules/staff/). The code
is kept for reference / rollback — re-enable by uncommenting the
schedule_staff_check() calls in app/modules/ai_runtime/camera_worker.py.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from loguru import logger
from sqlalchemy import text

from app.config import get_settings
from app.utils.time_utils import utc_now

_sem: Optional[asyncio.Semaphore] = None


class StaffCheckCache:
    def __init__(self) -> None:
        self.confirmed_staff: Set[uuid.UUID] = set()
        self.customer_until: Dict[uuid.UUID, datetime] = {}
        self.in_flight: Set[uuid.UUID] = set()

    def reset(self) -> None:
        self.confirmed_staff.clear()
        self.customer_until.clear()
        self.in_flight.clear()

    def should_skip(self, pid: uuid.UUID, now: datetime) -> bool:
        if pid in self.confirmed_staff:
            return True
        if pid in self.in_flight:
            return True
        until = self.customer_until.get(pid)
        return until is not None and now < until


cache = StaffCheckCache()


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(2)
    return _sem


def passes_staff_rule(
    days,
    consec: int = 5,
    window: int = 15,
    min_days: int = 11,
) -> bool:
    if not days:
        return False
    ds = sorted(set(days))
    streak = 1
    max_streak = 1
    for i in range(1, len(ds)):
        if (ds[i] - ds[i - 1]).days == 1:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 1
    if max_streak >= consec:
        return True
    span = window - 1
    j = 0
    n = len(ds)
    for i in range(n):
        while j < n and (ds[j] - ds[i]).days <= span:
            j += 1
        if j - i >= min_days:
            return True
    return False


def schedule_staff_check(person_id) -> None:
    if person_id is None:
        return
    pid = person_id if isinstance(person_id, uuid.UUID) else uuid.UUID(str(person_id))
    if cache.should_skip(pid, utc_now()):
        return
    cache.in_flight.add(pid)
    try:
        asyncio.get_running_loop().create_task(_run_staff_check(pid))
    except RuntimeError:
        cache.in_flight.discard(pid)


async def _run_staff_check(pid: uuid.UUID) -> None:
    try:
        async with _get_sem():
            await _evaluate_and_apply(pid)
    except Exception as e:
        logger.warning(f"Staff check failed for {str(pid)[:8]}: {e}")
    finally:
        cache.in_flight.discard(pid)


async def _evaluate_and_apply(pid: uuid.UUID) -> None:
    from app.core.db.session import AsyncSessionLocal

    settings = get_settings()
    async with AsyncSessionLocal() as db:
        meta = (
            await db.execute(
                text(
                    """
                    SELECT pi.is_staff,
                           EXISTS(
                               SELECT 1 FROM person_face_embeddings fe
                               WHERE fe.person_identity_id = pi.id
                           ) AS has_face
                    FROM person_identities pi
                    WHERE pi.id = :pid
                    """
                ),
                {"pid": pid},
            )
        ).first()
        if meta is None:
            return
        is_staff = bool(meta[0])
        has_face = bool(meta[1])
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
        ok = (
            has_face or not settings.STAFF_REQUIRE_FACE
        ) and passes_staff_rule(
            days,
            consec=settings.STAFF_CONSECUTIVE_DAYS,
            window=settings.STAFF_WINDOW_DAYS,
            min_days=settings.STAFF_WINDOW_MIN_DAYS,
        )
        short = str(pid)[:8]
        if ok:
            if not is_staff:
                await db.execute(
                    text("UPDATE person_identities SET is_staff = TRUE WHERE id = :pid"),
                    {"pid": pid},
                )
                await db.commit()
                logger.info(f"Staff check: {short} promoted (days={len(days)})")
            else:
                logger.debug(f"Staff check: {short} confirmed")
            cache.customer_until.pop(pid, None)
            cache.confirmed_staff.add(pid)
            return
        if is_staff:
            await db.execute(
                text("UPDATE person_identities SET is_staff = FALSE WHERE id = :pid"),
                {"pid": pid},
            )
            await db.commit()
            logger.info(f"Staff check: {short} demoted (days={len(days)})")
        else:
            logger.debug(f"Staff check: {short} customer, retry after 24h")
        cache.confirmed_staff.discard(pid)
        cache.customer_until[pid] = utc_now() + timedelta(
            hours=settings.STAFF_CUSTOMER_RECHECK_HOURS
        )
