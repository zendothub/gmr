"""Service layer for querying employee check-in events."""

from __future__ import annotations

from typing import Tuple, List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.attendance import EmployeeCheckIn


async def get_checkins(
    db: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
) -> Tuple[List[EmployeeCheckIn], int]:
    """Fetch check-in events ordered by checked_in_at DESC (most recent first).

    Simple pagination — no filters.
    """
    query = select(EmployeeCheckIn)

    # Total count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Order descending by checked_in_at (most recent first) + paginate
    query = (
        query.order_by(EmployeeCheckIn.checked_in_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(query)
    return list(result.scalars().all()), total
