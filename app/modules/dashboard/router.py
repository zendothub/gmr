"""Dashboard API — current-day attendance summary.

  GET /api/v1/dashboard/today
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.dashboard import service as dashboard_svc
from app.modules.dashboard.schemas import TodayDashboardResponse

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


@router.get("/today", response_model=TodayDashboardResponse)
async def get_today_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Current-day attendance summary for the dashboard.

    Always resolves to today's date (IST) — no parameters. Counts:
    - **present** / **absent** / **on_leave** / **late** / **weekend_offs** — same
      definitions as `GET /employees/attendance/report?period=daily`
    - **gender** — breakdown (male/female/other/unspecified) of employees present
      today (present + late), not the total workforce
    """
    return await dashboard_svc.get_today_summary(db)
