"""Debug detection service."""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func, and_, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.db.models.debug import PersonDebug
from app.core.db.models.camera import Camera
from app.core.db.models.store import Store
from app.modules.debug.schemas import PersonDebugResponse, DebugSummary, DebugListResponse
from app.utils.time_utils import utc_now

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))


class DebugService:
    
    @staticmethod
    async def get_debug_records(
        db: AsyncSession,
        store_id: Optional[UUID] = None,
        camera_id: Optional[UUID] = None,
        time_range: str = "today",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: str = "all",  # all | detected | not_detected
        page: int = 1,
        limit: int = 50,
    ) -> DebugListResponse:
        """Get paginated debug records with filters."""
        
        # Handle time_range with IST timezone (like analytics API)
        now_utc = utc_now()
        now_ist = now_utc.astimezone(IST)
        
        if time_range == "today":
            # Start of today in IST (00:00 IST)
            start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
            # End of today in IST (23:59:59 IST)
            end_ist = start_ist + timedelta(days=1, microseconds=-1)
            start_time = start_ist.astimezone(timezone.utc)
            end_time = end_ist.astimezone(timezone.utc)
        elif time_range == "this_week":
            # Start of this week (Monday 00:00 IST)
            days_since_monday = now_ist.weekday()
            start_ist = (now_ist - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            # End is now
            end_ist = now_ist
            start_time = start_ist.astimezone(timezone.utc)
            end_time = end_ist.astimezone(timezone.utc)
        elif time_range == "custom":
            # Use provided start_time and end_time
            if not start_time or not end_time:
                # Default to last 24 hours if custom but no times provided
                end_time = now_utc
                start_time = end_time - timedelta(hours=24)
        else:
            # Default fallback
            end_time = now_utc
            start_time = end_time - timedelta(hours=24)
        
        # Build base query
        conditions = [
            PersonDebug.occurred_at >= start_time,
            PersonDebug.occurred_at <= end_time,
        ]
        
        if store_id:
            conditions.append(PersonDebug.store_id == store_id)
        if camera_id:
            conditions.append(PersonDebug.camera_id == camera_id)
        if status == "detected":
            conditions.append(PersonDebug.reid_success == True)
        elif status == "not_detected":
            conditions.append(PersonDebug.reid_success == False)
        
        # Count total
        count_query = select(func.count(PersonDebug.id)).where(and_(*conditions))
        total = (await db.execute(count_query)).scalar() or 0
        
        # Get records with joined data
        query = (
            select(PersonDebug)
            .options(joinedload(PersonDebug.camera), joinedload(PersonDebug.store))
            .where(and_(*conditions))
            .order_by(PersonDebug.occurred_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )
        
        result = await db.execute(query)
        records = result.unique().scalars().all()
        
        # Build summary
        summary_query = select(
            func.count(PersonDebug.id).label('total'),
            func.sum(func.cast(PersonDebug.reid_success, Integer)).label('detected'),
        ).where(and_(*conditions))
        
        summary_result = await db.execute(summary_query)
        summary_row = summary_result.first()
        
        total_count = summary_row.total or 0
        detected_count = summary_row.detected or 0
        
        # Get failure breakdown
        failure_query = select(
            PersonDebug.failure_stage,
            func.count(PersonDebug.id).label('count')
        ).where(
            and_(*conditions),
            PersonDebug.failure_stage.isnot(None)
        ).group_by(PersonDebug.failure_stage)
        
        failure_result = await db.execute(failure_query)
        failure_rows = failure_result.all()
        
        failure_by_stage = {row.failure_stage: row.count for row in failure_rows if row.failure_stage}
        
        summary = DebugSummary(
            total=total_count,
            detected=detected_count,
            not_detected=total_count - detected_count,
            detection_rate=round((detected_count / total_count * 100) if total_count > 0 else 0, 1),
            failure_by_stage=failure_by_stage,
        )
        
        # Convert to response objects
        response_records = []
        for record in records:
            record_dict = {k: v for k, v in record.__dict__.items() if not k.startswith('_')}
            record_dict["camera_name"] = record.camera.name if record.camera else None
            record_dict["store_name"] = record.store.name if record.store else None
            # Normalize: body_crop_path may be None; include it explicitly
            record_dict.setdefault("body_crop_path", record_dict.get("body_crop_path") or record_dict.get("crop_path"))
            response_records.append(PersonDebugResponse(**record_dict))
        
        return DebugListResponse(
            summary=summary,
            records=response_records,
            total=total,
            page=page,
            limit=limit,
        )
