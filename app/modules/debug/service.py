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
from app.core.db.models.person import PersonIdentity, PersonEmbedding
from app.core.db.models.tracking import TrackSession
from app.modules.debug.schemas import (
    PersonDebugResponse, DebugSummary, DebugListResponse,
    PersonIdentityDebugResponse, PersonsListResponse,
    PersonTrackDebugResponse, PersonTracksListResponse
)
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
    
    @staticmethod
    async def get_debug_persons(
        db: AsyncSession,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        limit: int = 25,
    ) -> PersonsListResponse:
        """Get paginated list of unique persons."""
        
        # Default to last 24 hours if no times provided
        if not start_time or not end_time:
            end_time = utc_now()
            start_time = end_time - timedelta(hours=24)
        
        # Subquery to count tracks per person
        track_count_subquery = (
            select(
                TrackSession.person_identity_id,
                func.count(TrackSession.id).label('track_count')
            )
            .group_by(TrackSession.person_identity_id)
            .subquery()
        )
        
        # Subquery to get best quality body crop per person
        best_crop_subquery = (
            select(
                PersonEmbedding.person_identity_id,
                PersonEmbedding.crop_path,
                func.row_number().over(
                    partition_by=PersonEmbedding.person_identity_id,
                    order_by=PersonEmbedding.crop_quality.desc()
                ).label('rn')
            )
            .subquery()
        )
        
        # Main query
        conditions = [
            PersonIdentity.first_seen_at >= start_time,
            PersonIdentity.first_seen_at <= end_time,
        ]
        
        # Count total
        count_query = select(func.count(PersonIdentity.id)).where(and_(*conditions))
        total = (await db.execute(count_query)).scalar() or 0
        
        # Get persons with all computed fields
        query = (
            select(
                PersonIdentity.id,
                PersonIdentity.first_seen_at,
                PersonIdentity.last_seen_at,
                func.coalesce(track_count_subquery.c.track_count, 0).label('total_tracks'),
                PersonIdentity.estimated_age.label('age'),
                PersonIdentity.gender,
                PersonIdentity.best_face_score,
                PersonIdentity.face_crop_path,
                best_crop_subquery.c.crop_path.label('body_crop_path'),
            )
            .outerjoin(
                track_count_subquery,
                PersonIdentity.id == track_count_subquery.c.person_identity_id
            )
            .outerjoin(
                best_crop_subquery,
                and_(
                    PersonIdentity.id == best_crop_subquery.c.person_identity_id,
                    best_crop_subquery.c.rn == 1
                )
            )
            .where(and_(*conditions))
            .order_by(PersonIdentity.first_seen_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )
        
        result = await db.execute(query)
        rows = result.all()
        
        # Build response
        persons = []
        for row in rows:
            persons.append(PersonIdentityDebugResponse(
                id=row.id,
                first_seen_at=row.first_seen_at,
                last_seen_at=row.last_seen_at,
                total_tracks=row.total_tracks,
                age=row.age,
                gender=row.gender,
                avg_reid_score=None,  # Would need to compute from embeddings if needed
                has_face=row.best_face_score is not None and row.best_face_score > 0,
                face_age=row.age,  # Use same as estimated_age
                face_gender=row.gender,
                body_crop_path=row.body_crop_path,
                face_crop_path=row.face_crop_path,
            ))
        
        return PersonsListResponse(
            persons=persons,
            total=total,
            page=page,
            limit=limit,
        )
    
    @staticmethod
    async def get_person_tracks(
        db: AsyncSession,
        person_id: UUID,
        page: int = 1,
        limit: int = 10,
    ) -> PersonTracksListResponse:
        """Get paginated tracks for a specific person."""
        
        # Count total tracks for this person
        count_query = select(func.count(TrackSession.id)).where(
            TrackSession.person_identity_id == person_id
        )
        total = (await db.execute(count_query)).scalar() or 0
        
        if total == 0:
            # Person not found or has no tracks
            return PersonTracksListResponse(
                tracks=[],
                total=0,
                page=page,
                limit=limit,
            )
        
        # Get tracks with camera info
        query = (
            select(TrackSession)
            .options(joinedload(TrackSession.camera))
            .where(TrackSession.person_identity_id == person_id)
            .order_by(TrackSession.started_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )
        
        result = await db.execute(query)
        track_sessions = result.unique().scalars().all()
        
        # Build response
        tracks = []
        for track in track_sessions:
            # Calculate duration in seconds
            duration_seconds = None
            if track.ended_at:
                duration_seconds = (track.ended_at - track.started_at).total_seconds()
            
            # Parse age from age_group if needed (e.g., "young_adult" -> estimate)
            age = None
            if track.age_group:
                age_map = {
                    "child": 10,
                    "teenager": 16,
                    "young_adult": 25,
                    "middle_adult": 45,
                    "senior": 65,
                }
                age = age_map.get(track.age_group)
            
            tracks.append(PersonTrackDebugResponse(
                id=track.id,
                person_identity_id=track.person_identity_id,
                camera_id=track.camera_id,
                camera_name=track.camera.name if track.camera else None,
                started_at=track.started_at,
                ended_at=track.ended_at,
                duration_seconds=duration_seconds,
                total_frames=track.total_frames,
                avg_quality_score=track.stability_score,  # Use stability as proxy for quality
                avg_detection_confidence=track.avg_confidence,
                age=age,
                gender=track.gender,
                body_crop_path=track.best_crop_path,
                face_crop_path=None,  # Not stored on TrackSession
            ))
        
        return PersonTracksListResponse(
            tracks=tracks,
            total=total,
            page=page,
            limit=limit,
        )
