"""Debug detection service."""

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from typing import Optional
from uuid import UUID

from app.core.db.models.camera import Camera
from app.core.db.models.tracking import TrackSession
from app.modules.debug.schemas import (
    ActiveTracksRealtimeResponse, ActiveTrackResponse, ActiveTrackDemographics,
    UniquePersonItem, PaginatedUniquePersonsResponse, PersonTrackDetail, PaginatedTracksResponse
)


class DebugService:
    
    @staticmethod
    async def get_unique_persons(
        db: AsyncSession,
        page: int = 1,
        size: int = 10,
        search: Optional[str] = None
    ) -> PaginatedUniquePersonsResponse:
        """Get all unique person identities with basic statistics (paginated)."""
        from app.core.db.models.person import PersonIdentity
        from app.core.db.models.tracking import TrackSession
        from sqlalchemy import Date, or_, String, func

        # Build count query
        count_stmt = select(func.count(PersonIdentity.id))
        if search:
            count_stmt = count_stmt.where(
                or_(
                    PersonIdentity.gender.ilike(f"%{search}%"),
                    PersonIdentity.age_group.ilike(f"%{search}%"),
                    func.cast(PersonIdentity.id, String).ilike(f"%{search}%")
                )
            )
        total_count = (await db.execute(count_stmt)).scalar() or 0

        # Build main paginated query with outer join to count tracks and distinct days
        stmt = (
            select(
                PersonIdentity,
                func.count(TrackSession.id).label("total_tracks"),
                func.count(func.distinct(func.cast(TrackSession.started_at, Date))).label("total_days")
            )
            .outerjoin(TrackSession, PersonIdentity.id == TrackSession.person_identity_id)
        )
        if search:
            stmt = stmt.where(
                or_(
                    PersonIdentity.gender.ilike(f"%{search}%"),
                    PersonIdentity.age_group.ilike(f"%{search}%"),
                    func.cast(PersonIdentity.id, String).ilike(f"%{search}%")
                )
            )
        
        stmt = (
            stmt.group_by(PersonIdentity.id)
            .order_by(PersonIdentity.last_seen_at.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        
        result = await db.execute(stmt)
        rows = result.all()

        persons_list = []
        for person, total_tracks, total_days in rows:
            persons_list.append(
                UniquePersonItem(
                    id=person.id,
                    gender=person.gender,
                    age_group=person.age_group,
                    estimated_age=person.estimated_age,
                    best_face_score=person.best_face_score,
                    face_crop_path=person.face_crop_path,
                    first_seen_at=person.first_seen_at,
                    last_seen_at=person.last_seen_at,
                    visit_count=person.visit_count,
                    total_tracks=total_tracks,
                    total_days=total_days,
                )
            )

        return PaginatedUniquePersonsResponse(
            total_count=total_count,
            page=page,
            size=size,
            persons=persons_list
        )

    @staticmethod
    async def get_unique_person_tracks(
        db: AsyncSession,
        person_id: UUID,
        page: int = 1,
        size: int = 10
    ) -> PaginatedTracksResponse:
        """Get all track sessions where a person was detected (paginated)."""
        from app.core.db.models.tracking import TrackSession
        from app.core.db.models.camera import Camera
        from app.core.db.models.person import PersonFaceEmbedding

        # Count total tracks
        count_stmt = select(func.count(TrackSession.id)).where(TrackSession.person_identity_id == person_id)
        total_count = (await db.execute(count_stmt)).scalar() or 0

        # Fetch paginated tracks
        stmt = (
            select(TrackSession, Camera.name)
            .join(Camera, TrackSession.camera_id == Camera.id)
            .where(TrackSession.person_identity_id == person_id)
            .order_by(TrackSession.started_at.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        result = await db.execute(stmt)
        rows = result.all()

        # Fetch all face crops for this person to correlate them by timestamp
        face_stmt = select(PersonFaceEmbedding.face_crop_path, PersonFaceEmbedding.captured_at).where(
            PersonFaceEmbedding.person_identity_id == person_id
        )
        face_result = await db.execute(face_stmt)
        face_crops = face_result.all()

        tracks_list = []
        for ts, camera_name in rows:
            end_time = ts.ended_at or ts.last_seen_at
            duration = (end_time - ts.started_at).total_seconds()

            # Find matching face crop captured during this track session
            session_face_path = None
            for face_path, cap_time in face_crops:
                if face_path and ts.started_at <= cap_time <= end_time:
                    session_face_path = face_path
                    break

            tracks_list.append(
                PersonTrackDetail(
                    track_session_id=ts.id,
                    camera_name=camera_name,
                    started_at=ts.started_at,
                    ended_at=ts.ended_at,
                    duration_seconds=max(0.0, duration),
                    body_crop_path=ts.best_crop_path,
                    face_crop_path=session_face_path,
                )
            )

        return PaginatedTracksResponse(
            total_count=total_count,
            page=page,
            size=size,
            tracks=tracks_list
        )
    
    @staticmethod
    async def get_active_tracks(db: AsyncSession) -> ActiveTracksRealtimeResponse:
        """Get all active tracks currently in memory across all cameras."""
        # 1. Fetch friendly camera names to resolve UUID mapping
        camera_query = select(Camera.id, Camera.name)
        camera_rows = (await db.execute(camera_query)).all()
        camera_map = {row.id: row.name for row in camera_rows}

        # 2. Fetch the total number of inactive tracks (is_active = False)
        inactive_query = select(func.count(TrackSession.id)).where(TrackSession.is_active == False)
        total_inactive = (await db.execute(inactive_query)).scalar() or 0

        # 3. Retrieve active camera workers from Supervisor
        from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
        supervisor = WorkerSupervisor.get_instance()
        
        # 4. Collect resolved person identities to fetch their registered face crops from DB
        resolved_ids = set()
        for key, worker in supervisor.workers.items():
            if worker.is_running:
                for track in worker.track_manager.get_active_tracks():
                    if track.person_identity_id:
                        resolved_ids.add(track.person_identity_id)

        identity_face_map = {}
        if resolved_ids:
            from app.core.db.models.person import PersonIdentity
            identity_query = select(PersonIdentity.id, PersonIdentity.face_crop_path).where(PersonIdentity.id.in_(resolved_ids))
            identity_rows = (await db.execute(identity_query)).all()
            identity_face_map = {row.id: row.face_crop_path for row in identity_rows}

        active_tracks_list = []
        total_identified = 0

        for key, worker in supervisor.workers.items():
            if not worker.is_running:
                continue
            
            camera_name = camera_map.get(worker.camera_id, f"Camera {str(worker.camera_id)[:8]}")
            tracks = worker.track_manager.get_active_tracks()
            
            for track in tracks:
                face_crop_path = None
                face_score = None
                demographics = None
                
                if track.best_demographics:
                    face_crop_path = track.best_demographics.get("face_crop_path")
                    face_score = track.best_demographics.get("face_score")
                    demographics = ActiveTrackDemographics(
                        age=track.best_demographics.get("age"),
                        gender=track.best_demographics.get("gender"),
                        age_group=track.best_demographics.get("age_group")
                    )

                identity_face_crop_path = None
                if track.person_identity_id:
                    identity_face_crop_path = identity_face_map.get(track.person_identity_id)

                if track.person_identity_id is not None:
                    total_identified += 1

                active_tracks_list.append(
                    ActiveTrackResponse(
                        camera_id=worker.camera_id,
                        camera_name=camera_name,
                        local_track_id=track.local_track_id,
                        track_session_id=track.track_session_id,
                        person_identity_id=track.person_identity_id,
                        started_at=track.started_at,
                        last_seen_at=track.last_seen_at,
                        age_seconds=track.track_age_seconds,
                        total_frames=track.total_frames,
                        stability_score=track.stability_score,
                        reid_attempted=track.reid_attempted,
                        reid_resolved=track.reid_resolved,
                        reid_confident=track.reid_confident,
                        reid_score=track.reid_score,
                        reid_frame_count=track.reid_frame_count,
                        best_crop_quality=track.best_crop_quality,
                        best_crop_path=track.best_crop_path,
                        current_crop_quality=track.current_crop_quality,
                        current_crop_path=track.current_crop_path,
                        face_crop_path=face_crop_path,
                        current_face_crop_path=track.current_face_crop_path,
                        current_face_score=track.current_face_score,
                        identity_face_crop_path=identity_face_crop_path,
                        face_score=face_score,
                        demographics=demographics
                    )
                )

        return ActiveTracksRealtimeResponse(
            total_active_tracks=len(active_tracks_list),
            total_identified_tracks=total_identified,
            total_inactive_tracks=total_inactive,
            active_tracks=active_tracks_list
        )
