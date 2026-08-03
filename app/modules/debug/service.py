"""Debug detection service."""

from datetime import datetime
from typing import Optional, Any
from uuid import UUID

from sqlalchemy import select, func, or_, String, Date, exists
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.camera import Camera
from app.core.db.models.tracking import TrackSession
from app.core.db.models.person import PersonIdentity
from app.core.db.models.audit import IdentityMergeEvent, FragmentedTrackEvent
from app.modules.debug.schemas import (
    ActiveTracksRealtimeResponse,
    ActiveTrackResponse,
    ActiveTrackDemographics,
    UniquePersonItem,
    PaginatedUniquePersonsResponse,
    PersonTrackDetail,
    PaginatedTracksResponse,
    TrackSessionDebugItem,
    TrackSessionPersonSummary,
    PaginatedTrackSessionsResponse,
    IdentityMergeEventItem,
    PaginatedIdentityMergeEventsResponse,
    FragmentedTrackEventItem,
    PaginatedFragmentedTrackEventsResponse,
)


def _parse_session_debug_log(bbox_history: Any) -> dict:
    """Safely extract quality/face debug fields from track_sessions.bbox_history.

    Legacy: JSON array of boxes → all debug fields None (UI shows N/A).
    Current: object with boxes + quality/face keys.
    Never raises on bad/unexpected shapes.
    """
    empty = {
        "best_crop_quality": None,
        "torso_visibility_ratio": None,
        "best_face_crop_path": None,
        "best_face_score": None,
    }
    if bbox_history is None:
        return empty
    if isinstance(bbox_history, list):
        return empty
    if not isinstance(bbox_history, dict):
        return empty

    def _opt_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _opt_str(v):
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    return {
        "best_crop_quality": _opt_float(bbox_history.get("best_crop_quality")),
        "torso_visibility_ratio": _opt_float(bbox_history.get("torso_visibility_ratio")),
        "best_face_crop_path": _opt_str(bbox_history.get("best_face_crop_path")),
        "best_face_score": _opt_float(bbox_history.get("best_face_score")),
    }


class DebugService:
    
    @staticmethod
    async def get_unique_persons(
        db: AsyncSession,
        page: int = 1,
        size: int = 10,
        search: Optional[str] = None,
        gender: Optional[str] = None,
        is_staff: Optional[bool] = None,
        has_purchase: Optional[bool] = None,
    ) -> PaginatedUniquePersonsResponse:
        """Get all unique person identities with basic statistics (paginated)."""
        from app.core.db.models.person import PersonIdentity
        from app.core.db.models.tracking import TrackSession
        from app.core.db.models.billing import BillingInteraction
        from sqlalchemy import Date, or_, String, func

        # ── Purchase count subquery ──────────────────────────────────────
        purchase_subq = (
            select(
                BillingInteraction.person_identity_id,
                func.count(BillingInteraction.id).label("purchase_count"),
            )
            .where(BillingInteraction.person_identity_id.isnot(None))
            .group_by(BillingInteraction.person_identity_id)
            .subquery()
        )

        # ── Count query ──────────────────────────────────────────────────
        count_stmt = select(func.count(PersonIdentity.id))
        if search:
            count_stmt = count_stmt.where(
                or_(
                    PersonIdentity.gender.ilike(f"%{search}%"),
                    PersonIdentity.age_group.ilike(f"%{search}%"),
                    func.cast(PersonIdentity.id, String).ilike(f"%{search}%")
                )
            )
        if gender:
            count_stmt = count_stmt.where(PersonIdentity.gender == gender)
        if is_staff is not None:
            count_stmt = count_stmt.where(PersonIdentity.is_staff.is_(is_staff))
        if has_purchase is not None:
            if has_purchase:
                # Persons who have at least one BillingInteraction
                count_stmt = count_stmt.where(
                    PersonIdentity.id.in_(
                        select(purchase_subq.c.person_identity_id)
                    )
                )
            else:
                count_stmt = count_stmt.where(
                    PersonIdentity.id.notin_(
                        select(purchase_subq.c.person_identity_id)
                    )
                )
        total_count = (await db.execute(count_stmt)).scalar() or 0

        # ── Main paginated query ─────────────────────────────────────────
        stmt = (
            select(
                PersonIdentity,
                func.count(TrackSession.id).label("total_tracks"),
                func.count(func.distinct(func.cast(TrackSession.started_at, Date))).label("total_days"),
                func.coalesce(func.max(purchase_subq.c.purchase_count), 0).label("purchase_count"),
            )
            .outerjoin(TrackSession, PersonIdentity.id == TrackSession.person_identity_id)
            .outerjoin(purchase_subq, PersonIdentity.id == purchase_subq.c.person_identity_id)
        )
        if search:
            stmt = stmt.where(
                or_(
                    PersonIdentity.gender.ilike(f"%{search}%"),
                    PersonIdentity.age_group.ilike(f"%{search}%"),
                    func.cast(PersonIdentity.id, String).ilike(f"%{search}%")
                )
            )
        if gender:
            stmt = stmt.where(PersonIdentity.gender == gender)
        if is_staff is not None:
            stmt = stmt.where(PersonIdentity.is_staff.is_(is_staff))
        if has_purchase is not None:
            if has_purchase:
                stmt = stmt.where(
                    PersonIdentity.id.in_(
                        select(purchase_subq.c.person_identity_id)
                    )
                )
            else:
                stmt = stmt.where(
                    PersonIdentity.id.notin_(
                        select(purchase_subq.c.person_identity_id)
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
        for person, total_tracks, total_days, purchase_count in rows:
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
                    is_staff=person.is_staff,
                    total_purchases=purchase_count,
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
    async def list_track_sessions(
        db: AsyncSession,
        page: int = 1,
        size: int = 20,
        search: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        assignment: str = "all",
        camera_id: Optional[UUID] = None,
        has_face: Optional[bool] = None,
        has_billing: Optional[bool] = None,
    ) -> PaginatedTrackSessionsResponse:
        """Browse track sessions with debug fields from bbox_history (safe for legacy arrays)."""
        from app.core.db.models.person import PersonIdentity
        from app.core.db.models.billing import BillingInteraction
        from sqlalchemy import cast, and_

        billing_count_subq = (
            select(
                BillingInteraction.track_session_id,
                func.count(BillingInteraction.id).label("billing_count"),
            )
            .where(BillingInteraction.track_session_id.isnot(None))
            .group_by(BillingInteraction.track_session_id)
            .subquery()
        )

        # Face path present only on object-shaped debug logs (legacy array → no face)
        face_txt = func.nullif(
            func.jsonb_extract_path_text(TrackSession.bbox_history, "best_face_crop_path"),
            "",
        )
        has_face_path = and_(
            func.jsonb_typeof(TrackSession.bbox_history) == "object",
            face_txt.isnot(None),
        )

        def _apply_filters(stmt):
            if search:
                stmt = stmt.where(
                    cast(TrackSession.id, String).ilike(f"%{search.strip()}%")
                )
            if start_time is not None:
                stmt = stmt.where(TrackSession.started_at >= start_time)
            if end_time is not None:
                stmt = stmt.where(TrackSession.started_at <= end_time)
            if assignment == "assigned":
                stmt = stmt.where(TrackSession.person_identity_id.isnot(None))
            elif assignment == "unassigned":
                stmt = stmt.where(TrackSession.person_identity_id.is_(None))
            if camera_id is not None:
                stmt = stmt.where(TrackSession.camera_id == camera_id)
            if has_face is True:
                stmt = stmt.where(has_face_path)
            elif has_face is False:
                stmt = stmt.where(~has_face_path)
            if has_billing is True:
                stmt = stmt.where(
                    exists(
                        select(1).where(
                            BillingInteraction.track_session_id == TrackSession.id
                        )
                    )
                )
            elif has_billing is False:
                stmt = stmt.where(
                    ~exists(
                        select(1).where(
                            BillingInteraction.track_session_id == TrackSession.id
                        )
                    )
                )
            return stmt

        count_stmt = _apply_filters(select(func.count(TrackSession.id)))
        total_count = (await db.execute(count_stmt)).scalar() or 0

        stmt = (
            select(
                TrackSession,
                Camera.name.label("camera_name"),
                func.coalesce(billing_count_subq.c.billing_count, 0).label("billing_count"),
            )
            .join(Camera, TrackSession.camera_id == Camera.id)
            .outerjoin(
                billing_count_subq,
                billing_count_subq.c.track_session_id == TrackSession.id,
            )
        )
        stmt = _apply_filters(stmt)
        stmt = (
            stmt.order_by(TrackSession.started_at.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        rows = (await db.execute(stmt)).all()

        person_ids = {
            ts.person_identity_id for ts, _, _ in rows if ts.person_identity_id is not None
        }
        person_map: dict = {}
        if person_ids:
            purchase_subq = (
                select(
                    BillingInteraction.person_identity_id,
                    func.count(BillingInteraction.id).label("purchase_count"),
                )
                .where(BillingInteraction.person_identity_id.in_(person_ids))
                .group_by(BillingInteraction.person_identity_id)
                .subquery()
            )
            tracks_subq = (
                select(
                    TrackSession.person_identity_id,
                    func.count(TrackSession.id).label("total_tracks"),
                    func.count(
                        func.distinct(func.cast(TrackSession.started_at, Date))
                    ).label("total_days"),
                )
                .where(TrackSession.person_identity_id.in_(person_ids))
                .group_by(TrackSession.person_identity_id)
                .subquery()
            )
            p_stmt = (
                select(
                    PersonIdentity,
                    func.coalesce(tracks_subq.c.total_tracks, 0).label("total_tracks"),
                    func.coalesce(tracks_subq.c.total_days, 0).label("total_days"),
                    func.coalesce(purchase_subq.c.purchase_count, 0).label("purchase_count"),
                )
                .outerjoin(tracks_subq, tracks_subq.c.person_identity_id == PersonIdentity.id)
                .outerjoin(purchase_subq, purchase_subq.c.person_identity_id == PersonIdentity.id)
                .where(PersonIdentity.id.in_(person_ids))
            )
            for person, total_tracks, total_days, purchase_count in (await db.execute(p_stmt)).all():
                person_map[person.id] = TrackSessionPersonSummary(
                    id=person.id,
                    gender=person.gender,
                    age_group=person.age_group,
                    estimated_age=person.estimated_age,
                    is_staff=bool(person.is_staff),
                    first_seen_at=person.first_seen_at,
                    last_seen_at=person.last_seen_at,
                    visit_count=person.visit_count or 0,
                    total_tracks=int(total_tracks or 0),
                    total_days=int(total_days or 0),
                    face_crop_path=person.face_crop_path,
                    best_face_score=person.best_face_score,
                    total_purchases=int(purchase_count or 0),
                )

        tracks_out: list[TrackSessionDebugItem] = []
        for ts, camera_name, billing_count in rows:
            dbg = _parse_session_debug_log(ts.bbox_history)
            end_time = ts.ended_at or ts.last_seen_at
            duration = max(0.0, (end_time - ts.started_at).total_seconds()) if end_time else 0.0
            pid = ts.person_identity_id
            tracks_out.append(
                TrackSessionDebugItem(
                    track_session_id=ts.id,
                    camera_id=ts.camera_id,
                    camera_name=camera_name or str(ts.camera_id)[:8],
                    local_track_id=ts.local_track_id,
                    person_identity_id=pid,
                    started_at=ts.started_at,
                    ended_at=ts.ended_at,
                    last_seen_at=ts.last_seen_at,
                    duration_seconds=duration,
                    total_frames=ts.total_frames or 0,
                    is_active=bool(ts.is_active),
                    gender=ts.gender,
                    age_group=ts.age_group,
                    avg_confidence=ts.avg_confidence,
                    stability_score=ts.stability_score,
                    body_crop_path=ts.best_crop_path,
                    best_crop_quality=dbg["best_crop_quality"],
                    torso_visibility_ratio=dbg["torso_visibility_ratio"],
                    face_crop_path=dbg["best_face_crop_path"],
                    best_face_score=dbg["best_face_score"],
                    has_billing=int(billing_count or 0) > 0,
                    billing_count=int(billing_count or 0),
                    person=person_map.get(pid) if pid else None,
                )
            )

        return PaginatedTrackSessionsResponse(
            total_count=total_count,
            page=page,
            size=size,
            tracks=tracks_out,
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

    @staticmethod
    async def list_merged_persons(
        db: AsyncSession,
        page: int = 1,
        size: int = 20,
        search: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> PaginatedIdentityMergeEventsResponse:
        filters = []
        if start_time is not None:
            filters.append(IdentityMergeEvent.merged_at >= start_time)
        if end_time is not None:
            filters.append(IdentityMergeEvent.merged_at <= end_time)
        if source:
            filters.append(IdentityMergeEvent.source == source)
        if search:
            term = f"%{search.strip()}%"
            filters.append(
                or_(
                    IdentityMergeEvent.winner_person_id.cast(String).ilike(term),
                    IdentityMergeEvent.loser_person_id.cast(String).ilike(term),
                    IdentityMergeEvent.source.ilike(term),
                )
            )

        count_q = select(func.count()).select_from(IdentityMergeEvent)
        if filters:
            count_q = count_q.where(*filters)
        total = int((await db.execute(count_q)).scalar() or 0)

        q = (
            select(IdentityMergeEvent, PersonIdentity)
            .outerjoin(
                PersonIdentity,
                PersonIdentity.id == IdentityMergeEvent.winner_person_id,
            )
            .order_by(IdentityMergeEvent.merged_at.desc())
        )
        if filters:
            q = q.where(*filters)
        q = q.offset((page - 1) * size).limit(size)
        rows = (await db.execute(q)).all()

        items: list[IdentityMergeEventItem] = []
        for ev, winner in rows:
            items.append(
                IdentityMergeEventItem(
                    id=ev.id,
                    merged_at=ev.merged_at,
                    source=ev.source,
                    winner_person_id=ev.winner_person_id,
                    loser_person_id=ev.loser_person_id,
                    face_similarity=ev.face_similarity,
                    winner_face_score=ev.winner_face_score,
                    loser_face_score=ev.loser_face_score,
                    winner_first_seen_at=ev.winner_first_seen_at,
                    loser_first_seen_at=ev.loser_first_seen_at,
                    loser_visit_count=ev.loser_visit_count or 0,
                    loser_track_count=ev.loser_track_count or 0,
                    winner_visit_count_before=ev.winner_visit_count_before,
                    winner_face_crop_path=ev.winner_face_crop_path,
                    loser_face_crop_path=ev.loser_face_crop_path,
                    winner_still_exists=winner is not None,
                    winner_is_staff=bool(winner.is_staff) if winner is not None else None,
                    winner_gender=winner.gender if winner is not None else None,
                    winner_visit_count_now=winner.visit_count if winner is not None else None,
                    metadata_json=ev.metadata_json,
                )
            )
        return PaginatedIdentityMergeEventsResponse(
            total_count=total, page=page, size=size, items=items
        )

    @staticmethod
    async def list_fragmented_tracks(
        db: AsyncSession,
        page: int = 1,
        size: int = 20,
        search: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None,
    ) -> PaginatedFragmentedTrackEventsResponse:
        filters = []
        if start_time is not None:
            filters.append(FragmentedTrackEvent.occurred_at >= start_time)
        if end_time is not None:
            filters.append(FragmentedTrackEvent.occurred_at <= end_time)
        if event_type:
            filters.append(FragmentedTrackEvent.event_type == event_type)
        if search:
            term = f"%{search.strip()}%"
            filters.append(
                or_(
                    FragmentedTrackEvent.person_identity_id.cast(String).ilike(term),
                    FragmentedTrackEvent.billing_interaction_id.cast(String).ilike(term),
                    FragmentedTrackEvent.primary_track_session_id.cast(String).ilike(term),
                    FragmentedTrackEvent.event_type.ilike(term),
                    FragmentedTrackEvent.stitch_reason.ilike(term),
                )
            )

        count_q = select(func.count()).select_from(FragmentedTrackEvent)
        if filters:
            count_q = count_q.where(*filters)
        total = int((await db.execute(count_q)).scalar() or 0)

        q = (
            select(FragmentedTrackEvent, Camera, PersonIdentity)
            .outerjoin(Camera, Camera.id == FragmentedTrackEvent.camera_id)
            .outerjoin(
                PersonIdentity,
                PersonIdentity.id == FragmentedTrackEvent.person_identity_id,
            )
            .order_by(FragmentedTrackEvent.occurred_at.desc())
        )
        if filters:
            q = q.where(*filters)
        q = q.offset((page - 1) * size).limit(size)
        rows = (await db.execute(q)).all()

        items: list[FragmentedTrackEventItem] = []
        for ev, cam, person in rows:
            frags = ev.fragment_session_ids
            if isinstance(frags, list):
                frag_ids = [str(x) for x in frags]
            else:
                frag_ids = None
            items.append(
                FragmentedTrackEventItem(
                    id=ev.id,
                    occurred_at=ev.occurred_at,
                    event_type=ev.event_type,
                    person_identity_id=ev.person_identity_id,
                    camera_id=ev.camera_id,
                    camera_name=cam.name if cam is not None else None,
                    zone_id=ev.zone_id,
                    billing_interaction_id=ev.billing_interaction_id,
                    primary_track_session_id=ev.primary_track_session_id,
                    fragment_session_ids=frag_ids,
                    fragment_count=ev.fragment_count or 1,
                    sum_dwell_seconds=ev.sum_dwell_seconds,
                    dwell_threshold=ev.dwell_threshold,
                    stitch_gap_seconds=ev.stitch_gap_seconds,
                    stitch_reason=ev.stitch_reason,
                    body_median=ev.body_median,
                    face_max=ev.face_max,
                    person_is_staff=bool(person.is_staff) if person is not None else None,
                    person_gender=person.gender if person is not None else None,
                    person_face_crop_path=person.face_crop_path if person is not None else None,
                    metadata_json=ev.metadata_json,
                )
            )
        return PaginatedFragmentedTrackEventsResponse(
            total_count=total, page=page, size=size, items=items
        )
