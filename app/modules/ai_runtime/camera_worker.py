"""Camera worker - per-camera AI processing pipeline.

Pipeline per sampled frame:
  RTSP (latest-frame buffer) -> YOLO track -> camera-view filter
  -> track state update -> zone update -> zone event detection
  -> ReID (when stable, 5-frame accumulation + refinement) -> demographics (InsightFace buffalo_l)
  -> rule evaluation -> event persistence.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Set

import cv2
from loguru import logger
import numpy as np

from app.config import get_settings
from app.core.db.session import AsyncSessionLocal
from app.modules.ai_runtime.frame_buffer import LatestFrameBuffer
from app.modules.ai_runtime.inference_pool import run_inference
from app.modules.detection.yolo_detector import get_shared_detector
from app.modules.tracking.track_manager import TrackManager, ActiveTrack
from app.modules.reid.crop_quality import assess_crop_quality
from app.modules.reid.osnet_extractor import get_shared_extractor
from app.modules.reid.insightface_analyzer import get_shared_analyzer
from app.modules.reid.identity_decision_engine import IdentityDecisionEngine
from app.modules.rule_engine.rule_evaluator import RuleEvaluator, RuleEvent
from app.modules.rule_engine.zone_event_detector import ZoneEventDetector, ZoneEvent
from app.utils.image_utils import extract_crop, save_image

from app.utils.time_utils import utc_now
from app.utils.geometry import polygon_from_json

# How often to sample a track_observation row per track
OBS_SAMPLE_SECONDS = 2.0


class CameraWorker:
    """Runs the AI pipeline for a single camera."""

    def __init__(self, camera_config: dict, runtime_config: dict):
        self.settings = get_settings()
        self.camera_config = camera_config
        self.camera_id: uuid.UUID = camera_config["id"]
        self.camera_role: str = camera_config.get("role", "general")
        self.fps_target: int = max(1, int(camera_config.get("fps_target") or self.settings.DEFAULT_FPS_TARGET))
        self.reid_enabled: bool = bool(camera_config.get("reid_enabled", True))
        self.demographic_enabled: bool = bool(camera_config.get("demographic_enabled", False))

        # Components (models are shared across all workers)
        rotation = camera_config.get("frame_rotation")
        self.frame_buffer = LatestFrameBuffer(camera_config["rtsp_url"], frame_rotation=rotation)
        
        self.detector = get_shared_detector(
            model_path=self.settings.YOLO_MODEL_PATH,
            confidence_threshold=self.settings.YOLO_CONFIDENCE_THRESHOLD,
            allowed_classes=self.settings.yolo_allowed_classes_list,
        )
        self.track_manager = TrackManager(self.camera_id)
        self.rule_evaluator = RuleEvaluator()
        self.zone_event_detector = ZoneEventDetector(self.camera_id)
        
        self.reid_extractor = get_shared_extractor(self.settings.OSNET_MODEL_PATH) if self.reid_enabled else None
        self.identity_engine = IdentityDecisionEngine() if self.reid_enabled else None
        self.insightface_analyzer = get_shared_analyzer(self.settings.INSIGHTFACE_MODEL) if self.demographic_enabled else None

        # Runtime config (zones/rules). Cameras are static -> no ROI/views.
        self.zones: List[dict] = []

        self.apply_runtime_config(runtime_config)

        # view_engine placeholder — cameras are static, no ROI filtering needed.
        self.view_engine = None

        # Observation sampling state: local_track_id -> last sample monotonic time
        self._last_obs_time: Dict[int, float] = {}

        # In-memory ReID embeddings accumulation window state
        self.track_embeddings: Dict[int, List[tuple]] = {}  # local_track_id -> list of (embedding, quality, crop_path)
        self.temporary_person_ids: Set[uuid.UUID] = set()

        # State
        self._task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        self.started_at: Optional[float] = None
        self.frames_processed: int = 0
        self.current_fps: float = 0.0
        self.error_message: Optional[str] = None
        self.last_tracker_reset: float = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the frame buffer and processing loop."""
        if self.is_running:
            logger.warning(f"Camera worker {self.camera_id} already running")
            return
        self.frame_buffer.start()
        self.is_running = True
        self.started_at = time.time()
        self.error_message = None
        self.last_tracker_reset = time.time()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Camera worker started: {self.camera_id} (fps_target={self.fps_target}, rotation={self.camera_config.get('frame_rotation')})")

    async def stop(self):
        """Stop the processing loop and release resources."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Stop capture thread in executor (it joins a thread)
        await asyncio.to_thread(self.frame_buffer.stop)
        # Close any open track sessions
        await self._close_all_track_sessions()
        self.track_manager.reset()
        self._last_obs_time.clear()
        self.track_embeddings.clear()
        self.temporary_person_ids.clear()

        # Close GUI window if open
        if getattr(self, "_gui_available", False):
            try:
                window_title = f"Camera Stream - {self.camera_id}"
                cv2.destroyWindow(window_title)
            except Exception as e:
                logger.debug(f"Error destroying GUI window: {e}")

        logger.info(f"Camera worker stopped: {self.camera_id}")

    def apply_runtime_config(self, runtime_config: dict):
        """Apply (or re-apply) runtime configuration loaded from PostgreSQL."""
        self.zones = runtime_config.get("zones", [])
        self.rule_evaluator.cache.load(

            runtime_config.get("rules", []),
            runtime_config.get("zones_by_id", {}),
        )
        logger.info(
            f"Camera worker {self.camera_id} config applied: "
            f"{len(self.zones)} zones, {len(runtime_config.get('rules', []))} rules"
        )

    def get_status(self) -> dict:
        """Return worker health status."""
        frame, frame_ts = self.frame_buffer.get_latest()
        return {
            "camera_id": str(self.camera_id),
            "is_running": self.is_running,
            "is_streaming": self.frame_buffer.is_connected,
            "current_fps": round(self.current_fps, 2),
            "fps_target": self.fps_target,
            "frames_processed": self.frames_processed,
            "active_tracks": len(self.track_manager.tracks),
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else None,
            "error_message": self.error_message or self.frame_buffer.last_error,
            "last_frame_time": frame_ts if frame_ts > 0 else None,
            "reconnect_count": self.frame_buffer.reconnect_count,
            "demographics_enabled": self.demographic_enabled,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self):
        """Main processing loop sampled at fps_target."""
        interval = 1.0 / self.fps_target
        last_frame_ts = 0.0
        last_success_ts = time.time()
        fps_window_start = time.time()
        fps_window_count = 0

        while self.is_running:
            loop_start = time.time()
            try:
                frame, frame_ts = self.frame_buffer.get_latest()

                # Stream watchdog check:
                if time.time() - last_success_ts > self.settings.WORKER_WATCHDOG_TIMEOUT:
                    logger.warning(f"Watchdog: No new frame from stream {self.camera_id} for 30s. Resetting frame buffer.")
                    await asyncio.to_thread(self.frame_buffer.reset)
                    last_success_ts = time.time()
                    continue

                if frame is None or frame_ts <= last_frame_ts:
                    await asyncio.sleep(interval / 2)
                    continue

                last_frame_ts = frame_ts
                last_success_ts = time.time()

                # Periodic tracker reset check (6 hours)
                reset_interval = self.settings.WORKER_TRACKER_RESET_HOURS * 3600
                if time.time() - self.last_tracker_reset > reset_interval:
                    logger.info(f"Periodic reset: Resetting tracker state for camera {self.camera_id}")
                    await self._close_all_track_sessions()
                    self.track_manager.reset()
                    await run_inference(self.detector.reset_tracker)
                    self.last_tracker_reset = time.time()

                await self._process_frame(frame)

                self.frames_processed += 1
                fps_window_count += 1
                now = time.time()
                if now - fps_window_start >= 5.0:
                    self.current_fps = fps_window_count / (now - fps_window_start)
                    fps_window_start = now
                    fps_window_count = 0

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.error_message = str(e)
                logger.error(f"Camera worker {self.camera_id} frame processing error: {e}")

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_for = max(0.001, interval - elapsed)
            await asyncio.sleep(sleep_for)

    async def _process_frame(self, frame):
        """Run the full pipeline on a single frame."""
        # 1) Unified YOLO Detection & Tracking on the shared inference pool
        tracked_detections = await run_inference(self.detector.track, frame)

        # 2) Update in-memory track state and zones; collect pending DB work
        #    (Cameras are static — no ROI filtering needed; detection runs on full frame)
        now_mono = time.monotonic()
        active_tracks: List[ActiveTrack] = []
        new_tracks: List[ActiveTrack] = []
        reid_tracks: List[ActiveTrack] = []
        observations: List[dict] = []

        for td in tracked_detections:
            track = self.track_manager.update_track(td.track_id, td.bbox, td.confidence)
            self.track_manager.update_zones(track, self.zones)
            active_tracks.append(track)

            if track.track_session_id is None:
                new_tracks.append(track)

            if (
                self.reid_enabled
                and self.reid_extractor
                and self.identity_engine
                and track.should_run_reid()
            ):
                reid_tracks.append(track)

            # Sampled observation (1 per track every OBS_SAMPLE_SECONDS)
            last_obs = self._last_obs_time.get(track.local_track_id, 0.0)
            if now_mono - last_obs >= OBS_SAMPLE_SECONDS:
                self._last_obs_time[track.local_track_id] = now_mono
                observations.append(
                    {
                        "track": track,
                        "bbox": dict(td.bbox),
                        "confidence": td.confidence,
                        "zone_ids": sorted(track.current_zones),
                    }
                )

        # 4) Automatic zone event detection
        zone_events = self.zone_event_detector.detect(active_tracks)

        # 5) Rule evaluation (pure in-memory; no DB)
        rule_events = self.rule_evaluator.evaluate(
            self.camera_id, active_tracks, camera_role=self.camera_role
        )

        # 6) Stale tracks (in-memory removal; sessions closed in the same batch)
        stale = self.track_manager.cleanup_stale_tracks()
        for t in stale:
            self._last_obs_time.pop(t.local_track_id, None)
            self.track_embeddings.pop(t.local_track_id, None)

        # 7) Persist - ONLY if there is actual work. Quiet frames skip the DB.
        if new_tracks or reid_tracks or rule_events or zone_events or observations or stale:
            await self._persist_batch(frame, new_tracks, reid_tracks, rule_events, zone_events, observations, stale)

        # 8) Optional GUI display
        self._display_gui_frame(frame, active_tracks)

    # ------------------------------------------------------------------
    # Persistence (batched)
    # ------------------------------------------------------------------

    async def _persist_batch(
        self,
        frame,
        new_tracks: List[ActiveTrack],
        reid_tracks: List[ActiveTrack],
        rule_events: List[RuleEvent],
        zone_events: List[ZoneEvent],
        observations: List[dict],
        stale_tracks: List[ActiveTrack],
    ):
        """Open one DB session and persist all pending work in one transaction."""
        async with AsyncSessionLocal() as db:
            try:
                for track in new_tracks:
                    await self._create_track_session(db, frame, track)

                for track in reid_tracks:
                    await self._run_reid(db, frame, track)

                if rule_events or zone_events:
                    await self._persist_events(db, frame, rule_events, zone_events)

                if observations:
                    await self._persist_observations(db, observations)

                for track in stale_tracks:
                    await self._close_track_session(db, track)

                await db.commit()
            except Exception:
                await db.rollback()
                raise

    @staticmethod
    def _det_bbox(detection) -> dict:
        """Normalize a DetectionResult / TrackedDetection bbox to dict format."""
        bbox = detection.bbox
        if isinstance(bbox, dict):
            return bbox
        x1, y1, x2, y2 = bbox
        return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}

    async def _create_track_session(self, db, frame, track: ActiveTrack):
        """Persist a new track_session row when a new local_track_id appears."""
        from app.core.db.models.tracking import TrackSession

        # Extract initial crop
        crop_path = None
        crop = extract_crop(frame, track.bbox)
        if crop is not None and crop.size > 0:
            crop_dir = f"{self.settings.STORAGE_ROOT}/{self.settings.CROP_DIR}"
            crop_path = save_image(crop, crop_dir, prefix=f"crop_{self.camera_id}")
            track.best_crop_path = crop_path
            # Initial quality is 0, will be overwritten if ReID runs

        session = TrackSession(
            camera_id=self.camera_id,
            local_track_id=track.local_track_id,
            started_at=track.started_at,
            last_seen_at=track.last_seen_at,
            total_frames=track.total_frames,
            best_crop_path=crop_path,
            is_active=True,
        )
        db.add(session)
        await db.flush()
        track.track_session_id = session.id
        logger.debug(
            f"Track session created: camera={self.camera_id} "
            f"local_track={track.local_track_id} session={session.id}"
        )
        
        # Fire person_entered_view immediately (person_identity_id is None initially)
        from app.core.db.models.event import Event, EventSeverity
        enter_event = Event(
            camera_id=self.camera_id,
            person_identity_id=None,
            track_session_id=track.track_session_id,
            event_type="person_entered_view",
            severity=EventSeverity.INFO,
            description="Person entered view.",
            snapshot_path=crop_path,
            metadata_json={"local_track_id": track.local_track_id},
            occurred_at=track.started_at
        )
        db.add(enter_event)

    async def _persist_observations(self, db, observations: List[dict]):
        """Insert sampled track observations in a batch."""
        from app.core.db.models.tracking import TrackObservation

        for obs in observations:
            track: ActiveTrack = obs["track"]
            if not track.track_session_id:
                continue
            db.add(
                TrackObservation(
                    track_session_id=track.track_session_id,
                    timestamp=utc_now(),
                    bbox=obs["bbox"],
                    confidence=obs["confidence"],
                    zone_ids=obs["zone_ids"],
                )
            )

    async def _close_track_session(self, db, track: ActiveTrack):
        """Mark a track session as ended in PostgreSQL."""
        if not track.track_session_id:
            return
        from sqlalchemy import update, select
        from app.core.db.models.tracking import TrackSession
        from app.core.db.models.person import PersonIdentity

        # 1. Update TrackSession
        session_values = {
            "ended_at": utc_now(),
            "last_seen_at": track.last_seen_at,
            "is_active": False,
            "total_frames": track.total_frames,
            "avg_confidence": track.avg_confidence,
            "stability_score": track.stability_score,
            "bbox_history": track.bbox_history[-30:],
            "best_crop_path": track.best_crop_path,
        }
        
        # If demographics were captured, store them in the TrackSession
        if track.best_demographics:
            session_values["gender"] = track.best_demographics["gender"]
            session_values["age_group"] = track.best_demographics["age_group"]

        await db.execute(
            update(TrackSession)
            .where(TrackSession.id == track.track_session_id)
            .values(**session_values)
        )
        
        # 1.5 Fire person_left_view event
        from app.core.db.models.event import Event, EventSeverity
        exit_event = Event(
            camera_id=self.camera_id,
            person_identity_id=track.person_identity_id,
            track_session_id=track.track_session_id,
            event_type="person_left_view",
            severity=EventSeverity.INFO,
            description=f"Person {str(track.person_identity_id)[:8] if track.person_identity_id else 'unknown'} left view.",
            occurred_at=utc_now(),
            metadata_json={"total_frames": track.total_frames, "duration_seconds": track.last_seen_at.timestamp() - track.started_at.timestamp()}
        )
        db.add(exit_event)

        # 2. Update PersonIdentity demographics with the highest quality face crop
        if track.person_identity_id and track.best_demographics:
            result = await db.execute(
                select(PersonIdentity).where(PersonIdentity.id == track.person_identity_id)
            )
            person = result.scalar_one_or_none()
            if person:
                current_score = person.best_face_score or 0.0
                new_score = track.best_demographics["face_score"]
                if new_score > current_score:
                    person.gender = track.best_demographics["gender"]
                    person.age_group = track.best_demographics["age_group"]
                    person.estimated_age = track.best_demographics["age"]
                    person.best_face_score = new_score
                    person.face_crop_path = track.best_demographics["face_crop_path"]
                    logger.info(
                        f"Updated PersonIdentity {track.person_identity_id} demographics: "
                        f"gender={person.gender}, age_group={person.age_group}, age={person.estimated_age} (score={new_score:.3f})"
                    )

    async def _close_all_track_sessions(self):
        """Close all open track sessions on shutdown."""
        try:
            async with AsyncSessionLocal() as db:
                for track in self.track_manager.get_active_tracks():
                    await self._close_track_session(db, track)
                await db.commit()
        except Exception as e:
            logger.warning(f"Could not close track sessions for camera {self.camera_id}: {e}")

    async def _run_reid(self, db, frame, track: ActiveTrack):
        """Run ReID pipeline: crop -> quality -> embedding -> accumulation -> decision."""
        track.last_reid_time = utc_now()
        was_resolved = track.reid_resolved
        try:
            crop = extract_crop(frame, track.bbox)
            if crop is None or crop.size == 0:
                return

            quality = assess_crop_quality(crop)
            if quality < self.settings.REID_CROP_QUALITY_THRESHOLD:
                logger.debug(
                    f"ReID crop rejected (quality={quality:.2f}) "
                    f"track={track.local_track_id} camera={self.camera_id}"
                )
                return

            embedding = await run_inference(self.reid_extractor.extract, crop)
            if embedding is None:
                return

            # Save crop for audit / debugging
            crop_dir = f"{self.settings.STORAGE_ROOT}/{self.settings.CROP_DIR}"
            crop_path = save_image(crop, crop_dir, prefix=f"crop_{self.camera_id}")

            if quality > track.best_crop_quality:
                track.best_crop_quality = quality
                track.best_crop_path = crop_path

            # Run demographics extraction if enabled and budget allows
            face_embedding = None
            face_score = 0.0
            face_crop_path = None
            
            if self.demographic_enabled and self.insightface_analyzer and track.face_analysis_count < self.settings.INSIGHTFACE_MAX_ATTEMPTS:
                face_result = await run_inference(self.insightface_analyzer.analyze, crop)
                if face_result:
                    track.face_analysis_count += 1
                    face_embedding = face_result.embedding
                    face_score = face_result.face_score
                    
                    if face_result.face_crop is not None:
                        # Save face crop
                        face_crop_dir = f"{self.settings.STORAGE_ROOT}/{self.settings.CROP_DIR}"
                        face_crop_path = save_image(face_result.face_crop, face_crop_dir, prefix=f"face_{self.camera_id}")

                    # Keep the best face score demographics for display and TrackSession updating
                    if (track.best_demographics is None or 
                        face_score > track.best_demographics.get("face_score", 0.0)):
                        track.best_demographics = {
                            "age": face_result.age,
                            "gender": face_result.gender,
                            "age_group": face_result.age_group,
                            "face_score": face_score,
                            "face_crop_path": face_crop_path,
                        }
                        logger.debug(f"Demographics updated for track {track.local_track_id}: {track.best_demographics}")

            # Accumulate embedding
            accum_list = self.track_embeddings.setdefault(track.local_track_id, [])
            accum_list.append((embedding, quality, crop_path, face_embedding, face_score, face_crop_path))
            track.reid_frame_count += 1

            # Decision execution on reaching the window size (5 frames)
            if len(accum_list) == self.settings.REID_ACCUMULATION_FRAMES:
                embeddings = [item[0] for item in accum_list]
                mean_embedding = np.mean(embeddings, axis=0)
                mean_norm = np.linalg.norm(mean_embedding)
                if mean_norm > 0:
                    mean_embedding = mean_embedding / mean_norm

                best_crop_item = max(accum_list, key=lambda item: item[1])
                best_quality = best_crop_item[1]
                best_crop_path = best_crop_item[2]
                
                # Find the best face in this accumulation window
                best_face_item = max(accum_list, key=lambda item: item[4])  # Max by face_score
                best_face_embedding = best_face_item[3]
                best_face_score = best_face_item[4]
                best_face_crop_path = best_face_item[5]

                # Delete the other unused crops in this window from disk
                import os
                for item in accum_list:
                    other_path = item[2]
                    other_face_path = item[5]
                    
                    try:
                        if other_path and other_path != best_crop_path and os.path.exists(other_path):
                            os.remove(other_path)
                        if other_face_path and other_face_path != best_face_crop_path and os.path.exists(other_face_path):
                            os.remove(other_face_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove unused crop: {e}")

                accum_list.clear()

                is_temp = track.person_identity_id in self.temporary_person_ids

                person_id, score, is_confident, is_new, prune_old_id = await self.identity_engine.decide_identity(
                    db=db,
                    mean_embedding=mean_embedding,
                    camera_id=self.camera_id,
                    crop_quality_score=best_quality,
                    crop_path=best_crop_path,
                    current_person_id=track.person_identity_id,
                    previous_score=track.reid_score,
                    is_temporary=is_temp,
                    face_embedding=best_face_embedding,
                    face_score=best_face_score,
                    face_crop_path=best_face_crop_path,
                )

                if is_new:
                    self.temporary_person_ids.add(person_id)
                if prune_old_id:
                    self.temporary_person_ids.discard(prune_old_id)

                track.person_identity_id = person_id
                track.reid_score = score
                track.reid_confident = is_confident
                track.reid_resolved = True
                track.reid_attempted = True

                # Update current track session in PostgreSQL
                if track.track_session_id:
                    from sqlalchemy import update
                    from app.core.db.models.tracking import TrackSession
                    from app.core.db.models.event import Event

                    await db.execute(
                        update(TrackSession)
                        .where(TrackSession.id == track.track_session_id)
                        .values(person_identity_id=person_id, last_seen_at=track.last_seen_at)
                    )
                    
                    # Update the person_entered_view event with the resolved identity
                    if not was_resolved:
                        await db.execute(
                            update(Event)
                            .where(Event.track_session_id == track.track_session_id)
                            .where(Event.event_type == "person_entered_view")
                            .values(person_identity_id=person_id, description=f"Person {str(person_id)[:8]} entered view.")
                        )

                logger.info(
                    f"ReID resolved (count={track.reid_frame_count}): track={track.local_track_id} "
                    f"person={person_id} score={score:.2f} confident={is_confident}"
                )

        except Exception as e:
            logger.error(f"ReID failed for track {track.local_track_id}: {e}")

    async def _persist_events(self, db, frame, rule_events: List[RuleEvent], zone_events: List[ZoneEvent]):
        """Persist rule engine events and automatic zone events to PostgreSQL."""
        from app.core.db.models.event import Event, EventSeverity
        from app.core.db.models.billing import BillingInteraction

        snapshot_dir = f"{self.settings.STORAGE_ROOT}/{self.settings.SNAPSHOT_DIR}"
        snapshot_path = save_image(frame, snapshot_dir, prefix=f"event_{self.camera_id}")

        for ev in rule_events:
            try:
                event = Event(
                    camera_id=ev.camera_id,
                    rule_id=ev.rule_id,
                    zone_id=ev.zone_id,
                    person_identity_id=ev.person_identity_id,
                    track_session_id=ev.track_session_id,
                    event_type=ev.event_type,
                    severity=ev.severity,
                    description=ev.description,
                    metadata_json=ev.metadata,
                    snapshot_path=snapshot_path,
                    occurred_at=utc_now(),
                )
                db.add(event)

                # Billing interactions get an additional structured record
                if ev.rule_type == "billing_interaction":
                    interaction = BillingInteraction(
                        camera_id=ev.camera_id,
                        person_identity_id=ev.person_identity_id,
                        track_session_id=ev.track_session_id,
                        zone_id=ev.zone_id,
                        entered_at=utc_now(),
                        dwell_seconds=ev.metadata.get("dwell_seconds"),
                        interaction_type="billing_counter",
                        metadata_json=ev.metadata,
                    )
                    db.add(interaction)

                logger.info(
                    f"Event saved: {ev.event_type} camera={ev.camera_id} "
                    f"severity={ev.severity} track_session={ev.track_session_id}"
                )
            except Exception as e:
                logger.error(f"Failed to persist rule event {ev.event_type}: {e}")

        for ev in zone_events:
            try:
                event = Event(
                    camera_id=self.camera_id,
                    rule_id=None,
                    zone_id=ev.zone_id,
                    person_identity_id=ev.person_identity_id,
                    track_session_id=ev.track_session_id,
                    event_type=ev.event_type,
                    severity=EventSeverity.INFO,
                    description=ev.description,
                    metadata_json=ev.metadata,
                    snapshot_path=snapshot_path,
                    occurred_at=utc_now(),
                )
                db.add(event)
                logger.info(f"Auto Zone Event saved: {ev.event_type} camera={self.camera_id} zone={ev.zone_id}")
            except Exception as e:
                logger.error(f"Failed to persist automatic zone event {ev.event_type}: {e}")

    # ------------------------------------------------------------------
    # GUI Display
    # ------------------------------------------------------------------

    def _check_gui_availability(self) -> bool:
        """Check if a GUI/display environment is available."""
        if not self.settings.RUNTIME_SHOW_GUI:
            return False

        if not hasattr(self, "_gui_available"):
            try:
                # Try to create and destroy a test window to ensure GUI is supported
                cv2.namedWindow("GUI_Check", cv2.WINDOW_AUTOSIZE)
                cv2.destroyWindow("GUI_Check")
                self._gui_available = True
                logger.info(f"GUI display is available. Opening video stream window for camera {self.camera_id}.")
            except Exception as e:
                logger.debug(f"GUI display not available (headless environment): {e}")
                self._gui_available = False

        return self._gui_available

    def _display_gui_frame(self, frame, active_tracks: List[ActiveTrack]):
        """Helper to draw overlays and display the video stream frame in an OpenCV window if GUI is present/enabled."""
        if not self._check_gui_availability():
            return

        try:
            display_frame = frame.copy()
            height, width, _ = display_frame.shape

            # 1. Draw zones
            for zone in self.zones:
                zone_name = zone.get("name", "Zone")
                poly = zone.get("polygon")
                poly_points = polygon_from_json(poly)
                if poly_points:
                    pts = np.array([(int(x * width), int(y * height)) for x, y in poly_points], dtype=np.int32)
                    
                    # Semi-transparent overlay for the zone
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green fill
                    cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)
                    # Bounding line
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    
                    # Draw Zone Name at the first point
                    first_pt = pts[0]
                    cv2.putText(
                        display_frame,
                        zone_name,
                        (first_pt[0], first_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # 2. Draw tracks
            for track in active_tracks:
                if not track.bbox:
                    continue
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, [track.bbox["x1"], track.bbox["y1"], track.bbox["x2"], track.bbox["y2"]])
                color = (255, 0, 0)  # BGR format: Blue for default track
                if track.reid_confident:
                    color = (0, 255, 255)  # Cyan if ReID resolved & confident
                elif track.person_identity_id:
                    color = (255, 255, 0)  # Yellow if ReID resolved but not confident
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Construct label
                label_parts = [f"ID: {track.local_track_id}"]
                if track.person_identity_id:
                    person_str = str(track.person_identity_id)[:8]
                    label_parts.append(f"Person: {person_str}")
                    if track.reid_confident:
                        label_parts.append("(conf)")
                
                if track.best_demographics:
                    gender = track.best_demographics.get("gender")
                    age_group = track.best_demographics.get("age_group")
                    if gender and age_group:
                        label_parts.append(f"{gender}, {age_group}")
                
                label = " | ".join(label_parts)
                
                # Draw label background
                (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(display_frame, (x1, y1 - lbl_h - 6), (x1 + lbl_w + 4, y1), color, -1)
                
                # Draw text
                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0) if color != (255, 0, 0) else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Show frame
            window_title = f"Camera Stream - {self.camera_id}"
            cv2.imshow(window_title, display_frame)
            cv2.waitKey(1)
        except Exception as e:
            logger.warning(f"Error in OpenCV display GUI: {e}")
            self._gui_available = False