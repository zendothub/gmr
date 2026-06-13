"""Camera worker - per-camera AI processing pipeline.

Pipeline per sampled frame:
  RTSP (latest-frame buffer) -> YOLO detection (full frame; cameras are static)
  -> ByteTrack tracking -> track state update -> zone update

  -> ReID (when stable) -> rule evaluation -> event persistence.

Frames are sampled at the camera's fps_target, NOT at the native stream FPS.

Efficiency notes:
- YOLO / OSNet models are process-wide singletons (shared across workers).
- Inference runs on a small capped thread pool (never oversubscribes CPU/GPU).
- A DB session is opened ONLY when there is something to persist
  (new track sessions, ReID, events, observation batch, stale closes) -
  quiet frames never touch PostgreSQL.
- track_observations are sampled (1 per track every OBS_SAMPLE_SECONDS)
  and inserted in batches.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional

from loguru import logger

from app.config import get_settings
from app.core.db.session import AsyncSessionLocal
from app.modules.ai_runtime.frame_buffer import LatestFrameBuffer
from app.modules.ai_runtime.inference_pool import run_inference
from app.modules.detection.yolo_detector import get_shared_detector
from app.modules.tracking.bytetrack_adapter import ByteTrackAdapter
from app.modules.tracking.track_manager import TrackManager, ActiveTrack
from app.modules.reid.crop_quality import assess_crop_quality
from app.modules.reid.osnet_extractor import get_shared_extractor
from app.modules.reid.identity_decision_engine import IdentityDecisionEngine
from app.modules.rule_engine.rule_evaluator import RuleEvaluator, RuleEvent
from app.utils.image_utils import extract_crop, save_image

from app.utils.time_utils import utc_now

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

        # Components (models are shared across all workers)
        self.frame_buffer = LatestFrameBuffer(camera_config["rtsp_url"])
        self.detector = get_shared_detector(
            model_path=self.settings.YOLO_MODEL_PATH,
            confidence_threshold=self.settings.YOLO_CONFIDENCE_THRESHOLD,
            allowed_classes=self.settings.yolo_allowed_classes_list,
        )
        self.tracker = ByteTrackAdapter()
        self.track_manager = TrackManager(self.camera_id)
        self.rule_evaluator = RuleEvaluator()

        self.reid_extractor = get_shared_extractor(self.settings.OSNET_MODEL_PATH) if self.reid_enabled else None
        self.identity_engine = IdentityDecisionEngine() if self.reid_enabled else None

        # Runtime config (zones/rules). Cameras are static -> no ROI/views.
        self.zones: List[dict] = []

        self.apply_runtime_config(runtime_config)

        # Observation sampling state: local_track_id -> last sample monotonic time
        self._last_obs_time: Dict[int, float] = {}

        # State
        self._task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        self.started_at: Optional[float] = None
        self.frames_processed: int = 0
        self.current_fps: float = 0.0
        self.error_message: Optional[str] = None

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
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Camera worker started: {self.camera_id} (fps_target={self.fps_target})")

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
        self.tracker.reset()
        self._last_obs_time.clear()
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
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self):
        """Main processing loop sampled at fps_target."""
        interval = 1.0 / self.fps_target
        last_frame_ts = 0.0
        fps_window_start = time.time()
        fps_window_count = 0

        while self.is_running:
            loop_start = time.time()
            try:
                frame, frame_ts = self.frame_buffer.get_latest()

                if frame is None or frame_ts <= last_frame_ts:
                    await asyncio.sleep(interval / 2)
                    continue
                last_frame_ts = frame_ts

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
            sleep_for = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep_for)

    async def _process_frame(self, frame):
        """Run the full pipeline on a single frame."""
        frame_shape = frame.shape

        # 1) Detection on the shared capped inference pool (full frame -
        #    cameras are static, so there is no ROI/view to pre-filter against).
        detections = await run_inference(self.detector.detect, frame)

        # 2) Tracking (cheap; run inline)

        track_outputs = self.tracker.update(detections, frame_shape)

        # 4) Update in-memory track state and zones; collect pending DB work
        now_mono = time.monotonic()
        active_tracks: List[ActiveTrack] = []
        new_tracks: List[ActiveTrack] = []
        reid_tracks: List[ActiveTrack] = []
        observations: List[dict] = []

        for to in track_outputs:
            track = self.track_manager.update_track(to.local_track_id, to.bbox, to.confidence)
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
                        "bbox": dict(to.bbox),
                        "confidence": to.confidence,
                        "zone_ids": sorted(track.current_zones),
                    }
                )

        # 5) Rule evaluation (pure in-memory; no DB)
        rule_events = self.rule_evaluator.evaluate(
            self.camera_id, active_tracks, camera_role=self.camera_role
        )

        # 6) Stale tracks (in-memory removal; sessions closed in the same batch)
        stale = self.track_manager.cleanup_stale_tracks()
        for t in stale:
            self._last_obs_time.pop(t.local_track_id, None)

        # 7) Persist - ONLY if there is actual work. Quiet frames skip the DB.
        if new_tracks or reid_tracks or rule_events or observations or stale:
            await self._persist_batch(frame, new_tracks, reid_tracks, rule_events, observations, stale)

    # ------------------------------------------------------------------
    # Persistence (batched)
    # ------------------------------------------------------------------

    async def _persist_batch(
        self,
        frame,
        new_tracks: List[ActiveTrack],
        reid_tracks: List[ActiveTrack],
        rule_events: List[RuleEvent],
        observations: List[dict],
        stale_tracks: List[ActiveTrack],
    ):
        """Open one DB session and persist all pending work in one transaction."""
        async with AsyncSessionLocal() as db:
            try:
                for track in new_tracks:
                    await self._create_track_session(db, track)

                for track in reid_tracks:
                    await self._run_reid(db, frame, track)

                if rule_events:
                    await self._persist_events(db, frame, rule_events)

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
        """Normalize a DetectionResult bbox to dict format."""
        bbox = detection.bbox
        if isinstance(bbox, dict):
            return bbox
        x1, y1, x2, y2 = bbox
        return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}

    async def _create_track_session(self, db, track: ActiveTrack):
        """Persist a new track_session row when a new local_track_id appears."""
        from app.core.db.models.tracking import TrackSession

        session = TrackSession(
            camera_id=self.camera_id,
            local_track_id=track.local_track_id,
            started_at=track.started_at,
            last_seen_at=track.last_seen_at,
            total_frames=track.total_frames,
            is_active=True,
        )
        db.add(session)
        await db.flush()
        track.track_session_id = session.id
        logger.debug(
            f"Track session created: camera={self.camera_id} "
            f"local_track={track.local_track_id} session={session.id}"
        )

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
        from sqlalchemy import update
        from app.core.db.models.tracking import TrackSession

        await db.execute(
            update(TrackSession)
            .where(TrackSession.id == track.track_session_id)
            .values(
                ended_at=utc_now(),
                last_seen_at=track.last_seen_at,
                is_active=False,
                total_frames=track.total_frames,
                avg_confidence=track.avg_confidence,
                stability_score=track.stability_score,
                bbox_history=track.bbox_history[-30:],
            )
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
        """Run ReID: crop -> quality gate -> embedding -> identity decision."""
        track.last_reid_time = utc_now()
        try:
            crop = extract_crop(frame, track.bbox)
            if crop is None:
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

            person_id, is_new = await self.identity_engine.decide_identity(
                db=db,
                embedding=embedding,
                crop_quality_score=quality,
                camera_id=self.camera_id,
                track_stability=track.stability_score,
                crop_path=crop_path,
            )

            track.person_identity_id = person_id
            track.reid_attempted = True

            # Update track session with the resolved identity
            if track.track_session_id:
                from sqlalchemy import update
                from app.core.db.models.tracking import TrackSession

                await db.execute(
                    update(TrackSession)
                    .where(TrackSession.id == track.track_session_id)
                    .values(person_identity_id=person_id, last_seen_at=track.last_seen_at)
                )

            logger.info(
                f"ReID resolved: track={track.local_track_id} person={person_id} "
                f"new={is_new} quality={quality:.2f} camera={self.camera_id}"
            )
        except Exception as e:
            logger.error(f"ReID failed for track {track.local_track_id}: {e}")

    async def _persist_events(self, db, frame, rule_events: List[RuleEvent]):
        """Persist rule engine events (and billing interactions) to PostgreSQL."""
        from app.core.db.models.event import Event
        from app.core.db.models.billing import BillingInteraction

        snapshot_dir = f"{self.settings.STORAGE_ROOT}/{self.settings.SNAPSHOT_DIR}"

        # One snapshot per frame batch is enough - events from the same frame
        # share the same image.
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
                logger.error(f"Failed to persist event {ev.event_type}: {e}")