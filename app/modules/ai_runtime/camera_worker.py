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
from app.core.db.models.person import PersonIdentity
from app.modules.ai_runtime.frame_buffer import LatestFrameBuffer
from app.modules.ai_runtime.inference_pool import run_inference
from app.modules.ai_runtime.stream_broadcaster import StreamBroadcaster
from app.modules.detection.yolo_detector import get_camera_detector, get_shared_pose_model
from app.modules.tracking.track_manager import TrackManager, ActiveTrack
from app.modules.reid.crop_quality import assess_crop_quality
from app.modules.reid.osnet_extractor import get_shared_extractor
from app.modules.reid.insightface_analyzer import get_shared_analyzer
from app.modules.reid.siglip2_analyzer import get_shared_siglip2
from app.modules.reid.identity_decision_engine import IdentityDecisionEngine
from app.modules.rule_engine.rule_evaluator import RuleEvaluator, RuleEvent
from app.modules.rule_engine.zone_event_detector import ZoneEventDetector, ZoneEvent
from app.utils.image_utils import extract_crop, save_image, save_image_async, resize_pad_square

from app.utils.time_utils import utc_now
from app.utils.geometry import polygon_from_json, bbox_iou, face_area_in_body_frac

# How often to sample a track_observation row per track
OBS_SAMPLE_SECONDS = 2.0


class CameraWorker:
    """Runs the AI pipeline for a single camera."""

    def __init__(self, camera_config: dict, runtime_config: dict):
        self.settings = get_settings()
        self.camera_config = camera_config
        self.camera_id: uuid.UUID = camera_config["id"]
        self.fps_target: int = max(1, int(camera_config.get("fps_target") or self.settings.DEFAULT_FPS_TARGET))
        self.reid_enabled: bool = bool(camera_config.get("reid_enabled", True))
        self.demographic_enabled: bool = bool(camera_config.get("demographic_enabled", True))

        # Components — YOLO detector is per-camera (not shared) to isolate ByteTrack state
        rotation = camera_config.get("frame_rotation")
        self.frame_buffer = LatestFrameBuffer(camera_config["rtsp_url"], frame_rotation=rotation)
        
        self.detector = get_camera_detector(
            camera_id=str(self.camera_id),
            model_path=self.settings.YOLO_MODEL_PATH,
            confidence_threshold=self.settings.YOLO_CONFIDENCE_THRESHOLD,
            allowed_classes=self.settings.yolo_allowed_classes_list,
        )
        self.track_manager = TrackManager(self.camera_id)
        self.rule_evaluator = RuleEvaluator()
        self.zone_event_detector = ZoneEventDetector(self.camera_id)
        
        self.reid_extractor = get_shared_extractor(self.settings.OSNET_MODEL_PATH) if self.reid_enabled else None
        self.yolo_pose = get_shared_pose_model() if self.reid_enabled else None
        self.identity_engine = IdentityDecisionEngine() if self.reid_enabled else None
        self.insightface_analyzer = get_shared_analyzer(self.settings.INSIGHTFACE_MODEL) if self.demographic_enabled else None
        self.siglip2_analyzer = get_shared_siglip2() if self.demographic_enabled else None

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

        # Stream burn-in (bounding boxes on the live stream)
        self.burnin_enabled: bool = bool(camera_config.get("burnin_enabled", False))
        self.stream_broadcaster: Optional[StreamBroadcaster] = None
        # Shared mutable state: latest YOLO bounding boxes for the broadcaster.
        # Populated each time _process_frame runs tracking. Format:
        #   [{"x1": int, "y1": int, "x2": int, "y2": int}, ...]
        self.latest_tracks: List[dict] = []

        # State
        self._task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        self.started_at: Optional[float] = None
        self.frames_processed: int = 0
        self.current_fps: float = 0.0
        self.error_message: Optional[str] = None
        self.last_tracker_reset: float = time.time()

    # ------------------------------------------------------------------
    # Stream burn-in helpers
    # ------------------------------------------------------------------

    def _start_broadcaster(self) -> None:
        """Start the StreamBroadcaster that pipes annotated frames to MediaMTX.

        Resolution is resolved from the *actual* first frame so FFmpeg is told
        the correct stride.  Falling back to the camera-config hint avoids an
        infinite wait if the buffer hasn't filled yet at start time.
        """
        # Try to get the real frame dimensions from the live buffer first
        frame, _ = self.frame_buffer.get_latest()
        if frame is not None:
            height, width = frame.shape[:2]
        else:
            # Fall back to camera config hint
            resolution = self.camera_config.get("resolution", "1920x1080")
            try:
                w_str, h_str = resolution.split("x")
                width, height = int(w_str), int(h_str)
            except (ValueError, AttributeError):
                width, height = 1920, 1080

        self.stream_broadcaster = StreamBroadcaster(
            frame_buffer=self.frame_buffer,
            camera_id=self.camera_id,
            width=width,
            height=height,
            fps=self.settings.STREAM_BURNIN_FPS,
        )
        # Share the *same* list object so updates in _process_frame are visible
        # to the broadcaster without any copy.  Use .clear() + extend() to
        # mutate it in-place instead of reassigning the reference.
        self.stream_broadcaster.latest_tracks = self.latest_tracks
        # Share zone definitions so the broadcaster can draw them
        self.stream_broadcaster.latest_zones = self.zones
        self.stream_broadcaster.start()

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
        # Set _last_success_ts in start() so the watchdog in _run_loop()
        # doesn't immediately trigger when a previous run left a stale ts.
        self._last_success_ts = time.time()
        self._task = asyncio.create_task(self._run_loop())

        # Start stream burn-in broadcaster if enabled for this camera
        if self.burnin_enabled:
            self._start_broadcaster()
            logger.info(f"Stream burn-in enabled for camera {self.camera_id}")

        logger.info(
            f"Camera worker started: {self.camera_id} "
            f"(fps_target={self.fps_target}, rotation={self.camera_config.get('frame_rotation')}, "
            f"body_crop_padding={self.settings.BODY_CROP_PADDING_PCT}, "
            f"hungarian_face={self.settings.ENABLE_HUNGARIAN_FACE_ASSIGN}, "
            f"skip_body_when_occluded={self.settings.SKIP_BODY_REID_WHEN_OCCLUDED})"
        )

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
        # Stop stream burn-in broadcaster if running
        if self.stream_broadcaster:
            self.stream_broadcaster.stop()
            self.stream_broadcaster = None

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

    def _score_face_for_track(self, face: dict, body_bbox: dict, track: "ActiveTrack") -> Optional[float]:
        """Score a face candidate for a body track.

        Returns the composite score if the face qualifies under geometry gates,
        or None if rejected.  Immature tracks (good_face_count < 2) require the
        face centre in the upper body and ≥ FACE_ASSIGN_MIN_OVERLAP of the face
        box inside the body.  Continuity bonus is disabled while occluded.

        Scoring weights:
          - Without continuity: 0.50 × size + 0.50 × centre
          - With continuity:    0.35 × size + 0.30 × centre + 0.35 × continuity
        """
        fb = face["bbox"]
        fx_cx = (fb["x1"] + fb["x2"]) / 2.0
        fy_cy = (fb["y1"] + fb["y2"]) / 2.0

        bx = body_bbox["x1"]
        by = body_bbox["y1"]
        bx2 = body_bbox["x2"]
        by2 = body_bbox["y2"]

        # Membership check — use ORIGINAL bbox, no expansion.
        # The 15% expansion caused adjacent persons' faces to qualify when
        # people stood shoulder-to-shoulder.
        if not (bx <= fx_cx <= bx2):
            return None
        if not (by <= fy_cy <= by2):
            return None

        bw = bx2 - bx
        bh = by2 - by
        if bw <= 0 or bh <= 0:
            return None

        immature = track.good_face_count < 2
        if immature:
            fy_rel = (fy_cy - by) / bh
            if fy_rel > self.settings.FACE_ASSIGN_UPPER_BODY_FRAC:
                return None
            overlap = face_area_in_body_frac(fb, body_bbox)
            if overlap < self.settings.FACE_ASSIGN_MIN_OVERLAP:
                return None

        body_cx = (bx + bx2) / 2.0
        fw = fb["x2"] - fb["x1"]
        fh = fb["y2"] - fb["y1"]
        face_area = fw * fh
        body_area = max(bw * bh, 1.0)

        size_score = min(1.0, face_area / (body_area * 0.03))
        centre_dev = abs(fx_cx - body_cx) / max(bw / 2.0, 1.0)
        centre_score = max(0.0, 1.0 - centre_dev)

        use_continuity = (
            track.last_face_center is not None
            and not track.is_occluded
        )
        if use_continuity:
            _prev_cx, _prev_cy = track.last_face_center
            _dist = ((fx_cx - _prev_cx) ** 2 + (fy_cy - _prev_cy) ** 2) ** 0.5
            continuity_score = max(0.0, 1.0 - _dist / 300.0)
            geo = 0.35 * size_score + 0.30 * centre_score + 0.35 * continuity_score
        else:
            geo = 0.50 * size_score + 0.50 * centre_score

        score = float(face["det_score"]) * geo
        if immature and score < self.settings.FACE_ASSIGN_MIN_SCORE_IMMATURE:
            return None
        return score

    def _mark_occluded_tracks(self, active_tracks: List[ActiveTrack]) -> None:
        """Mark tracks whose body bboxes overlap above OCCLUSION_IOU_THRESHOLD."""
        for t in active_tracks:
            t.is_occluded = False
        n = len(active_tracks)
        if n < 2:
            return
        thr = self.settings.OCCLUSION_IOU_THRESHOLD
        occluded_ids: List[int] = []
        for i in range(n):
            bi = active_tracks[i].bbox
            if bi is None:
                continue
            for j in range(i + 1, n):
                bj = active_tracks[j].bbox
                if bj is None:
                    continue
                if bbox_iou(bi, bj) >= thr:
                    active_tracks[i].is_occluded = True
                    active_tracks[j].is_occluded = True
        for t in active_tracks:
            if t.is_occluded:
                occluded_ids.append(t.local_track_id)
        if occluded_ids:
            logger.debug(
                f"FaceAssign: occluded tracks={occluded_ids} "
                f"cam={self.camera_id} iou_thr={thr}"
            )

    def _assign_faces_to_tracks(self, all_faces: list, active_tracks: List[ActiveTrack]) -> None:
        """Global face→track assignment with harden gates + Hungarian / greedy."""
        for track in active_tracks:
            track._matched_face = None

        if not all_faces or not active_tracks:
            return

        candidates: List[tuple] = []  # (score, track, face_idx)
        for track in active_tracks:
            if track.bbox is None:
                continue
            for fi, f in enumerate(all_faces):
                score = self._score_face_for_track(f, track.bbox, track)
                if score is not None:
                    candidates.append((score, track, fi))

        if not candidates:
            return

        # Ambiguous reject: if two tracks compete for the same face within ratio, drop face entirely
        face_scores: Dict[int, List[tuple]] = {}
        for score, track, fi in candidates:
            face_scores.setdefault(fi, []).append((score, track))
        ambiguous_faces: Set[int] = set()
        ratio_thr = self.settings.FACE_ASSIGN_AMBIGUITY_RATIO
        for fi, pairs in face_scores.items():
            if len(pairs) < 2:
                continue
            pairs.sort(key=lambda x: x[0], reverse=True)
            best_s, second_s = pairs[0][0], pairs[1][0]
            if best_s > 0 and (second_s / best_s) >= ratio_thr:
                ambiguous_faces.add(fi)
                logger.info(
                    f"FaceAssign: rejected face (ambiguous) cam={self.camera_id} "
                    f"face_idx={fi} scores={[round(p[0], 3) for p in pairs[:3]]} "
                    f"tracks={[p[1].local_track_id for p in pairs[:3]]}"
                )

        candidates = [c for c in candidates if c[2] not in ambiguous_faces]
        if not candidates:
            return

        assigned: Dict[int, tuple] = {}  # track_local_id -> (score, face_idx)

        if self.settings.ENABLE_HUNGARIAN_FACE_ASSIGN:
            track_ids = sorted({t.local_track_id for _, t, _ in candidates})
            face_ids = sorted({fi for _, _, fi in candidates})
            t_index = {tid: i for i, tid in enumerate(track_ids)}
            f_index = {fi: j for j, fi in enumerate(face_ids)}
            n_t, n_f = len(track_ids), len(face_ids)
            cost = np.full((n_t, n_f), 1e6, dtype=np.float64)
            score_lookup: Dict[tuple, float] = {}
            track_by_id = {t.local_track_id: t for t in active_tracks}
            for score, track, fi in candidates:
                i, j = t_index[track.local_track_id], f_index[fi]
                cost[i, j] = -score
                score_lookup[(track.local_track_id, fi)] = score
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] >= 1e5:
                        continue
                    tid = track_ids[r]
                    fi = face_ids[c]
                    assigned[tid] = (score_lookup[(tid, fi)], fi)
            except Exception as e:
                logger.warning(f"FaceAssign: Hungarian failed ({e}), falling back to greedy")
                assigned = {}

        if not assigned:
            # Greedy fallback (also used when Hungarian disabled)
            all_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
            claimed_faces: Set[int] = set()
            assigned_tracks: Set[int] = set()
            for score, track, fi in all_candidates:
                if fi in claimed_faces or track.local_track_id in assigned_tracks:
                    continue
                assigned[track.local_track_id] = (score, fi)
                claimed_faces.add(fi)
                assigned_tracks.add(track.local_track_id)

        for track in active_tracks:
            hit = assigned.get(track.local_track_id)
            if hit is None:
                continue
            _score, fi = hit
            track._matched_face = all_faces[fi]
            if not track.is_occluded:
                fb = all_faces[fi]["bbox"]
                track.last_face_center = (
                    (fb["x1"] + fb["x2"]) / 2.0,
                    (fb["y1"] + fb["y2"]) / 2.0,
                )

    async def _process_frame(self, frame):
        """Run the full pipeline on a single frame."""
        height, width = frame.shape[:2]

        # 1) Unified YOLO Detection & Tracking on the shared inference pool
        tracked_detections = await run_inference(self.detector.track, frame)

        # 2) Full-frame face detection (ONE InsightFace call for ALL tracks)
        all_faces: list[dict] = []
        if self.demographic_enabled and self.insightface_analyzer:
            all_faces = await run_inference(
                self.insightface_analyzer.detect_all_faces, frame
            )

        # 3) Update in-memory track state and zones; match faces to tracks; collect pending DB work
        now_mono = time.monotonic()
        active_tracks: List[ActiveTrack] = []
        new_tracks: List[ActiveTrack] = []
        reid_tracks: List[ActiveTrack] = []
        observations: List[dict] = []

        for td in tracked_detections:
            if td.track_id is not None:
                track = self.track_manager.update_track(td.track_id, td.bbox, td.confidence)
                await asyncio.to_thread(self.track_manager.update_zones, track, self.zones, width, height)
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

        # ── Occlusion flags + global face-to-track assignment ─────────────
        # Overlapping bodies are marked is_occluded; immature tracks use
        # upper-body + face-overlap gates; ambiguous faces are rejected for
        # the frame; Hungarian (or greedy) picks the jointly best mapping.
        self._mark_occluded_tracks(active_tracks)
        self._assign_faces_to_tracks(all_faces, active_tracks)

        # 4) Automatic zone event detection
        zone_events = self.zone_event_detector.detect(active_tracks)

        # 5) Rule evaluation (pure in-memory; no DB)
        rule_events = self.rule_evaluator.evaluate(
            self.camera_id, active_tracks,
            frame_width=width, frame_height=height,
        )

        # 6) Stale tracks (in-memory removal; sessions closed in the same batch)
        stale = self.track_manager.cleanup_stale_tracks()
        for t in stale:
            self._last_obs_time.pop(t.local_track_id, None)
            # --- Change 3 & 6: clean up MinIO files that belong to this track before
            # discarding the in-memory state.
            #
            # 3) Delete the last current-face-crop (never stored in DB, never in accum_list).
            if t.current_face_crop_path:
                self._minio_cleanup(t.current_face_crop_path)
            #
            # 6) Delete any body/face crops from an incomplete ReID window (less than
            #    REID_ACCUMULATION_FRAMES arrived before the track went stale, so the
            #    normal window-cleanup never ran for them).  Protect paths still referenced
            #    by persistent track state that _close_track_session is about to save to DB.
            _partial_window = self.track_embeddings.pop(t.local_track_id, [])
            if _partial_window:
                _protected = {t.best_crop_path, t.best_face_crop_path_for_id}
                if t.best_demographics:
                    _protected.add(t.best_demographics.get("face_crop_path"))
                for _fe, _fs, _fp in t.face_embedding_list:
                    if _fp:
                        _protected.add(_fp)
                _protected.discard(None)
                for _item in _partial_window:
                    for _path in (_item[2], _item[5]):   # body crop, face crop
                        if _path and _path not in _protected:
                            self._minio_cleanup(_path)

        # 7) Persist - ONLY if there is actual work. Quiet frames skip the DB.
        if new_tracks or reid_tracks or rule_events or zone_events or observations or stale:
            await self._persist_batch(frame, new_tracks, reid_tracks, rule_events, zone_events, observations, stale)

        # 8) Update shared bounding box state for the StreamBroadcaster.
        #    IMPORTANT: mutate the list IN-PLACE (clear + extend) so the
        #    StreamBroadcaster — which holds a reference to this exact list
        #    object — always sees the latest data.  Do NOT reassign
        #    self.latest_tracks = [] because that would break the reference.
        self.latest_tracks.clear()
        for t in active_tracks:
            if t.bbox:
                # Use crop quality when available, fall back to YOLO confidence
                quality = t.current_crop_quality if t.current_crop_quality > 0 else t.avg_confidence
                
                # Check for face bbox in body crop coordinate space, and translate it to full-frame coordinates!
                face_bbox_full = None
                if getattr(t, "current_face_bbox", None) is not None:
                    # Face bbox coordinates are relative to the body crop!
                    bx = t.bbox["x1"]
                    by = t.bbox["y1"]
                    face_bbox_full = {
                        "x1": bx + t.current_face_bbox["x1"],
                        "y1": by + t.current_face_bbox["y1"],
                        "x2": bx + t.current_face_bbox["x2"],
                        "y2": by + t.current_face_bbox["y2"],
                    }
                
                self.latest_tracks.append({
                    **t.bbox,
                    "track_id": t.local_track_id,  # ByteTrack native ID
                    "quality": quality,
                    "confidence": getattr(t, "current_confidence", t.avg_confidence),
                    "face_bbox": face_bbox_full,
                    "face_score": getattr(t, "current_face_score", 0.0),
                })
        for td in tracked_detections:
            if td.track_id is None:
                self.latest_tracks.append({
                    **td.bbox,
                    "track_id": None,
                    "quality": 0.0,
                    "confidence": td.confidence,
                    "face_bbox": None,
                    "face_score": 0.0,
                })

        # 9) Optional GUI display
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

        # Extract initial crop (tight YOLO box; padding configurable)
        crop_path = None
        crop = await asyncio.to_thread(
            extract_crop, frame, track.bbox,
            padding_pct=self.settings.BODY_CROP_PADDING_PCT,
        )
        if crop is not None and crop.size > 0:
            crop_path = await save_image_async(crop, self.settings.CROP_DIR, prefix=f"crop_{self.camera_id}")
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
            severity=EventSeverity.LOW,
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
        """Mark a track session as ended in PostgreSQL.

        If the track has no resolved identity but has accumulated enough good
        face data, perform a final identity resolution before closing.
        """
        if not track.track_session_id:
            return
        from sqlalchemy import update, select
        from app.core.db.models.tracking import TrackSession
        from app.core.db.models.person import PersonIdentity
        from app.core.db.models.event import Event

        # Final identity resolution attempt for tracks that accumulated
        # enough good faces but hadn't yet met the window threshold
        close_resolved = False
        if (track.person_identity_id is None
                and track.best_face_embedding is not None
                and track.good_face_count >= self.settings.FACE_IDENTITY_MIN_DETECTIONS
                and self.identity_engine):
            try:
                person_id, score, is_confident, is_new, _ = None, 0.0, False, False, None
                try:
                    async with db.begin_nested():
                        person_id, score, is_confident, is_new, _ = await self.identity_engine.decide_identity(
                            db=db,
                            mean_embedding=track.best_body_embedding,
                            camera_id=self.camera_id,
                            crop_quality_score=track.best_crop_quality,
                            crop_path=track.best_crop_path,
                            current_person_id=None,
                            previous_score=0.0,
                            is_temporary=False,
                            face_embedding=track.best_face_embedding,
                            face_score=track.best_face_score_for_id,
                            face_crop_path=track.best_face_crop_path_for_id,
                            good_face_count=track.good_face_count,
                            face_embedding_list=track.face_embedding_list,
                            track_started_at=track.started_at,
                            track_session_id=track.track_session_id,
                        )
                except Exception as e:
                    logger.error(f"Close-track decide_identity savepoint failed: {e}")
                    person_id = None
                if person_id is not None:
                    track.person_identity_id = person_id
                    track.reid_score = score
                    track.reid_confident = is_confident
                    track.reid_resolved = True
                    close_resolved = True
                    if is_new:
                        self.temporary_person_ids.add(person_id)

                    # Store ALL accumulated good faces (skip the best, already stored by decide_identity)
                    person_id_uuid = person_id if isinstance(person_id, uuid.UUID) else uuid.UUID(person_id)
                    best_emb = track.best_face_embedding
                    for face_emb, face_scr, face_crp in track.face_embedding_list:
                        if face_emb is None:
                            continue
                        if best_emb is not None:
                            _best_n = np.linalg.norm(best_emb)
                            _face_n = np.linalg.norm(face_emb)
                            if _best_n > 0 and _face_n > 0:
                                sim_to_best = float(np.dot(best_emb, face_emb) / (_best_n * _face_n))
                            else:
                                sim_to_best = 0.0
                            if sim_to_best > 0.95:
                                continue
                        try:
                            async with db.begin_nested():
                                await self.identity_engine._store_face_embedding(
                                    db, person_id_uuid, face_emb, self.camera_id, face_scr, face_crp
                                )
                        except Exception as e:
                            logger.warning(f"Failed to store face embedding on close: {e}")

                    logger.info(
                        f"Track {track.local_track_id}: identity resolved on close "
                        f"person={str(person_id)[:8]} score={score:.2f}"
                    )
            except Exception as e:
                logger.warning(f"Final identity resolution failed for track {track.local_track_id}: {e}")

            # Post-resolution: update person_entered_view event and demographics
            if close_resolved:
                try:
                    # Update the person_entered_view event with the resolved identity
                    await db.execute(
                        update(Event)
                        .where(Event.track_session_id == track.track_session_id)
                        .where(Event.event_type == "person_entered_view")
                        .values(
                            person_identity_id=track.person_identity_id,
                            description=f"Person {str(track.person_identity_id)[:8]} entered view."
                        )
                    )

                    # Persist demographics to PersonIdentity if available
                    if track.best_demographics:
                        _votes = track.gender_votes
                        _gender = track.best_demographics.get("gender")
                        if _gender is None:
                            logger.warning(
                                f"Track {track.local_track_id}: no gender votes cast "
                                f"(M={_votes.get('M',0)} F={_votes.get('F',0)}) — "
                                f"PersonIdentity.gender will be NULL"
                            )
                        person_id_uuid = track.person_identity_id if isinstance(track.person_identity_id, uuid.UUID) else uuid.UUID(str(track.person_identity_id))
                        person_record = await db.get(PersonIdentity, person_id_uuid)
                        if person_record:
                            new_score = track.best_demographics.get("face_score", 0.0)
                            current_score = person_record.best_face_score or 0.0
                            if person_record.gender is None or new_score >= current_score:
                                person_record.gender = track.best_demographics["gender"]
                                person_record.age_group = track.best_demographics["age_group"]
                                person_record.estimated_age = track.best_demographics["age"]
                                person_record.best_face_score = new_score
                                person_record.face_crop_path = track.best_demographics["face_crop_path"]
                except Exception as e:
                    logger.warning(f"Post-resolution update failed for track {track.local_track_id}: {e}")

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
            "person_identity_id": track.person_identity_id,
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
            severity=EventSeverity.LOW,
            description=f"Person {str(track.person_identity_id)[:8] if track.person_identity_id else 'unknown'} left view.",
            occurred_at=utc_now(),
            metadata_json={"total_frames": track.total_frames, "duration_seconds": track.last_seen_at.timestamp() - track.started_at.timestamp()}
        )
        db.add(exit_event)
        # Note: PersonIdentity demographics (face_crop, age, gender) are now written
        # eagerly in _run_reid on every accumulation window. No deferred write needed here.


    async def _close_all_track_sessions(self):
        """Close all open track sessions on shutdown."""
        try:
            async with AsyncSessionLocal() as db:
                for track in self.track_manager.get_active_tracks():
                    await self._close_track_session(db, track)
                await db.commit()
        except Exception as e:
            logger.warning(f"Could not close track sessions for camera {self.camera_id}: {e}")



    # ── Deferred MinIO deletion ──────────────────────────────────────────────
    # All calls to _minio_cleanup() from this class add paths to this set instead
    # of deleting immediately.  The periodic ``deduplicate_persons()`` job
    # (every 10 min) sweeps MinIO ``crops/`` against a live list of DB-referenced
    # paths and deletes only truly unreferenced objects.  This eliminates the
    # entire class of "deleted-too-early" 404 bugs.
    _pending_minio_deletes: "set[str]" = set()

    @staticmethod
    def _minio_cleanup(full_path: str) -> None:
        """Queue a MinIO object for *deferred* deletion by the next dedup-job sweep.

        The object key is extracted from the full bucket/key path and added to the
        class-level ``_pending_minio_deletes`` set.  Actual deletion happens in
        ``deduplicate_persons()`` (every 10 min) after verifying no DB row still
        references this path.
        """
        if full_path:
            CameraWorker._pending_minio_deletes.add(full_path)

    async def _run_reid(self, db, frame, track: ActiveTrack):
        """Run ReID pipeline: crop -> quality -> embedding -> accumulation -> decision."""
        track.last_reid_time = utc_now()
        was_resolved = track.reid_resolved
        face_result = None  # ensure defined even when demographics are disabled
        try:
            # --- Change 4: clean up previous current_crop_path if it was never accumulated
            # and is not the overall best.  This prevents body-crop leaks from frames that
            # failed the quality/face gate and were never added to accum_list.
            _prev_body_crop = track.current_crop_path
            if _prev_body_crop:
                _accum_body_paths = {
                    item[2] for item in self.track_embeddings.get(track.local_track_id, [])
                }
                if _prev_body_crop not in _accum_body_paths and _prev_body_crop != track.best_crop_path:
                    self._minio_cleanup(_prev_body_crop)

            crop = await asyncio.to_thread(
                extract_crop, frame, track.bbox,
                padding_pct=self.settings.BODY_CROP_PADDING_PCT,
            )
            if crop is None or crop.size == 0:
                logger.warning(f"Track {track.local_track_id}: Failed to extract crop from bounding box.")
                return

            # Run YOLO-Pose keypoints check if pose model is available
            keypoint_visibility_ratio = None
            keypoint_gate_passed = False
            if self.yolo_pose:
                try:
                    pose_results = await run_inference(
                        self.yolo_pose.predict,
                        crop,
                        verbose=False,
                        conf=self.settings.YOLO_POSE_CONFIDENCE
                    )
                    if pose_results and pose_results[0].keypoints is not None:
                        kp_conf = pose_results[0].keypoints.conf
                        if kp_conf is not None and kp_conf.shape[0] > 0:
                            from app.modules.reid.crop_quality import check_torso_keypoints
                            keypoint_visibility_ratio, keypoint_gate_passed = check_torso_keypoints(
                                kp_conf[0].cpu().numpy(),
                                confidence_threshold=self.settings.YOLO_POSE_CONFIDENCE
                            )
                except Exception as pose_err:
                    logger.debug(f"Pose check failed: {pose_err}")

            # Assess quality (returns either a float score or a dict with detailed metrics)
            quality_result = await asyncio.to_thread(
                assess_crop_quality,
                crop,
                keypoint_visibility_ratio=keypoint_visibility_ratio,
                yolo_confidence=track.current_confidence,
            )
            if isinstance(quality_result, dict):
                quality = quality_result.get("quality_score", 0.0)
                keypoint_visibility_ratio = quality_result.get("keypoint_visibility_ratio", keypoint_visibility_ratio)
                keypoint_gate_passed = quality_result.get("keypoint_gate_passed", keypoint_gate_passed)
                sharpness_score = quality_result.get("sharpness_score")
                size_score = quality_result.get("size_score")
                aspect_ratio = quality_result.get("aspect_ratio")
                brightness_mean = quality_result.get("brightness_mean")
                quality_passed = quality >= self.settings.REID_CROP_QUALITY_THRESHOLD
            else:
                quality = float(quality_result)
                quality_passed = quality >= self.settings.REID_CROP_QUALITY_THRESHOLD
                sharpness_score = None
                size_score = None
                aspect_ratio = None
                brightness_mean = None

            h, w = crop.shape[:2]

            # Save crop for audit / debugging (MUST be before any debug log that references crop_path)
            crop_path = await save_image_async(crop, self.settings.CROP_DIR, prefix=f"crop_{self.camera_id}")

            # Run demographics / face detection on all crops first (regardless of body quality)
            face_embedding = None
            face_score = 0.0
            face_crop_path = None
            face_result = None
            face_frontal = False
            
            # --- Change 2: capture previous curr_face path before resetting, so we can
            # delete it from MinIO after uploading the replacement (prevents permanent leak).
            _prev_curr_face = track.current_face_crop_path
            track.current_face_crop_path = None
            track.current_face_score = 0.0
            track.current_face_bbox = None

            # ── Full-frame face (NO per-track InsightFace call) ────────────
            # face was pre-detected on the full frame in _process_frame and
            # matched to this track.  Extract the face crop at NATIVE resolution
            # from the full frame (not the body crop), pad-resize to 224² for
            # SigLIP2, and build an InsightFaceResult for the downstream pipeline.
            _face_info = getattr(track, '_matched_face', None)
            if _face_info:
                face_bbox = _face_info["bbox"]
                face_crop_raw = await asyncio.to_thread(
                    extract_crop, frame, face_bbox, padding_pct=0.30
                )
                if face_crop_raw is not None and face_crop_raw.size > 0:
                    # Pad-resize to square (preserves aspect ratio, no distortion)
                    face_crop_sq = resize_pad_square(face_crop_raw, 224)

                    # Save for real-time debug
                    current_face_path = await save_image_async(
                        face_crop_sq, self.settings.CROP_DIR, prefix=f"curr_face_{self.camera_id}"
                    )
                    track.current_face_crop_path = current_face_path
                    if _prev_curr_face and _prev_curr_face != current_face_path:
                        self._minio_cleanup(_prev_curr_face)

                    # Age from full-frame genderage (if present on matched detection)
                    _age = _face_info.get("age")
                    _age_group = None
                    if _age is not None and self.insightface_analyzer:
                        _age_group = self.insightface_analyzer._age_to_group(int(_age))

                    from app.modules.reid.insightface_analyzer import InsightFaceResult
                    face_result = InsightFaceResult(
                        age=_age, gender=None, age_group=_age_group,
                        face_score=float(_face_info.get("det_score", 0.0)),
                        face_bbox=face_bbox,
                        embedding=_face_info.get("embedding"),
                        face_crop=face_crop_sq,
                        kps=_face_info.get("kps"),
                    )
                    from app.modules.reid.crop_quality import assess_face_quality as a_fq
                    face_result.face_quality = a_fq(face_result)

                    # Compute eye_spread / frontality from kps (matches original)
                    fw = max(face_bbox["x2"] - face_bbox["x1"], 1.0)
                    fh = max(face_bbox["y2"] - face_bbox["y1"], 1.0)
                    fcx = (face_bbox["x1"] + face_bbox["x2"]) / 2.0
                    kps = _face_info.get("kps")
                    if kps is not None and len(kps) >= 2:
                        lx, ly = float(kps[0][0]), float(kps[0][1])
                        rx, ry = float(kps[1][0]), float(kps[1][1])
                        face_result.eye_spread = abs(rx - lx) / fw
                    if kps is not None and len(kps) >= 5:
                        spread_sc = min(1.0, face_result.eye_spread / 0.35)
                        nose_cx = float(kps[2][0])
                        nose_off = abs(nose_cx - fcx) / (fw / 2.0)
                        nose_sc = max(0.0, 1.0 - nose_off)
                        eye_vert = abs(float(kps[1][1]) - float(kps[0][1])) / fh
                        sym_sc = max(0.0, 1.0 - eye_vert * 4.0)
                        face_result.frontality_score = 0.55 * spread_sc + 0.30 * nose_sc + 0.15 * sym_sc
                    elif kps is not None and len(kps) >= 2:
                        face_result.frontality_score = min(1.0, face_result.eye_spread / 0.35)
                    else:
                        face_result.frontality_score = 0.5

                    # ── SigLIP2 gender (face-only + margin δ) ─────────────
                    _sig = None
                    if self.siglip2_analyzer:
                        _sig = await run_inference(
                            self.siglip2_analyzer.analyze, face_crop_sq
                        )
                        if _sig:
                            face_result.gender = _sig["gender"]
                            if "margin" in _sig:
                                track.gender_margins.append(float(_sig["margin"]))

                    # ── Age samples from InsightFace genderage ────────────
                    if face_result.age is not None:
                        track.age_samples.append(int(face_result.age))
                        _med = int(round(float(np.median(track.age_samples))))
                        face_result.age = _med
                        if self.insightface_analyzer:
                            face_result.age_group = self.insightface_analyzer._age_to_group(_med)

                    if _sig:
                        logger.debug(
                            f"Track {track.local_track_id}: SigLIP2={_sig['gender']} "
                            f"margin={_sig.get('margin', 0):+.2f} "
                            f"age_med={face_result.age}"
                        )
                    elif not self.siglip2_analyzer and not self.insightface_analyzer:
                        logger.warning(
                            f"Track {track.local_track_id}: No gender/age model available"
                        )
                    track.current_face_score = face_result.face_quality
                    track.current_face_bbox = face_result.face_bbox

                    # Clean up previous curr_face if face_crop failed (no new upload)
                elif _prev_curr_face:
                    self._minio_cleanup(_prev_curr_face)

            # If we got a face, proceed with frontality checks etc.
            if face_result:
                    
                    # Validate frontality criteria using settings.
                    # face_quality from assess_face_quality() now encodes both det_score
                    # AND geometric frontality (eye-spread, nose centering, eye symmetry).
                    # We still gate on raw det_score and pixel size as hard minimums,
                    # then use frontality_score (from insightface_analyzer) to discriminate
                    # frontal from angled faces for the ReID pipeline.
                    face_bbox   = face_result.face_bbox
                    face_width  = face_bbox["x2"] - face_bbox["x1"]
                    face_height = face_bbox["y2"] - face_bbox["y1"]

                    face_frontal     = True
                    rejection_reason = ""

                    # ── KPS geometry validation ──────────────────────────────
                    # InsightFace hallucinates face landmarks on hair / objects.
                    # Real faces have consistent geometry: left eye left of right
                    # eye, both eyes roughly horizontal, nose between them.
                    # Hallucinated landmarks violate these invariants.
                    kps = face_result.kps
                    if kps is not None and len(kps) >= 5:
                        lx, ly = float(kps[0][0]), float(kps[0][1])
                        rx, ry = float(kps[1][0]), float(kps[1][1])
                        nx     = float(kps[2][0])
                        if lx >= rx or abs(ry - ly) / max(face_height, 1) > 0.20:
                            face_frontal = False
                            rejection_reason = "invalid kps geometry (hair/object hallucination)"
                        else:
                            eye_mid = (lx + rx) / 2.0
                            eye_sep = max(rx - lx, 1.0)
                            if abs(nx - eye_mid) / eye_sep > 0.40:
                                face_frontal = False
                                rejection_reason = "nose offset from eye-midpoint (hallucinated landmarks)"

                    # ── Skin color check ──────────────────────────────────
                    # InsightFace falsely detects faces on hair, clothing, and
                    # objects. A real face crop has a significant portion of
                    # skin-colored pixels (HSV). If the crop is mostly non-skin
                    # (hair, fabric, wall), it's a false positive.
                    # False positive example: 2.4% skin. Real face: 15.5% skin.
                    if face_frontal and face_result.face_crop is not None:
                        try:
                            import cv2
                            hsv = cv2.cvtColor(face_result.face_crop, cv2.COLOR_BGR2HSV)
                            skin_mask = cv2.inRange(
                                hsv,
                                (0, 20, 70),   # lower HSV skin bound
                                (20, 255, 255) # upper HSV skin bound
                            )
                            skin_pct = skin_mask.sum() / 255 / (face_result.face_crop.shape[0] * face_result.face_crop.shape[1])
                            if skin_pct < 0.03:
                                face_frontal = False
                                rejection_reason = f"skin color check failed: {skin_pct*100:.1f}% skin pixels (hair/object false positive)"
                        except Exception as e:
                            logger.debug(f"Skin color check failed: {e}")

                    if face_frontal and face_result.face_score < self.settings.FACE_MIN_DET_SCORE:
                        face_frontal = False
                        rejection_reason = f"det_score {face_result.face_score:.2f} < {self.settings.FACE_MIN_DET_SCORE}"
                    elif face_frontal and face_width < self.settings.FACE_MIN_SIZE_PX:
                        face_frontal = False
                        rejection_reason = f"width {face_width:.0f}px < {self.settings.FACE_MIN_SIZE_PX}px"
                    elif face_frontal and face_result.eye_spread < self.settings.FACE_MIN_EYE_SPREAD:
                        face_frontal = False
                        rejection_reason = (
                            f"eye_spread {face_result.eye_spread:.2f} < {self.settings.FACE_MIN_EYE_SPREAD} "
                            f"(frontality={face_result.frontality_score:.2f})"
                        )

                    # ── Gender: mean SigLIP2 margin, female-biased δ ─────
                    # Per-frame gender still recorded for debug; decision uses
                    # mean(male_best − female_best) > SIGLIP2_GENDER_MARGIN_DELTA.
                    if face_result.gender in ("M", "F"):
                        track.gender_votes[face_result.gender] += 1
                    _majority_gender = None
                    if track.gender_margins:
                        _mean_margin = float(np.mean(track.gender_margins))
                        _delta = float(self.settings.SIGLIP2_GENDER_MARGIN_DELTA)
                        _majority_gender = "M" if _mean_margin > _delta else "F"
                    elif face_result.gender in ("M", "F"):
                        _majority_gender = face_result.gender

                    # Keep best_demographics gender/age in sync from running aggregates
                    if track.best_demographics is not None:
                        _prev_gender = track.best_demographics.get("gender")
                        if _majority_gender and _prev_gender != _majority_gender:
                            track.best_demographics["gender"] = _majority_gender
                            logger.debug(
                                f"Gender flipped for track {track.local_track_id}: "
                                f"{_prev_gender} → {_majority_gender} "
                                f"(mean_margin={float(np.mean(track.gender_margins)) if track.gender_margins else 0:+.2f})"
                            )
                        if track.age_samples:
                            _med_age = int(round(float(np.median(track.age_samples))))
                            track.best_demographics["age"] = _med_age
                            if self.insightface_analyzer:
                                track.best_demographics["age_group"] = (
                                    self.insightface_analyzer._age_to_group(_med_age)
                                )

                    if face_frontal:
                        face_embedding = face_result.embedding

                        # ── Contamination gate (Layer 2) ───────────────────
                        # If this track has already accumulated ≥2 prior good faces,
                        # validate the new face against the running consensus.
                        # A face with cosine similarity < CONTAMINATION_THRESHOLD to
                        # ALL prior faces is almost certainly from an adjacent person
                        # whose body crop overlapped ours — reject this frame's face.
                        if (face_embedding is not None
                                and len(track.face_embedding_list) >= 2):
                            # InsightFace embeddings are NOT L2-normalized (norms 12-27),
                            # so raw np.dot() gives values in the hundreds, not [-1, +1].
                            # Normalize both vectors before the dot product.
                            _face_n = np.linalg.norm(face_embedding)
                            _max_sim_to_prior = 0.0
                            for _prior_emb, _prior_scr, _ in track.face_embedding_list:
                                if _prior_emb is not None:
                                    _prior_n = np.linalg.norm(_prior_emb)
                                    if _face_n > 0 and _prior_n > 0:
                                        _sim = float(np.dot(_prior_emb, face_embedding) / (_prior_n * _face_n))
                                    else:
                                        _sim = 0.0
                                    if _sim > _max_sim_to_prior:
                                        _max_sim_to_prior = _sim
                            if _max_sim_to_prior < self.settings.FACE_CONTAMINATION_THRESHOLD:
                                logger.warning(
                                    f"Track {track.local_track_id}: Face contamination detected! "
                                    f"MaxPriorSim={_max_sim_to_prior:.3f} < "
                                    f"CONTAMINATION_THRESHOLD={self.settings.FACE_CONTAMINATION_THRESHOLD:.2f}. "
                                    f"Rejecting this frame's face."
                                )
                                face_frontal = False
                                rejection_reason = (
                                    f"contamination (max_prior_sim={_max_sim_to_prior:.3f} < "
                                    f"{self.settings.FACE_CONTAMINATION_THRESHOLD:.2f})"
                                )
                            else:
                                logger.info(
                                    f"Track {track.local_track_id}: Face PASSED contamination gate "
                                    f"(max_prior_sim={_max_sim_to_prior:.3f}, "
                                    f"prior_faces={len(track.face_embedding_list)})"
                                )

                        if face_frontal:
                            # Use face_quality (det_score × frontality) as the score so that
                            # a clean frontal detection always ranks higher than an angled one.
                            face_score = face_result.face_quality
                            if face_result.face_crop is not None:
                                face_crop_path = await save_image_async(face_result.face_crop, self.settings.CROP_DIR, prefix=f"face_{self.camera_id}")

                            if (track.best_demographics is None or
                                    face_score > track.best_demographics.get("face_score", 0.0)):
                                _demo_age = face_result.age
                                _demo_group = face_result.age_group
                                if track.age_samples:
                                    _demo_age = int(round(float(np.median(track.age_samples))))
                                    if self.insightface_analyzer:
                                        _demo_group = self.insightface_analyzer._age_to_group(_demo_age)
                                track.best_demographics = {
                                    "age":           _demo_age,
                                    "gender":        _majority_gender,
                                    "age_group":     _demo_group,
                                    "face_score":    face_score,
                                    "frontality":    face_result.frontality_score,
                                    "face_crop_path": face_crop_path,
                                }
                                logger.debug(
                                    f"Demographics updated for track {track.local_track_id}: "
                                    f"quality={face_score:.2f} frontality={face_result.frontality_score:.2f}"
                                )
                        else:
                            logger.debug(
                                f"Track {track.local_track_id}: "
                                f"Face rejected — {rejection_reason}"
                            )
            elif _prev_curr_face:
                # Demographics disabled this frame; prev curr_face file is now orphaned — delete it
                self._minio_cleanup(_prev_curr_face)

            # Update current frame crop quality and path (updated every frame for real-time debug)
            track.current_crop_quality = quality
            track.current_crop_path = crop_path

            # Update best crop quality and path seen so far
            if quality > track.best_crop_quality:
                track.best_crop_quality = quality
                track.best_crop_path = crop_path

            # Check if we should extract body ReID or fall back to face-only
            body_embedding = None
            should_accumulate = False
            
            skip_body = (
                self.settings.SKIP_BODY_REID_WHEN_OCCLUDED
                and track.is_occluded
            )
            if skip_body:
                logger.info(
                    f"BodyReID skipped (occluded) track={track.local_track_id} "
                    f"cam={self.camera_id}"
                )

            if quality_passed and not skip_body:
                # Standard path: extract OSNet body embedding
                body_embedding = await run_inference(self.reid_extractor.extract, crop)
                if body_embedding is not None:
                    should_accumulate = True
                else:
                    logger.warning("OSNet embedding extraction returned None.")
            elif face_frontal and face_embedding is not None:
                # Fallback path: body quality is low / occluded, but high-quality face is present
                if skip_body:
                    logger.info(
                        f"Track {track.local_track_id}: occluded body skipped, "
                        f"falling back to face-only identification."
                    )
                else:
                    logger.info(f"Track {track.local_track_id}: body quality rejected ({quality:.2f}), falling back to face-only identification.")
                should_accumulate = True

            if not should_accumulate:
                logger.debug(f"Track {track.local_track_id}: quality too low ({quality:.2f}) and no valid frontal face. Skipping ReID accumulation.")
                return

            # Accumulate embedding
            accum_list = self.track_embeddings.setdefault(track.local_track_id, [])
            accum_list.append((body_embedding, quality, crop_path, face_embedding, face_score, face_crop_path))
            track.reid_frame_count += 1

            # Track good faces across all windows for identity creation gating
            if face_frontal and face_score >= self.settings.FACE_IDENTITY_MIN_SCORE:
                track.good_face_count += 1
                # Accumulate all good faces (different angles) for later storage.
                # Deduplicate: skip if nearly identical (>0.95 cosine sim) to an
                # existing face already in the list — same crop/angle from
                # consecutive windows should not be stored multiple times.
                if face_embedding is not None:
                    is_duplicate = False
                    _face_n = np.linalg.norm(face_embedding)
                    _sims_to_existing = []
                    for existing_emb, _, _ in track.face_embedding_list:
                        _existing_n = np.linalg.norm(existing_emb)
                        if _face_n > 0 and _existing_n > 0:
                            _sim = float(np.dot(existing_emb, face_embedding) / (_existing_n * _face_n))
                        else:
                            _sim = 0.0
                        _sims_to_existing.append(_sim)
                        if _sim > 0.95:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        track.face_embedding_list.append((face_embedding, face_score, face_crop_path))
                        logger.info(
                            f"Track {track.local_track_id}: Face ACCUMULATED (#{len(track.face_embedding_list)}) "
                            f"(score={face_score:.2f}, sims_to_prior={[f'{s:.3f}' for s in _sims_to_existing]})"
                        )
                    else:
                        logger.debug(
                            f"Track {track.local_track_id}: Face SKIPPED as duplicate "
                            f"(sims_to_prior={[f'{s:.3f}' for s in _sims_to_existing]})"
                        )
                    # Keep the best face embedding for the identity matching step
                    if face_score > track.best_face_score_for_id:
                        track.best_face_embedding = face_embedding
                        track.best_face_score_for_id = face_score
                        track.best_face_crop_path_for_id = face_crop_path

            # Decision execution on reaching the window size (5 frames)
            if len(accum_list) == self.settings.REID_ACCUMULATION_FRAMES:
                body_items = [item for item in accum_list if item[0] is not None]
                if body_items:
                    best_body = max(body_items, key=lambda item: item[1])
                    selected_embedding = best_body[0]
                    norm = np.linalg.norm(selected_embedding)
                    if norm > 0:
                        selected_embedding = selected_embedding / norm
                else:
                    selected_embedding = None

                # Update track-level best body embedding
                if selected_embedding is not None:
                    track.best_body_embedding = selected_embedding

                best_crop_item = max(accum_list, key=lambda item: item[1])
                best_quality = best_crop_item[1]
                best_crop_path = best_crop_item[2]
                
                # Find the best face in this accumulation window
                best_face_item = max(accum_list, key=lambda item: item[4])
                window_face_embedding = best_face_item[3]
                window_face_score = best_face_item[4]
                window_face_crop_path = best_face_item[5]

                # Delete the other unused crops in this window from MinIO.
                # --- Change 1: build a protected set of face paths already referenced in
                # face_embedding_list — these will later be stored to PersonFaceEmbedding
                # and must NOT be deleted, otherwise the DB would hold 404 URLs.
                _protected_face_paths = {
                    entry[2] for entry in track.face_embedding_list if entry[2] is not None
                }
                for item in accum_list:
                    other_path = item[2]
                    other_face_path = item[5]
                    if other_path and other_path != best_crop_path:
                        self._minio_cleanup(other_path)
                    # Only delete if not the window survivor AND not already referenced
                    # in face_embedding_list (which will be persisted to DB).
                    if (other_face_path
                            and other_face_path != window_face_crop_path
                            and other_face_path not in _protected_face_paths):
                        self._minio_cleanup(other_face_path)

                # --- Change 5: if current_crop_path was one of the deleted window crops,
                # point it at the survivor so the active-tracks debug view never shows a 404.
                _window_body_paths = {item[2] for item in accum_list if item[2]}
                if track.current_crop_path in _window_body_paths and track.current_crop_path != best_crop_path:
                    track.current_crop_path = best_crop_path

                accum_list.clear()

                # --- FACE-FIRST DEFERRED RESOLUTION ---
                # If the track has no identity yet and hasn't accumulated enough
                # good faces, defer the identity decision. Keep accumulating faces
                # across subsequent windows until the threshold is met or track closes.
                if (track.person_identity_id is None
                        and track.good_face_count < self.settings.FACE_IDENTITY_MIN_DETECTIONS):
                    logger.debug(
                        f"Track {track.local_track_id}: deferring identity resolution "
                        f"(good_face_count={track.good_face_count}/{self.settings.FACE_IDENTITY_MIN_DETECTIONS})"
                    )
                    return

                is_temp = track.person_identity_id in self.temporary_person_ids

                logger.info(
                    f"Track {track.local_track_id}: ReID window firing — "
                    f"accumulated_faces={len(track.face_embedding_list)}, "
                    f"good_face_count={track.good_face_count}, "
                    f"best_face_score={track.best_face_score_for_id:.2f}, "
                    f"current_person={str(track.person_identity_id)[:8] if track.person_identity_id else 'None'}, "
                    f"is_temp={is_temp}"
                )

                # SAVEPOINT: identity persistence FK races must not poison batch txn
                try:
                    async with db.begin_nested():
                        person_id, score, is_confident, is_new, prune_old_id = await self.identity_engine.decide_identity(
                            db=db,
                            mean_embedding=selected_embedding,
                            camera_id=self.camera_id,
                            crop_quality_score=best_quality,
                            crop_path=best_crop_path,
                            current_person_id=track.person_identity_id,
                            previous_score=track.reid_score,
                            is_temporary=is_temp,
                            face_embedding=track.best_face_embedding,
                            face_score=track.best_face_score_for_id,
                            face_crop_path=track.best_face_crop_path_for_id,
                            good_face_count=track.good_face_count,
                            face_embedding_list=track.face_embedding_list,
                            track_started_at=track.started_at,
                            track_session_id=track.track_session_id,
                        )
                except Exception as e:
                    logger.error(
                        f"Track {track.local_track_id}: decide_identity savepoint failed: {e}"
                    )
                    person_id, score, is_confident, is_new, prune_old_id = (
                        track.person_identity_id, track.reid_score, track.reid_confident, False, None
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

                # Store ALL accumulated good faces (different angles) once identity is resolved.
                # decide_identity already stored the best face, so skip the one that matches it
                # (same embedding already in DB) to avoid duplicates.
                if person_id and track.face_embedding_list:
                    from app.core.db.models.person import PersonIdentity
                    person_id_uuid = person_id if isinstance(person_id, uuid.UUID) else uuid.UUID(person_id)
                    best_emb = track.best_face_embedding
                    stored_count = 0
                    for face_emb, face_scr, face_crp in track.face_embedding_list:
                        if face_emb is None:
                            continue
                        # Skip the face that decide_identity already stored (the best one)
                        if best_emb is not None:
                            _best_n = np.linalg.norm(best_emb)
                            _face_n = np.linalg.norm(face_emb)
                            if _best_n > 0 and _face_n > 0:
                                sim_to_best = float(np.dot(best_emb, face_emb) / (_best_n * _face_n))
                            else:
                                sim_to_best = 0.0
                            if sim_to_best > 0.95:
                                continue
                        try:
                            async with db.begin_nested():
                                await self.identity_engine._store_face_embedding(
                                    db, person_id_uuid, face_emb, self.camera_id, face_scr, face_crp
                                )
                            stored_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to store additional face embedding: {e}")

                    logger.info(
                        f"Track {track.local_track_id}: Identity resolved — "
                        f"person={str(person_id)[:8]}, "
                        f"faces_in_list={len(track.face_embedding_list)}, "
                        f"face_stored_via_store={stored_count}"
                    )



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

                if person_id and track.best_demographics:
                    # Ensure person_id is a UUID object (in case it is returned as a string from raw SQL)
                    if isinstance(person_id, str):
                        person_id_uuid = uuid.UUID(person_id)
                    else:
                        person_id_uuid = person_id
                    
                    person_record = await db.get(PersonIdentity, person_id_uuid)

                    if person_record:
                        new_score = track.best_demographics.get("face_score", 0.0)
                        current_score = person_record.best_face_score or 0.0

                        if person_record.gender is None or new_score >= current_score:
                            person_record.gender = track.best_demographics["gender"]
                            person_record.age_group = track.best_demographics["age_group"]
                            person_record.estimated_age = track.best_demographics["age"]
                            person_record.best_face_score = new_score
                            person_record.face_crop_path = track.best_demographics["face_crop_path"]
                            
                            logger.info(
                                f"Updated PersonIdentity {person_id} demographics (eager): "
                                f"gender={person_record.gender}, age={person_record.estimated_age} "
                                f"(score={new_score:.3f})"
                            )

                logger.info(
                    f"ReID resolved (count={track.reid_frame_count}): track={track.local_track_id} "
                    f"person={person_id} score={score:.2f} confident={is_confident}"
                )

        except Exception as e:
            logger.error(f"ReID failed for track {track.local_track_id}: {e}")


    async def _persist_events(self, db, frame, rule_events: List[RuleEvent], zone_events: List[ZoneEvent]):
        """Persist rule engine events and automatic zone events to PostgreSQL."""
        from sqlalchemy import select
        from app.core.db.models.event import Event, EventSeverity
        from app.core.db.models.billing import BillingInteraction

        snapshot_path = await save_image_async(frame, self.settings.SNAPSHOT_DIR, prefix=f"event_{self.camera_id}")

        for ev in rule_events:
            try:
                # Purchase events are High severity; all others are Low
                severity = EventSeverity.HIGH if ev.event_type == "purchase" else EventSeverity.LOW
                event = Event(
                    camera_id=ev.camera_id,
                    rule_id=ev.rule_id,
                    zone_id=ev.zone_id,
                    person_identity_id=ev.person_identity_id,
                    track_session_id=ev.track_session_id,
                    event_type=ev.event_type,
                    severity=severity,
                    description=ev.description,
                    metadata_json=ev.metadata,
                    snapshot_path=snapshot_path,
                    occurred_at=utc_now(),
                )
                db.add(event)

                # Billing interactions get an additional structured record.
                # Only create ONE per track_session + zone to prevent
                # cooldown resets from inflating purchase counts.
                if ev.rule_type == "billing_interaction":
                    existing = await db.execute(
                        select(BillingInteraction).where(
                            BillingInteraction.track_session_id == ev.track_session_id,
                            BillingInteraction.zone_id == ev.zone_id,
                        )
                    )
                    if existing.scalar_one_or_none() is not None:
                        continue  # already recorded for this track+zone combo
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
                    severity=EventSeverity.LOW,
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

                # Confidence and bbox size
                conf = getattr(track, "current_confidence", track.avg_confidence)
                bw = x2 - x1
                bh = y2 - y1
                label_parts.append(f"C:{conf:.2f}")
                label_parts.append(f"{bw}x{bh}")
                
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

            # 3. Draw untracked detections in local GUI
            for td in tracked_detections:
                if td.track_id is None:
                    try:
                        x1 = int(td.bbox["x1"])
                        y1 = int(td.bbox["y1"])
                        x2 = int(td.bbox["x2"])
                        y2 = int(td.bbox["y2"])
                        color = (128, 128, 128)  # Grey for untracked/anonymous detections
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"Anonymous C:{td.confidence:.2f}"
                        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(display_frame, (x1, y1 - lbl_h - 6), (x1 + lbl_w + 4, y1), color, -1)
                        cv2.putText(
                            display_frame,
                            label,
                            (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    except Exception:
                        continue

            # Show frame
            window_title = f"Camera Stream - {self.camera_id}"
            cv2.imshow(window_title, display_frame)
            cv2.waitKey(1)
        except Exception as e:
            logger.warning(f"Error in OpenCV display GUI: {e}")
            self._gui_available = False