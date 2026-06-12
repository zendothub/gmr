"""ByteTrack adapter for multi-object tracking."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger

from app.modules.detection.yolo_detector import DetectionResult


@dataclass
class TrackOutput:
    """Single tracked object output."""
    local_track_id: int
    bbox: dict  # {x1, y1, x2, y2}
    confidence: float
    class_id: int = 0


class ByteTrackAdapter:
    """Adapter for ByteTrack multi-object tracker."""

    def __init__(self, track_thresh: float = 0.45, match_thresh: float = 0.8, track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.tracker = None
        self._init_tracker()

    def _init_tracker(self):
        """Initialize ByteTrack tracker."""
        try:
            # TODO: Import actual ByteTrack implementation
            # from bytetrack import BYTETracker
            # For now, use a simple ID assignment tracker as stub
            self.tracker = SimpleTracker()
            logger.info("ByteTrack adapter initialized (using simple tracker stub)")
        except ImportError as e:
            logger.warning(f"ByteTrack not available, using simple tracker: {e}")
            self.tracker = SimpleTracker()

    def update(self, detections: List[DetectionResult], frame_shape: tuple = None) -> List[TrackOutput]:
        """
        Update tracker with new detections.

        Args:
            detections: List of DetectionResult from YOLO
            frame_shape: (height, width) of the frame

        Returns:
            List of TrackOutput with assigned track IDs
        """
        if not detections:
            return self.tracker.update_empty()

        try:
            # Convert detections to numpy array [x1, y1, x2, y2, confidence]
            det_array = np.array([
                [d.bbox["x1"], d.bbox["y1"], d.bbox["x2"], d.bbox["y2"], d.confidence]
                for d in detections
            ])

            tracks = self.tracker.update(det_array)
            return tracks

        except Exception as e:
            logger.error(f"Tracking update failed: {e}")
            return []

    def reset(self):
        """Reset tracker state."""
        self._init_tracker()


class SimpleTracker:
    """
    Simple IoU-based tracker as a fallback/stub for ByteTrack.
    Replace with actual ByteTrack implementation for production.
    """

    def __init__(self):
        self.next_id = 1
        self.active_tracks: dict = {}  # track_id -> last bbox
        self.max_age = 30  # frames before track is removed
        self.age_counter: dict = {}

    def update(self, detections: np.ndarray) -> List[TrackOutput]:
        """Update with detections array [N, 5] (x1,y1,x2,y2,conf)."""
        outputs = []

        if len(detections) == 0:
            return self.update_empty()

        # Simple nearest-neighbor matching based on IoU
        matched = set()
        for i, det in enumerate(detections):
            best_iou = 0.3  # minimum IoU threshold
            best_track_id = None

            for tid, last_bbox in self.active_tracks.items():
                iou = self._compute_iou(det[:4], last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None and best_track_id not in matched:
                # Update existing track
                self.active_tracks[best_track_id] = det[:4]
                self.age_counter[best_track_id] = 0
                matched.add(best_track_id)
                outputs.append(TrackOutput(
                    local_track_id=best_track_id,
                    bbox={"x1": float(det[0]), "y1": float(det[1]), "x2": float(det[2]), "y2": float(det[3])},
                    confidence=float(det[4]),
                ))
            else:
                # Create new track
                tid = self.next_id
                self.next_id += 1
                self.active_tracks[tid] = det[:4]
                self.age_counter[tid] = 0
                outputs.append(TrackOutput(
                    local_track_id=tid,
                    bbox={"x1": float(det[0]), "y1": float(det[1]), "x2": float(det[2]), "y2": float(det[3])},
                    confidence=float(det[4]),
                ))

        # Age unmatched tracks
        to_remove = []
        for tid in self.active_tracks:
            if tid not in matched:
                self.age_counter[tid] = self.age_counter.get(tid, 0) + 1
                if self.age_counter[tid] > self.max_age:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.active_tracks[tid]
            del self.age_counter[tid]

        return outputs

    def update_empty(self) -> List[TrackOutput]:
        """Handle frame with no detections."""
        to_remove = []
        for tid in self.active_tracks:
            self.age_counter[tid] = self.age_counter.get(tid, 0) + 1
            if self.age_counter[tid] > self.max_age:
                to_remove.append(tid)
        for tid in to_remove:
            del self.active_tracks[tid]
            del self.age_counter[tid]
        return []

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU of two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0
