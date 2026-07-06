"""YOLO detector module using ultralytics."""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from app.config import get_settings

# Shared detector instances (one per model path) so multiple camera workers
# don't each load their own copy of the weights into RAM/VRAM.
_shared_detectors: Dict[str, "YOLODetector"] = {}
_shared_lock = threading.Lock()

_shared_pose_models: Dict[str, "YOLO"] = {}
_shared_pose_lock = threading.Lock()


def get_shared_detector(
    model_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    allowed_classes: Optional[List[int]] = None,
) -> "YOLODetector":
    """Get (or lazily create) a process-wide shared detector for a model path."""
    settings = get_settings()
    key = model_path or settings.YOLO_MODEL_PATH
    with _shared_lock:
        if key not in _shared_detectors:
            _shared_detectors[key] = YOLODetector(
                model_path=key,
                confidence_threshold=confidence_threshold,
                allowed_classes=allowed_classes,
            )
            logger.info(f"Shared YOLO detector created for {key}")
        return _shared_detectors[key]


def get_shared_pose_model(model_path: Optional[str] = None):
    """Get (or lazily create) a process-wide shared YOLO-Pose model."""
    from ultralytics import YOLO
    import torch
    settings = get_settings()
    key = model_path or settings.YOLO_POSE_MODEL_PATH
    with _shared_pose_lock:
        if key not in _shared_pose_models:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            model = YOLO(key)
            model.to(device)
            _shared_pose_models[key] = model
            logger.info(f"Shared YOLO-Pose model created for {key} on {device}")
        return _shared_pose_models[key]


@dataclass
class DetectionResult:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: dict  # {x1, y1, x2, y2}


@dataclass
class TrackedDetection:
    """Detection with active tracker ID."""
    track_id: Optional[int]
    confidence: float
    bbox: dict  # {x1, y1, x2, y2}


class YOLODetector:
    """YOLO-based object detector and tracker using ultralytics."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        allowed_classes: Optional[List[int]] = None,
    ):
        settings = get_settings()
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.confidence_threshold = confidence_threshold or settings.YOLO_CONFIDENCE_THRESHOLD
        self.allowed_classes = allowed_classes or settings.yolo_allowed_classes_list
        self.model = None

        # Read the process-wide device decision made at startup (cuda / mps / cpu).
        from app.utils.device import get_device
        self.device = get_device()
        logger.info(f"YOLO detector using device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"YOLO model loaded on {self.device}: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {e}")
            logger.warning("YOLO model not available. Detection will return empty results.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=self.allowed_classes if self.allowed_classes else None,
                device=self.device,
                verbose=False,
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy()

                    detection = DetectionResult(
                        class_id=cls_id,
                        class_name=result.names.get(cls_id, "unknown"),
                        confidence=conf,
                        bbox={
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        },
                    )
                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def detect_persons(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect only person class (class_id=0 in COCO)."""
        all_detections = self.detect(frame)
        return [d for d in all_detections if d.class_id == 0]

    def track(self, frame: np.ndarray) -> List[TrackedDetection]:
        """
        Run detection and tracking on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of TrackedDetection objects
        """
        if self.model is None:
            return []

        try:
            results = self.model.track(
                frame,
                classes=[0],  # only person tracking is supported/needed
                conf=self.confidence_threshold,
                persist=True,
                tracker="bytetrack.yaml",
                device=self.device,
                verbose=False,
            )

            tracked_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                for i in range(len(boxes)):
                    track_id = track_ids[i]
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy()

                    tracked_det = TrackedDetection(
                        track_id=track_id,
                        confidence=conf,
                        bbox={
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        },
                    )
                    tracked_detections.append(tracked_det)

            return tracked_detections

        except Exception as e:
            logger.error(f"YOLO tracking failed: {e}")
            return []

    def reset_tracker(self):
        """Reset the internal tracker state."""
        try:
            if self.model is not None and hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers:
                    for tracker in self.model.predictor.trackers:
                        if hasattr(tracker, "reset"):
                            tracker.reset()
                    logger.info("YOLO tracking states reset successfully.")
        except Exception as e:
            logger.error(f"Error resetting YOLO tracker: {e}")
