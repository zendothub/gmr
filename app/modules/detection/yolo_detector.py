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


@dataclass
class DetectionResult:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: dict  # {x1, y1, x2, y2}


class YOLODetector:
    """YOLO-based object detector using ultralytics."""

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
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {self.model_path}: {e}")
            # TODO: Download model weights if not present
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
