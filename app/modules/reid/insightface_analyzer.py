"""InsightFace analyzer module for demographic classification (age/gender)."""

import threading
from dataclasses import dataclass
from typing import Optional, Dict
import cv2
import numpy as np
from loguru import logger
import torch

from app.config import get_settings

# Shared analyzer instances so weights aren't loaded multiple times
_shared_analyzers: Dict[str, "InsightFaceAnalyzer"] = {}
_shared_lock = threading.Lock()


def get_shared_analyzer(model_name: Optional[str] = None) -> "InsightFaceAnalyzer":
    """Get (or lazily create) a process-wide shared InsightFace analyzer."""
    settings = get_settings()
    key = model_name or settings.INSIGHTFACE_MODEL
    with _shared_lock:
        if key not in _shared_analyzers:
            _shared_analyzers[key] = InsightFaceAnalyzer(model_name=key)
            logger.info(f"Shared InsightFace analyzer created for {key}")
        return _shared_analyzers[key]


@dataclass
class InsightFaceResult:
    """Demographics and recognition result for a face."""
    age: int
    gender: str  # "M" or "F"
    age_group: str  # "child", "young_adult", "adult", "senior"
    face_score: float
    face_bbox: dict  # {"x1", "y1", "x2", "y2"}
    embedding: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None


class InsightFaceAnalyzer:
    """Demographics analyzer using InsightFace."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.INSIGHTFACE_MODEL
        try:
            width, height = map(int, settings.INSIGHTFACE_DET_SIZE.split(","))
            self.det_size = (width, height)
        except Exception:
            self.det_size = (640, 640)
        self.app = None
        self._load_model()

    def _load_model(self):
        """Lazy load InsightFace app."""
        try:
            from insightface.app import FaceAnalysis

            # Determine execution device: use GPU if CUDA is available, otherwise CPU.
            ctx_id = 0 if torch.cuda.is_available() else -1
            logger.info(f"InsightFace using ctx_id={ctx_id} (CUDA={torch.cuda.is_available()})")

            # buffalo_l is the default high-accuracy model pack
            self.app = FaceAnalysis(name=self.model_name, allowed_modules=['detection', 'genderage', 'recognition'])
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info(f"InsightFace FaceAnalysis prepared successfully (model={self.model_name})")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace analyzer: {e}")
            self.app = None

    def analyze(self, crop: np.ndarray) -> Optional[InsightFaceResult]:
        """
        Run face detection, demographic analysis, and face embedding extraction on a person crop.

        Args:
            crop: BGR person crop image

        Returns:
            InsightFaceResult if a face is detected, else None
        """
        if self.app is None or crop is None or crop.size == 0:
            return None

        try:
            # InsightFace expects RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the crop
            faces = self.app.get(rgb_crop)
            if not faces:
                return None

            # Get the face with the highest detection score
            best_face = max(faces, key=lambda f: f.det_score)
            
            bbox = best_face.bbox
            face_bbox = {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
            }
            
            age = int(best_face.age)
            # InsightFace: gender=1 for Male, gender=0 for Female
            gender_val = getattr(best_face, "gender", -1)
            gender = "M" if gender_val == 1 else "F"
            
            # Extract face crop for saving
            h, w = crop.shape[:2]
            x1_crop, y1_crop = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x2_crop, y2_crop = min(w, int(bbox[2])), min(h, int(bbox[3]))
            face_crop = crop[y1_crop:y2_crop, x1_crop:x2_crop] if y2_crop > y1_crop and x2_crop > x1_crop else None

            return InsightFaceResult(
                age=age,
                gender=gender,
                age_group=self._age_to_group(age),
                face_score=float(best_face.det_score),
                face_bbox=face_bbox,
                embedding=getattr(best_face, "embedding", None),
                face_crop=face_crop
            )

        except Exception as e:
            logger.error(f"InsightFace demographic analysis failed: {e}")
            return None

    def _age_to_group(self, age: int) -> str:
        """Map age integer to demographic age group category."""
        if age < 12:
            return "child"
        elif age < 25:
            return "young_adult"
        elif age < 60:
            return "adult"
        else:
            return "senior"
