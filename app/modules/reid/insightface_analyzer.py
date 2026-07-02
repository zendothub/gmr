"""InsightFace analyzer module for demographic classification (age/gender)."""

import threading
from dataclasses import dataclass
from typing import Optional, Dict
import cv2
import numpy as np
from loguru import logger

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
    kps: Optional[np.ndarray] = None


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
            from app.utils.device import insightface_ctx_id, get_device, get_insightface_providers

            device = get_device()
            ctx_id = insightface_ctx_id()
            providers = get_insightface_providers()
            logger.info(
                f"InsightFace using ctx_id={ctx_id} device={device} "
                f"providers={providers}"
            )

            # Pass providers= to the constructor so ONNX Runtime uses the correct
            # execution backend from the start:
            #   CUDA  → CUDAExecutionProvider
            #   MPS   → CoreMLExecutionProvider  (Apple Neural Engine)
            #   CPU   → CPUExecutionProvider
            self.app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=['detection', 'genderage', 'recognition'],
                providers=providers,
            )
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

            # Filter faces to avoid background face contamination
            valid_faces = []
            h, w = crop.shape[:2]
            for f in faces:
                bbox = f.bbox
                f_x1, f_y1, f_x2, f_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                f_w = f_x2 - f_x1
                f_h = f_y2 - f_y1
                f_y_center = (f_y1 + f_y2) / 2

                # Heuristic 1: Face center must be in the upper region of the body crop (top 45%)
                if f_y_center > h * 0.45:
                    logger.debug(f"InsightFace: face rejected (y_center {f_y_center:.1f} > {h * 0.45:.1f})")
                    continue

                # Heuristic 2: Face must not be too small (minimum 8% of body crop width/height)
                if f_w < w * 0.08 and f_h < h * 0.08:
                    logger.debug(f"InsightFace: face rejected (too small: {f_w:.1f}x{f_h:.1f} vs crop {w}x{h})")
                    continue

                valid_faces.append(f)

            if not valid_faces:
                logger.debug(f"InsightFace: no faces passed heuristics from {len(faces)} detected")
                return None

            # Select the face whose horizontal centre is closest to the crop's vertical centreline.
            # This ensures the tracked person's face wins over an adjacent person's face in
            # close-range scenarios, regardless of detection confidence scores.
            crop_cx = w / 2
            best_face = min(
                valid_faces,
                key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - crop_cx)
            )
            logger.debug(
                f"InsightFace: selected face at x_center "
                f"{(best_face.bbox[0] + best_face.bbox[2]) / 2:.1f} "
                f"(crop_cx={crop_cx:.1f}, score={best_face.det_score:.2f}) "
                f"from {len(valid_faces)} valid face(s)"
            )
            
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
                face_crop=face_crop,
                kps=getattr(best_face, "kps", None)
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
