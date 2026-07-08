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
    gender: str        # "M" or "F"
    age_group: str     # "child", "young_adult", "adult", "senior"
    face_score: float  # InsightFace raw detection confidence
    face_bbox: dict    # {"x1", "y1", "x2", "y2"}
    embedding: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None
    kps: Optional[np.ndarray] = None
    face_quality: float = 0.0      # composite quality (det_score × frontality)
    frontality_score: float = 0.0  # 0 = profile, 1 = perfectly frontal
    eye_spread: float = 0.0        # normalised eye-to-eye horizontal distance


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
                allowed_modules=['detection', 'recognition'],  # gender/age now handled by MiVOLO
                providers=providers,
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info(f"InsightFace FaceAnalysis prepared successfully (model={self.model_name})")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace analyzer: {e}")
            self.app = None

    def detect_all_faces(self, frame: np.ndarray) -> list[dict]:
        """Run SCRFD detection + ArcFace embedding on the FULL frame (not a body crop).

        Returns a flat list of raw face detections in full-frame pixel coordinates.
        Each dict has keys: ``bbox`` (x1,y1,x2,y2), ``embedding`` (512-dim),
        ``kps`` (5-point landmarks), ``det_score`` (float).

        The caller is responsible for matching faces to body tracks by checking
        whether a face centre falls inside a body bounding box.
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb_frame)
            if not faces:
                return []

            h, w = frame.shape[:2]
            detections = []
            for f in faces:
                bbox = f.bbox
                f_x1, f_y1, f_x2, f_y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                f_w = f_x2 - f_x1
                f_h = f_y2 - f_y1

                # Skip tiny / implausible faces
                if f_w < 5 or f_h < 5:
                    continue
                # Skip faces at the very edge of the frame
                if f_x1 < 0 or f_y1 < 0 or f_x2 > w or f_y2 > h:
                    continue

                detections.append({
                    "bbox": {"x1": f_x1, "y1": f_y1, "x2": f_x2, "y2": f_y2},
                    "embedding": getattr(f, "embedding", None),
                    "kps": getattr(f, "kps", None),
                    "det_score": float(f.det_score),
                })
            return detections
        except Exception as e:
            logger.error(f"Full-frame face detection failed: {e}")
            return []

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

            # Select the best face using a multi-signal heuristic that favours
            # the TRACKED person's own face and penalises adjacent-person contamination.
            # Signals: face size (large = own face), centreline proximity, upper-position,
            # detection confidence.
            #
            # Old logic (removed): face closest to crop horizontal centre wins.
            # This failed when two people stood shoulder-to-shoulder — the adjacent
            # person's face could appear more centred in the tracked person's crop.
            def _face_score(f) -> float:
                f_x1, f_y1, f_x2, f_y2 = f.bbox
                f_cx = (f_x1 + f_x2) / 2.0
                f_w  = f_x2 - f_x1
                f_h  = f_y2 - f_y1
                face_area = f_w * f_h
                crop_area  = w * h

                # Larger face relative to crop → more likely tracked person's own face
                size_score = min(1.0, face_area / (crop_area * 0.08))
                # Closer to horizontal centre
                centre_dev = abs(f_cx - w / 2.0) / (w / 2.0)
                centre_score = max(0.0, 1.0 - centre_dev)
                # Face should be in the upper region of the body crop
                f_cy = (f_y1 + f_y2) / 2.0
                upper_score = max(0.0, 1.0 - f_cy / (h * 0.40))

                # Weighted geometric score × detection confidence
                geo = 0.40 * size_score + 0.35 * centre_score + 0.25 * upper_score
                return float(f.det_score) * geo

            crop_cx = w / 2.0
            best_face = max(valid_faces, key=_face_score)
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

            # Gender/age handled by MiVOLO — InsightFace only does detection + embedding
            age = None
            gender = None

            # Extract face crop with 30% padding for better face recognition
            from app.utils.image_utils import extract_crop
            face_crop = extract_crop(crop, face_bbox, padding_pct=0.30)

            from app.modules.reid.crop_quality import assess_face_quality

            kps = getattr(best_face, "kps", None)

            result_obj = InsightFaceResult(
                age=age,
                gender=gender,
                age_group=self._age_to_group(age) if age is not None else None,
                face_score=float(best_face.det_score),
                face_bbox=face_bbox,
                embedding=getattr(best_face, "embedding", None),
                face_crop=face_crop,
                kps=kps,
            )

            # Compute frontality metrics from keypoints before calling assess_face_quality
            # so that the quality function can read them via result_obj.kps / result_obj.face_bbox
            if kps is not None and len(kps) >= 2:
                face_w = max(face_bbox["x2"] - face_bbox["x1"], 1.0)
                lx = float(kps[0][0])
                rx = float(kps[1][0])
                result_obj.eye_spread = abs(rx - lx) / face_w
            else:
                result_obj.eye_spread = 0.0

            result_obj.face_quality = assess_face_quality(result_obj)

            # Recompute frontality_score and expose it for camera_worker to use
            # as the gating value (replaces the disabled FACE_MIN_EYE_SPREAD check)
            face_w = max(face_bbox["x2"] - face_bbox["x1"], 1.0)
            face_h = max(face_bbox["y2"] - face_bbox["y1"], 1.0)
            face_cx = (face_bbox["x1"] + face_bbox["x2"]) / 2.0
            if kps is not None and len(kps) >= 5:
                spread_score = min(1.0, result_obj.eye_spread / 0.35)
                nose_cx = float(kps[2][0])
                nose_offset = abs(nose_cx - face_cx) / (face_w / 2.0)
                nose_score = max(0.0, 1.0 - nose_offset)
                eye_vert_diff = abs(float(kps[1][1]) - float(kps[0][1])) / face_h
                sym_score = max(0.0, 1.0 - eye_vert_diff * 4.0)
                result_obj.frontality_score = 0.55 * spread_score + 0.30 * nose_score + 0.15 * sym_score
            elif kps is not None and len(kps) >= 2:
                result_obj.frontality_score = min(1.0, result_obj.eye_spread / 0.35)
            else:
                result_obj.frontality_score = 0.5  # unknown

            logger.debug(
                f"InsightFace: det={result_obj.face_score:.2f}  "
                f"eye_spread={result_obj.eye_spread:.2f}  "
                f"frontality={result_obj.frontality_score:.2f}  "
                f"quality={result_obj.face_quality:.2f}"
            )
            return result_obj

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
