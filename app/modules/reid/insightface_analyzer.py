"""InsightFace analyzer module for demographic classification (age/gender) and anti-spoofing."""

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
    age: Optional[int]
    gender: Optional[str]        # "M" or "F" (product gender uses SigLIP2)
    age_group: Optional[str]     # "child", "young_adult", "adult", "senior"
    face_score: float  # InsightFace raw detection confidence
    face_bbox: dict    # {"x1", "y1", "x2", "y2"}
    embedding: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None
    kps: Optional[np.ndarray] = None
    face_quality: float = 0.0      # composite quality (det_score × frontality)
    frontality_score: float = 0.0  # 0 = profile, 1 = perfectly frontal
    eye_spread: float = 0.0        # normalised eye-to-eye horizontal distance
    antispoof_score: float = 0.0   # 0 = real live person, 1 = spoof/attack


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
        self._spoof_model = None  # Anti-spoofing model (loaded lazily)
        self._load_model()
        self._load_spoof_model()

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
            # genderage supplies age (product gender still uses SigLIP2 face+margin)
            self.app = FaceAnalysis(
                name=self.model_name,
                # Include antispoofing so the model file gets downloaded with the
                # rest of the buffalo_l pack. The model is loaded separately in
                # _load_spoof_model() to avoid interfering with face detection.
                allowed_modules=["detection", "recognition", "genderage"],
                providers=providers,
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info(
                f"InsightFace FaceAnalysis prepared successfully "
                f"(model={self.model_name}, modules={list(self.app.models.keys())})"
            )

            # Anti-spoofing now uses texture analysis (Laplacian + FFT + colour)
            # instead of the missing antispoofing.onnx.  No model download needed.
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace analyzer: {e}")
            self.app = None

    def _load_spoof_model(self):
        """Initialise texture-based anti-spoofing (no external ONNX model required).

        This replaces the missing InsightFace antispoofing.onnx which is NOT
        included in any standard model pack.  Instead we use multi-signal
        image-analysis:

        1. **Laplacian variance** — real faces have natural texture; printed
           photos are blurry, screens show moiré patterns.
        2. **FFT high-frequency energy** — screens emit periodic pixel-grid
           artefacts visible in the frequency domain.
        3. **Colour-space analysis** — printed/screen faces have narrower
           colour gamut and different saturation distribution.

        Combined these catch the primary attendance-fraud vectors (phone
        screen or printed photo held in front of the camera).
        """
        settings = get_settings()
        if not settings.ANTISPOOF_ENABLED:
            self._spoof_model = None
            logger.info("Anti-spoofing disabled (ANTISPOOF_ENABLED=False)")
            return

        # No model file needed — texture analysis uses OpenCV only
        self._spoof_model = "texture"  # sentinel so detect_spoof() knows it's active
        logger.info(
            "Anti-spoofing ACTIVE — texture-based analysis (Laplacian + FFT + colour)"
        )
        print("=" * 60)
        print("  ✅ ANTI-SPOOFING ACTIVATED — Texture-based liveness detection")
        print(f"  🔧 Threshold: {settings.ANTISPOOF_THRESHOLD}")
        print(f"  👁️  Audit-only: {settings.ANTISPOOF_AUDIT_ONLY}")
        print(f"  🛡️  Spoofed faces will be blocked from attendance")
        print("=" * 60)

    def detect_spoof(self, face_crop: np.ndarray) -> float:
        """Texture-based anti-spoofing — no external model required.

        Uses three complementary signals to detect printed photos and phone
        screens held in front of the camera:

        1. Laplacian variance (texture sharpness / moiré)
        2. FFT high-frequency energy ratio (screen pixel-grid artefacts)
        3. Colour saturation spread (print / screen gamut compression)

        Args:
            face_crop: BGR face crop image.

        Returns:
            spoof_score: float 0.0 (real) → 1.0 (spoof).
                         0.5 when analysis is unavailable.
        """
        if self._spoof_model is None or face_crop is None or face_crop.size == 0:
            return 0.5  # anti-spoofing disabled or bad crop

        try:
            h, w = face_crop.shape[:2]
            if h < 40 or w < 40:
                return 0.5  # too small for reliable analysis

            # Resize to consistent dimensions for stable thresholds
            std = cv2.resize(face_crop, (128, 128))
            gray = cv2.cvtColor(std, cv2.COLOR_BGR2GRAY).astype(np.float64)

            signals = []

            # ── Signal 1: Laplacian variance ─────────────────────────────
            # Measured on 128×128 grayscale face crops:
            #   Real face (camera):   200 – 5000  (normal texture)
            #   Blurry print:         < 80        (ink blur / low-res)
            #   Screen with moiré:    > 8000      (periodic pixel-grid)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < 80:
                signals.append(0.85)   # very blurry → likely print
            elif lap_var > 8000:
                signals.append(0.80)   # extreme texture → moiré screen
            else:
                signals.append(0.05)   # normal → real face

            # ── Signal 2: Spectral peak detection (screen grid) ──────────
            # Instead of raw HF energy (which is high for all images),
            # detect periodic peaks: screens produce sharp spectral spikes
            # at pixel-grid frequencies.  We measure kurtosis of the
            # magnitude spectrum — uniform spectrum (real) has low kurtosis,
            # peaked spectrum (screen) has high kurtosis.
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))
            mag_flat = magnitude.ravel()
            mag_mean = mag_flat.mean()
            mag_std = mag_flat.std() + 1e-6
            # Excess kurtosis: normal distribution ≈ 0, peaked > 3
            kurtosis = ((mag_flat - mag_mean) ** 4).mean() / (mag_std ** 4) - 3.0
            if kurtosis > 8.0:
                signals.append(0.75)   # peaked spectrum → screen artefacts
            else:
                signals.append(0.05)

            # ── Signal 3: Colour saturation analysis ─────────────────────
            # Real skin has varied saturation (std > 20).
            # Prints have flat saturation (low std, washed out).
            # Screens can over-saturate (high mean, narrow std).
            hsv = cv2.cvtColor(std, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1].astype(np.float64)
            sat_mean = sat.mean()
            sat_std = sat.std()
            if sat_std < 10:
                signals.append(0.75)   # flat saturation → print / mono screen
            elif sat_mean > 170:
                signals.append(0.65)   # extreme saturation → screen
            else:
                signals.append(0.05)

            # ── Combine (weighted average) ───────────────────────────────
            # Laplacian is the most reliable single signal (weight 0.50).
            weights = [0.50, 0.25, 0.25]
            score = sum(w * s for w, s in zip(weights, signals))

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Texture anti-spoofing failed: {e}")
            return 0.5

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

                age_val = getattr(f, "age", None)
                if age_val is not None:
                    try:
                        age_val = int(round(float(age_val)))
                    except (TypeError, ValueError):
                        age_val = None

                detections.append({
                    "bbox": {"x1": f_x1, "y1": f_y1, "x2": f_x2, "y2": f_y2},
                    "embedding": getattr(f, "embedding", None),
                    "kps": getattr(f, "kps", None),
                    "det_score": float(f.det_score),
                    "age": age_val,
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

            # Age from genderage head (product gender uses SigLIP2)
            age = None
            raw_age = getattr(best_face, "age", None)
            if raw_age is not None:
                try:
                    age = int(round(float(raw_age)))
                except (TypeError, ValueError):
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

            # ── Anti-spoofing inference ──────────────────────────────────
            if face_crop is not None and face_crop.size > 0:
                result_obj.antispoof_score = self.detect_spoof(face_crop)
                if get_settings().ANTISPOOF_ENABLED:
                    logger.debug(
                        f"Anti-spoof: score={result_obj.antispoof_score:.3f} "
                        f"(0=real, 1=spoof)"
                    )

            logger.debug(
                f"InsightFace: det={result_obj.face_score:.2f}  "
                f"eye_spread={result_obj.eye_spread:.2f}  "
                f"frontality={result_obj.frontality_score:.2f}  "
                f"quality={result_obj.face_quality:.2f}  "
                f"spoof={result_obj.antispoof_score:.3f}"
            )
            return result_obj

        except Exception as e:
            logger.error(f"InsightFace demographic analysis failed: {e}")
            return None

    def _age_to_group(self, age: Optional[int]) -> Optional[str]:
        """Map age integer to demographic age group category."""
        if age is None:
            return None
        if age < 12:
            return "child"
        elif age < 25:
            return "young_adult"
        elif age < 60:
            return "adult"
        else:
            return "senior"

    def estimate_age_from_crop(self, face_crop: np.ndarray) -> Optional[int]:
        """Run genderage on a face (or near-face) crop; return age years or None.

        Used by offline backfill scripts when only MinIO face crops are available.
        """
        if self.app is None or face_crop is None or face_crop.size == 0:
            return None
        try:
            bgr = face_crop
            if len(bgr.shape) == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            faces = self.app.get(bgr)
            if not faces:
                h, w = bgr.shape[:2]
                ph, pw = int(h * 0.35), int(w * 0.35)
                padded = cv2.copyMakeBorder(bgr, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
                faces = self.app.get(padded)
            if not faces:
                return None
            best = max(faces, key=lambda x: float(x.det_score))
            return int(round(float(best.age)))
        except Exception as e:
            logger.error(f"InsightFace age-from-crop failed: {e}")
            return None