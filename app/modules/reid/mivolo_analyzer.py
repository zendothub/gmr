"""MiVOLO gender + age analyzer — standalone ViT-Small (103 MB).

Uses the official MiVOLO-D1 checkpoint (ViT-Small backbone, 384-dim).
Supports both 1-output (regression) and 3-output (classification) head
formats from the official Google Drive checkpoints.

Auto-downloads from Google Drive on first use (default: FairFace classification).

Accuracy (FairFace, 3-class): ~95-97% gender, similar age performance as the
larger VOLO variant at 1/3 the size.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from app.config import get_settings

try:
    import timm
except ImportError:
    raise ImportError("timm is required for MiVOLO. Run: pip install timm")

_shared_analyzers: Dict[str, "MiVOLOAnalyzer"] = {}
_shared_lock = threading.Lock()


def get_shared_mivolo(model_path: Optional[str] = None) -> "MiVOLOAnalyzer":
    """Get (or lazily create) a process-wide shared MiVOLO analyzer."""
    settings = get_settings()
    key = model_path or settings.MIVOLO_MODEL_PATH
    with _shared_lock:
        if key not in _shared_analyzers:
            _shared_analyzers[key] = MiVOLOAnalyzer(model_path=key)
            logger.info(f"Shared MiVOLO analyzer created: {key}")
        return _shared_analyzers[key]


class MiVOLOModel(nn.Module):
    """MiVOLO-D1 (ViT-Small, 384-dim) gender + age model.

    Supports both head formats from the official checkpoints:
      • Regression (1-output): head/aux_head are Linear(384→1).  Age is a
        single scalar value; gender uses sigmoid on aux_head.
      • Classification (3-output): head/aux_head are Linear(384→3).  Age
        uses 3 age-group logits via softmax; gender uses 3-class softmax.
    """

    def __init__(self, embed_dim: int = 384, num_age: int = 1, num_gender: int = 1):
        super().__init__()
        self.num_age = num_age
        self.num_gender = num_gender
        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=False, num_classes=0
        )
        self.head = nn.Linear(embed_dim, num_age)
        self.aux_head = nn.Linear(embed_dim, num_gender)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (age_output, gender_output)."""
        features = self.backbone(x)
        return self.head(features), self.aux_head(features)


class MiVOLOAnalyzer:
    """Gender + age estimator using a MiVOLO-D1 checkpoint."""

    def __init__(self, model_path: Optional[str] = None):
        settings = get_settings()
        self.model_path = model_path or settings.MIVOLO_MODEL_PATH
        self._min_age: float = 0.0
        self._max_age: float = 122.0
        self._avg_age: float = 61.0
        self.model: Optional[MiVOLOModel] = None

        from app.utils.device import get_device
        self.device = get_device()
        self._load_model()

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _download_model(self) -> str:
        """Download the MiVOLO checkpoint if not already present."""
        import gdown

        target = self.model_path
        os.makedirs(os.path.dirname(target), exist_ok=True)

        if os.path.exists(target):
            logger.info(f"MiVOLO checkpoint found: {target}")
            return target

        # MiVOLO-D1 FairFace (3-out classification, ~103 MB)
        gdrive_id = "1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP"
        logger.info("Downloading MiVOLO checkpoint (103 MB) from Google Drive …")
        gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", target, quiet=False)
        if not os.path.exists(target):
            raise FileNotFoundError(f"MiVOLO download failed: {target}")
        logger.info(f"MiVOLO checkpoint downloaded: {target}")
        return target

    def _load_model(self) -> None:
        """Load the model architecture and trained weights."""
        try:
            self._download_model()

            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            meta = checkpoint

            self._min_age = float(meta.get("min_age", 0.0))
            self._max_age = float(meta.get("max_age", 122.0))
            self._avg_age = float(meta.get("avg_age", 61.0))

            state_dict = meta["state_dict"]
            head_w  = state_dict["head.weight"]
            aux_w   = state_dict["aux_head.weight"]
            num_age    = head_w.shape[0]
            num_gender = aux_w.shape[0]

            m = MiVOLOModel(num_age=num_age, num_gender=num_gender)
            m.load_state_dict(state_dict, strict=False)
            m.to(self.device)
            m.eval()
            self.model = m

            logger.info(
                f"MiVOLO model loaded on {self.device} "
                f"(heads: age={num_age}out, gender={num_gender}out, "
                f"age_range=[{self._min_age:.0f},{self._max_age:.0f}])"
            )
        except Exception as e:
            logger.error(f"Failed to load MiVOLO model: {e}")
            self.model = None

    def analyze(self, face_crop: np.ndarray) -> Optional[Dict[str, object]]:
        """Predict gender and age from a BGR face crop (numpy H×W×3)."""
        if self.model is None or face_crop is None or face_crop.size == 0:
            return None

        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
            tensor = tensor.div(255.0)
            tensor = (tensor - self._MEAN) / self._STD
            tensor = tensor.to(self.device)

            with torch.no_grad():
                age_out, gender_out = self.model(tensor)

            if self.model.num_age == 1:
                age_value = float(age_out.item())
                age_val = round(age_value * (self._max_age - self._min_age) + self._avg_age)
            else:
                age_probs = torch.softmax(age_out, dim=-1)[0].cpu().numpy()
                age_midpoints = [10.0, 40.0, 75.0]
                age_float = float((age_probs * age_midpoints).sum())
                age_val = round(age_float)
                age_value = age_float
            age_val = max(int(self._min_age), min(int(self._max_age), int(age_val)))

            if self.model.num_gender == 1:
                gender_prob = float(torch.sigmoid(gender_out).item())
                gender = "F" if gender_prob > 0.5 else "M"
            else:
                gender_logits = torch.softmax(gender_out, dim=-1)[0].cpu().numpy()
                gender_idx = int(gender_logits.argmax())
                gender = "F" if gender_idx == 1 else "M"
                female_prob = float(gender_logits[1]) if len(gender_logits) > 1 else 0.0
                gender_prob = female_prob

            gender_confidence = max(gender_prob, 1.0 - gender_prob)

            return {
                "gender": gender,
                "age": int(age_val),
                "age_float": float(age_value),
                "gender_prob": gender_prob,
                "gender_confidence": float(gender_confidence),
            }
        except Exception as e:
            logger.error(f"MiVOLO inference failed: {e}")
            return None

    def _age_to_group(self, age: int) -> str:
        if age is None or age < 12:
            return "child"
        elif age < 25:
            return "young_adult"
        elif age < 60:
            return "adult"
        else:
            return "senior"
