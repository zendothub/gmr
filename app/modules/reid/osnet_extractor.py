"""OSNet feature extractor for ReID embeddings."""

import threading
from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings

# Shared extractor instances (one per model path) so multiple camera workers
# don't each load their own copy of the model.
_shared_extractors: Dict[str, "OSNetExtractor"] = {}
_shared_lock = threading.Lock()


def get_shared_extractor(model_path: Optional[str] = None) -> "OSNetExtractor":
    """Get (or lazily create) a process-wide shared OSNet extractor."""
    settings = get_settings()
    key = model_path or settings.OSNET_MODEL_PATH
    with _shared_lock:
        if key not in _shared_extractors:
            _shared_extractors[key] = OSNetExtractor(model_path=key)
            logger.info(f"Shared OSNet extractor created for {key}")
        return _shared_extractors[key]


class OSNetExtractor:
    """Extract 512-dimensional ReID embeddings using OSNet."""

    def __init__(self, model_path: Optional[str] = None):
        settings = get_settings()
        self.model_path = model_path or settings.OSNET_MODEL_PATH
        self.embedding_dim = settings.REID_EMBEDDING_DIM
        self.model = None
        
        # Detect device: cuda -> mps -> cpu
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        logger.info(f"OSNet extractor device set to: {self.device}")
        self._load_model()

    def _load_model(self):
        """Load OSNet model."""
        try:
            from torchreid.reid.utils import FeatureExtractor
            # Note: FeatureExtractor automatically handles resizing and transforms
            self.model = FeatureExtractor(
                model_name='osnet_x1_0',
                model_path=self.model_path if self.model_path else None,
                device=self.device
            )
            logger.info(f"OSNet FeatureExtractor loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize OSNet extractor: {e}")
            self.model = None

    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dimensional embedding from a person crop.

        Args:
            crop: BGR person crop image

        Returns:
            512-dim numpy array (L2 normalized), or None on failure
        """
        if self.model is None:
            return None

        try:
            if crop is None or crop.size == 0:
                return None

            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            import torch
            with torch.no_grad():
                features = self.model(rgb_crop)
                embedding = features[0].cpu().numpy()  # shape (512,)
                
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def extract_batch(self, crops: list) -> list:
        """Extract embeddings for a batch of crops."""
        return [self.extract(crop) for crop in crops if crop is not None]
