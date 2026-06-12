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
        self._input_size = (128, 256)  # (width, height)
        self._load_model()

    def _load_model(self):
        """Load OSNet model."""
        try:
            import torch
            import torchvision.transforms as T

            # TODO: Load actual torchreid OSNet model weights
            # from torchreid.utils import FeatureExtractor
            # self.model = FeatureExtractor(
            #     model_name='osnet_x1_0',
            #     model_path=self.model_path,
            #     device='cuda' if torch.cuda.is_available() else 'cpu'
            # )

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self._input_size[::-1]),  # (H, W)
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            logger.info(f"OSNet extractor initialized (device={self.device})")
            logger.warning("OSNet model weights not loaded - using random embeddings as stub. "
                         "Place real weights at: " + self.model_path)

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
        try:
            if crop is None or crop.size == 0:
                return None

            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            if self.model is not None:
                # Use actual model
                # features = self.model(rgb_crop)
                # embedding = features.cpu().numpy().flatten()
                pass

            # TODO: Replace stub with actual model inference
            # Stub: Generate deterministic embedding based on pixel content
            import torch
            tensor = self.transform(rgb_crop).unsqueeze(0)

            # Simple feature extraction stub using mean pooling of pixel values
            np.random.seed(int(np.mean(rgb_crop) * 1000) % (2**31))
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)

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
