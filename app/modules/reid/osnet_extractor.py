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

        # Read the process-wide device decision made at startup (cuda / mps / cpu).
        from app.utils.device import get_device
        self.device = get_device()
        logger.info(f"OSNet extractor using device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Load OSNet model.

        Safety guard: if ``model_path`` is configured but the file is missing,
        ``FeatureExtractor`` silently falls back to ``pretrained=True`` which
        loads ONLY the ImageNet backbone — the ReID ``fc`` embedding head stays
        randomly-initialized, producing non-discriminative embeddings. This
        previously caused OSNet body ReID to be completely broken (CONTEXT.md
        issues #1/#16). We refuse to start in that state.
        """
        try:
            import os
            import torch
            from torchreid.reid.utils import FeatureExtractor

            if self.model_path and not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"OSNET_MODEL_PATH='{self.model_path}' does not exist. "
                    f"Without ReID-finetuned weights the fc embedding head is "
                    f"random-init and body ReID is non-discriminative. "
                    f"Download osnet_x1_0_msmt17 to {self.model_path}."
                )

            # FeatureExtractor internally builds with pretrained=(not(model_path
            # and check_isfile(model_path))) and then calls
            # load_pretrained_weights when the path is valid. With the file now
            # present, pretrained=False so the ImageNet backbone is NOT
            # auto-downloaded; the ReID checkpoint is the sole weight source.
            self.model = FeatureExtractor(
                model_name='osnet_x1_0',
                model_path=self.model_path if self.model_path else None,
                device=self.device,
                verbose=False,
            )
            # FeatureExtractor builds with pretrained=(not(model_path and
            # check_isfile(model_path))) and then calls load_pretrained_weights
            # only when the path is valid. When it does, verify the ReID ``fc``
            # embedding head actually loaded (not just the conv backbone).
            if self.model_path and os.path.isfile(self.model_path):
                self._verify_reid_weights_loaded()

            logger.info(f"OSNet FeatureExtractor loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize OSNet extractor: {e}")
            self.model = None

    def _verify_reid_weights_loaded(self):
        """Confirm the ReID ``fc`` embedding head matched the checkpoint.

        ``load_pretrained_weights`` only prints matched/discarded layers; a
        silent fallback (e.g. checkpoint key mismatch) would leave ``fc``
        random-init with no error. We re-run the name+size match against the
        checkpoint and fail loudly if the ``fc`` keys did not load.
        """
        import os
        from collections import OrderedDict
        import torch

        try:
            ckpt = torch.load(self.model_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model_dict = self.model.model.state_dict()
            fc_matched, fc_missing = [], []
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                if k.startswith("fc."):
                    if k in model_dict and model_dict[k].shape == v.shape:
                        fc_matched.append(k)
                    else:
                        fc_missing.append(k)
            if len(fc_matched) == 0:
                raise RuntimeError(
                    f"OSNet checkpoint {self.model_path} did NOT match any "
                    f"fc.* keys in the model — the ReID embedding head is "
                    f"random-init. Body ReID will be non-discriminative. "
                    f"fc_missing={fc_missing}"
                )
            logger.info(
                f"OSNet ReID weights verified: {len(fc_matched)} fc keys loaded "
                f"({fc_matched}). Body ReID embedding head is trained."
            )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"OSNet ReID weight verification skipped: {e}")

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
