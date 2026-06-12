"""Tests for ReID crop quality assessment."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from app.modules.reid.crop_quality import assess_crop_quality


class TestCropQuality:
    def test_none_crop(self):
        assert assess_crop_quality(None) == 0.0

    def test_empty_crop(self):
        crop = np.zeros((0, 0, 3), dtype=np.uint8)
        assert assess_crop_quality(crop) == 0.0

    def test_too_small_crop(self):
        crop = np.random.randint(0, 255, (40, 20, 3), dtype=np.uint8)
        assert assess_crop_quality(crop) == 0.0

    def test_dark_crop_scores_low(self):
        crop = np.zeros((256, 128, 3), dtype=np.uint8)  # all black
        score = assess_crop_quality(crop)
        assert 0.0 <= score < 0.7

    def test_textured_crop_scores_in_range(self):
        crop = np.random.randint(60, 200, (256, 128, 3), dtype=np.uint8)
        score = assess_crop_quality(crop)
        assert 0.0 <= score <= 1.0