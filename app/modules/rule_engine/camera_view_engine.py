"""Camera view engine - filters detections based on active camera views."""

from typing import List, Tuple, Optional
from loguru import logger

from app.utils.geometry import point_in_polygon, polygon_from_json, bbox_center


class CameraViewEngine:
    """Checks if detections fall within active camera views/ROI."""

    def __init__(self):
        self._active_views: List[dict] = []

    def load_views(self, views: List[dict]):
        """Load active camera views into memory."""
        self._active_views = [v for v in views if v.get("is_active", True)]
        logger.debug(f"Loaded {len(self._active_views)} active camera views")

    def is_detection_in_view(self, bbox: dict) -> bool:
        """
        Check if a detection's center point is inside any active camera view.
        If no views are configured, accept all detections (full_frame).
        """
        if not self._active_views:
            return True  # No views configured = full frame

        center = bbox_center(bbox)

        for view in self._active_views:
            if view.get("view_type") == "ignore_area":
                poly_points = polygon_from_json(view.get("polygon"))
                if poly_points and point_in_polygon(center, poly_points):
                    return False  # Detection is in ignore area

        # Check if in any non-ignore view
        non_ignore_views = [v for v in self._active_views if v.get("view_type") != "ignore_area"]

        if not non_ignore_views:
            return True  # Only ignore areas configured

        for view in non_ignore_views:
            if view.get("view_type") == "full_frame":
                return True
            poly_points = polygon_from_json(view.get("polygon"))
            if poly_points and point_in_polygon(center, poly_points):
                return True

        return False

    def get_view_for_detection(self, bbox: dict) -> Optional[dict]:
        """Get the view that contains this detection."""
        center = bbox_center(bbox)
        for view in self._active_views:
            if view.get("view_type") == "full_frame":
                return view
            poly_points = polygon_from_json(view.get("polygon"))
            if poly_points and point_in_polygon(center, poly_points):
                return view
        return None
