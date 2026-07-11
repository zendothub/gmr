"""Geometry utilities for point-in-polygon checks and line crossing detection."""

from typing import List, Tuple, Optional
from shapely.geometry import Point, Polygon, LineString
from loguru import logger


def point_in_polygon(point: Tuple[float, float], polygon_points: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon.

    Args:
        point: (x, y) coordinates of the point (e.g., center of bbox)
        polygon_points: List of (x, y) vertices defining the polygon

    Returns:
        True if point is inside the polygon
    """
    if not polygon_points or len(polygon_points) < 3:
        return False
    try:
        p = Point(point)
        poly = Polygon(polygon_points)
        return poly.contains(p)
    except Exception as e:
        logger.error(f"Point-in-polygon check failed: {e}")
        return False


def bbox_center(bbox: dict) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: dict with keys x1, y1, x2, y2

    Returns:
        (center_x, center_y) tuple
    """
    cx = (bbox["x1"] + bbox["x2"]) / 2.0
    cy = (bbox["y1"] + bbox["y2"]) / 2.0
    return (cx, cy)


def bbox_bottom_center(bbox: dict) -> Tuple[float, float]:
    """
    Calculate the bottom-center point of a bounding box (foot position).

    Args:
        bbox: dict with keys x1, y1, x2, y2

    Returns:
        (center_x, bottom_y) tuple
    """
    cx = (bbox["x1"] + bbox["x2"]) / 2.0
    return (cx, bbox["y2"])


def line_crossing_check(
    prev_point: Tuple[float, float],
    curr_point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> Optional[str]:
    """
    Check if movement from prev_point to curr_point crosses a line.

    Args:
        prev_point: Previous position (x, y)
        curr_point: Current position (x, y)
        line_start: Start of the line (x, y)
        line_end: End of the line (x, y)

    Returns:
        'forward' or 'backward' if crossing detected, None otherwise.
        Direction is relative to the line's normal vector.
    """
    try:
        movement = LineString([prev_point, curr_point])
        boundary = LineString([line_start, line_end])

        if not movement.intersects(boundary):
            return None

        # Determine direction using cross product
        line_dx = line_end[0] - line_start[0]
        line_dy = line_end[1] - line_start[1]
        move_dx = curr_point[0] - prev_point[0]
        move_dy = curr_point[1] - prev_point[1]

        cross = line_dx * move_dy - line_dy * move_dx

        if cross > 0:
            return "forward"
        elif cross < 0:
            return "backward"
        return None

    except Exception as e:
        logger.error(f"Line crossing check failed: {e}")
        return None


def polygon_from_json(polygon_json: dict) -> Optional[List[Tuple[float, float]]]:
    """
    Convert a polygon JSON (from DB) to list of (x, y) tuples.

    Expected format: {"points": [[x1,y1], [x2,y2], ...]}
    """
    if not polygon_json:
        return None
    points = polygon_json.get("points", [])
    if not points or len(points) < 3:
        return None
    return [(p[0], p[1]) for p in points]


def line_from_json(line_json: dict) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert a line JSON (from DB) to start/end tuples.

    Expected format: {"start": [x1, y1], "end": [x2, y2]}
    """
    if not line_json:
        return None
    start = line_json.get("start")
    end = line_json.get("end")
    if not start or not end:
        return None
    return ((start[0], start[1]), (end[0], end[1]))


def bbox_height(bbox: dict) -> float:
    """Calculate the height of a bounding box."""
    return abs(bbox["y2"] - bbox["y1"])


def bbox_area(bbox: dict) -> float:
    """Calculate the area of a bounding box."""
    return abs(bbox["x2"] - bbox["x1"]) * abs(bbox["y2"] - bbox["y1"])


def bbox_intersection_area(a: dict, b: dict) -> float:
    """Area of axis-aligned intersection between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def bbox_iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    inter = bbox_intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def face_area_in_body_frac(face_bbox: dict, body_bbox: dict) -> float:
    """Fraction of face box area that lies inside the body box (0–1)."""
    fa = bbox_area(face_bbox)
    if fa <= 0.0:
        return 0.0
    return bbox_intersection_area(face_bbox, body_bbox) / fa
