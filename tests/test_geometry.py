"""Tests for geometry utilities (point-in-polygon, line crossing, bbox helpers)."""

from app.utils.geometry import (
    point_in_polygon,
    bbox_center,
    bbox_bottom_center,
    bbox_height,
    bbox_area,
    polygon_from_json,
    line_from_json,
    line_crossing_check,
)


SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


class TestPointInPolygon:
    def test_inside(self):
        assert point_in_polygon((50, 50), SQUARE) is True

    def test_outside(self):
        assert point_in_polygon((150, 50), SQUARE) is False

    def test_far_outside(self):
        assert point_in_polygon((-10, -10), SQUARE) is False


class TestBBoxHelpers:
    BBOX = {"x1": 10, "y1": 20, "x2": 50, "y2": 100}

    def test_center(self):
        assert bbox_center(self.BBOX) == (30.0, 60.0)

    def test_bottom_center(self):
        assert bbox_bottom_center(self.BBOX) == (30.0, 100.0)

    def test_height(self):
        assert bbox_height(self.BBOX) == 80

    def test_area(self):
        assert bbox_area(self.BBOX) == 40 * 80


class TestJsonParsers:
    def test_polygon_from_json(self):
        poly = polygon_from_json({"points": [[0, 0], [10, 0], [10, 10], [0, 10]]})
        assert poly is not None
        assert len(poly) == 4

    def test_polygon_from_none(self):
        assert polygon_from_json(None) is None

    def test_line_from_json(self):
        line = line_from_json({"start": [0, 50], "end": [100, 50]})
        assert line is not None


class TestLineCrossing:
    def test_crossing_detected(self):
        # Line y=50 horizontal; movement from above to below crosses it
        result = line_crossing_check((50, 30), (50, 70), (0, 50), (100, 50))
        assert result is not None

    def test_no_crossing(self):
        result = line_crossing_check((50, 10), (50, 20), (0, 50), (100, 50))
        assert result is None or result is False or result == "none" or not result