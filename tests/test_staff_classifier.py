"""Tests for staff presence rule (consec-5 or 70% of any 15-day window)."""

from datetime import date, timedelta

from app.modules.reid.staff_classifier import passes_staff_rule


def _days(*ints):
    base = date(2026, 8, 1)
    return [base + timedelta(days=i) for i in ints]


class TestPassesStaffRule:
    def test_five_consecutive_pass(self):
        assert passes_staff_rule(_days(0, 1, 2, 3, 4)) is True

    def test_four_consecutive_fail(self):
        assert passes_staff_rule(_days(0, 1, 2, 3)) is False

    def test_eleven_of_fifteen_pass(self):
        assert passes_staff_rule(_days(0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12)) is True

    def test_ten_of_fifteen_fail(self):
        assert passes_staff_rule(_days(0, 1, 2, 3, 5, 6, 7, 8, 10, 11)) is False

    def test_holiday_gap_in_window_pass(self):
        assert passes_staff_rule(_days(0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14)) is True

    def test_scattered_nine_days_fail(self):
        assert passes_staff_rule(_days(0, 2, 4, 6, 8, 10, 12, 20, 22)) is False

    def test_empty_fail(self):
        assert passes_staff_rule([]) is False

    def test_six_consecutive_with_hole_before_pass(self):
        assert passes_staff_rule(_days(0, 10, 11, 12, 13, 14)) is True
