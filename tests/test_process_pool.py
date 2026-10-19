"""Tests for the spawn job process pool (app/modules/jobs/process_pool.py)."""

import asyncio

import pytest

from app.modules.jobs import process_pool as pp


def _double(x: int) -> int:
    return x * 2


def _boom() -> None:
    raise ValueError("task failed")


class _BrokenStubExecutor:
    """submit() raises like a pool whose children died."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.shutdown_calls = 0

    def submit(self, fn, *args, **kwargs):
        raise self._exc

    def shutdown(self, wait: bool = True, **kwargs):
        self.shutdown_calls += 1


def test_run_in_subprocess_returns_value():
    assert asyncio.run(pp.run_in_subprocess(_double, 21)) == 42


def test_task_exception_propagates():
    with pytest.raises(ValueError, match="task failed"):
        asyncio.run(pp.run_in_subprocess(_boom))


def test_broken_pool_retries_then_aborts():
    """BrokenProcessPool → rebuild+retry; second break aborts (no thread run)."""
    stub = _BrokenStubExecutor(pp.BrokenProcessPool("children died"))
    orig = pp._get_pool
    gets = {"n": 0}

    def fake_get():
        gets["n"] += 1
        return stub

    pp._get_pool = fake_get
    try:
        with pytest.raises(RuntimeError, match="died twice"):
            asyncio.run(pp.run_in_subprocess(_double, 1))
    finally:
        pp._get_pool = orig
    assert gets["n"] == 2  # one attempt per fresh pool


def test_oserror_falls_back_to_thread():
    """Pool cannot start at all → body still completes on a worker thread."""
    stub = _BrokenStubExecutor(OSError("cannot spawn"))
    orig = pp._get_pool
    orig_pool = pp._pool
    pp._get_pool = lambda: stub
    pp._pool = stub  # _reset_pool() shuts down the module-level pool
    try:
        assert asyncio.run(pp.run_in_subprocess(_double, 7)) == 14
    finally:
        pp._get_pool = orig
        pp._pool = orig_pool
    assert stub.shutdown_calls >= 1


def test_pool_recycles_after_max_tasks(monkeypatch):
    monkeypatch.setattr("app.config.get_settings", lambda: type(
        "S", (), {"JOB_POOL_MAX_TASKS": 2, "JOB_POOL_WORKERS": 1}
    )())
    orig_reset = pp._reset_pool
    resets = []
    pp._reset_pool = lambda reason="": resets.append(reason)
    try:
        pp._tasks_completed = 0
        pp._note_task_done()
        assert not resets
        pp._note_task_done()
        assert len(resets) == 1 and "recycled" in resets[0]
    finally:
        pp._reset_pool = orig_reset
        pp._tasks_completed = 0
