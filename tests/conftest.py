"""Shared pytest fixtures.

A session-scoped event loop is required for tests that use the module-level
async DB engine (app.core.db.session.AsyncSessionLocal / sync_engine's async
counterpart): asyncpg connections are pinned to the event loop that opened
them, so pytest-asyncio's default per-test loop causes a
"Future attached to a different loop" error on the second DB-touching test.
"""
import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
