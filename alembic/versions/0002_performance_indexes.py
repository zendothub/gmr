"""Performance indexes for dashboard queries.

Revision ID: 0002_perf_indexes
Revises: 0001_initial
Create Date: 2026-06-12
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_perf_indexes"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Dashboard event listing / analytics filters
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_camera_occurred "
        "ON events (camera_id, occurred_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_type_occurred "
        "ON events (event_type, occurred_at DESC)"
    )
    # Billing analytics time-range scans
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_billing_camera_entered "
        "ON billing_interactions (camera_id, entered_at DESC)"
    )
    # Track session range scans (dwell analytics, person journey)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_track_sessions_camera_started "
        "ON track_sessions (camera_id, started_at DESC)"
    )
    # ReID recency filter on embeddings
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_person_embeddings_captured "
        "ON person_embeddings (captured_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_events_camera_occurred")
    op.execute("DROP INDEX IF EXISTS idx_events_type_occurred")
    op.execute("DROP INDEX IF EXISTS idx_billing_camera_entered")
    op.execute("DROP INDEX IF EXISTS idx_track_sessions_camera_started")
    op.execute("DROP INDEX IF EXISTS idx_person_embeddings_captured")