"""Update event severity: drop native PG enum, use VARCHAR, clean up stale data.

Old DB state: native enum type event_severity_enum with labels such as
HIGH, LOW, INFO, WARNING, ALERT (created from Python enum names at initial
schema time).  Some rows have severity = 'INFO' or other now-invalid values.

New state: severity column is VARCHAR(50), only 'High' and 'Low' are stored.
  - purchase events → 'High'
  - everything else → 'Low'

Revision ID: 0012_update_event_severity_enum
Revises: 0011_add_burnin_to_camera
Create Date: 2026-06-17
"""

from alembic import op
import sqlalchemy as sa

revision = '0012_update_event_severity_enum'
down_revision = '8b8b559916d9'
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Convert severity column from native enum → VARCHAR(50).
    # The USING clause casts any existing enum label to text first.
    op.execute("""
        ALTER TABLE events
        ALTER COLUMN severity TYPE VARCHAR(50)
        USING severity::text
    """)

    # Step 2: Drop the now-unused native enum type (IF EXISTS to be safe).
    op.execute("DROP TYPE IF EXISTS event_severity_enum")

    # Step 3: Normalise all existing rows to High / Low.
    #   - purchase events → 'High'
    #   - everything else (INFO, WARNING, ALERT, HIGH, LOW, info, …) → 'Low'
    op.execute("""
        UPDATE events
        SET severity = 'High'
        WHERE event_type = 'purchase'
    """)
    op.execute("""
        UPDATE events
        SET severity = 'Low'
        WHERE severity NOT IN ('High', 'Low')
    """)

    # Step 4: Normalise capitalisation for any rows that happen to have the
    # exact old-style uppercase values already set correctly by the rule engine.
    op.execute("""
        UPDATE events
        SET severity = 'High'
        WHERE severity IN ('HIGH', 'high', 'alert', 'ALERT')
          AND event_type = 'purchase'
    """)
    op.execute("""
        UPDATE events
        SET severity = 'Low'
        WHERE severity IN ('HIGH', 'high', 'LOW', 'low',
                           'INFO', 'info', 'WARNING', 'warning',
                           'ALERT', 'alert')
          AND event_type != 'purchase'
    """)


def downgrade():
    # Recreate the old native enum type and cast back.
    op.execute("""
        CREATE TYPE event_severity_enum AS ENUM ('HIGH', 'LOW', 'INFO', 'WARNING', 'ALERT')
    """)
    op.execute("""
        UPDATE events SET severity = 'HIGH' WHERE severity = 'High'
    """)
    op.execute("""
        UPDATE events SET severity = 'LOW' WHERE severity = 'Low'
    """)
    op.execute("""
        ALTER TABLE events
        ALTER COLUMN severity TYPE event_severity_enum
        USING severity::event_severity_enum
    """)
