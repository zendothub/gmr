"""Add detection-event zone types to zone_type_enum.

Adds 6 new values to the existing PostgreSQL ``zone_type_enum`` that drive
the polygon-editor detection events (eye-icon feature):

  footfall      → Footfall tracking
  dwell_time    → Dwell Time analysis
  queue_length  → Queue Length monitoring
  entry_exit    → Entry / Exit counting (combined line)
  heatmap       → Heatmap heat density
  purchase_intent → Purchase Intent signals at shelf / counter

The existing legacy values (entry_line, exit_line, billing_zone, etc.) are
preserved unchanged — this is an additive-only migration.

Revision ID: 0003_detection_zone_types
Revises: 0002_add_store_id
Create Date: 2026-06-23
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_detection_zone_types"
down_revision = "0002_add_store_id"
branch_labels = None
depends_on = None

# New enum values to add (V2 polygon-editor detection event types)
_NEW_ZONE_TYPE_VALUES = [
    "footfall",
    "dwell_time",
    "queue_length",
    "entry_exit",
    "heatmap",
    "purchase_intent",
]


def upgrade() -> None:
    """Add detection-event values to zone_type_enum.

    PostgreSQL does NOT allow ``ALTER TYPE … ADD VALUE`` inside an open
    transaction (for PG < 12).  We explicitly COMMIT the current transaction,
    run the DDL statements, then BEGIN a fresh transaction so that the rest of
    the migration (if any) can continue normally.
    """
    connection = op.get_bind()

    # Close the implicit transaction opened by Alembic so that
    # ALTER TYPE ADD VALUE succeeds on PostgreSQL < 12.
    connection.execute(sa.text("COMMIT"))

    for val in _NEW_ZONE_TYPE_VALUES:
        connection.execute(
            sa.text(f"ALTER TYPE zone_type_enum ADD VALUE IF NOT EXISTS '{val}'")
        )

    # Re-open a transaction so Alembic's bookkeeping (alembic_version update) works.
    connection.execute(sa.text("BEGIN"))


def downgrade() -> None:
    """PostgreSQL does NOT support removing individual enum values.

    To downgrade, the enum must be recreated from scratch:
      1. Add a temporary TEXT column on zones
      2. Copy values
      3. Drop old column
      4. DROP TYPE zone_type_enum
      5. CREATE TYPE zone_type_enum with only the original values
      6. Re-add the column + copy values back

    This is intentionally left as a no-op because:
    - Removing enum values in production is destructive.
    - Any rows that used the new values would become invalid.
    - If you need to roll back, restore from a DB snapshot.
    """
    pass
