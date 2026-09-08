"""add is_staff column to person_identities

Revision ID: 8d1c5e2f0a9b
Revises: 0005
Create Date: 2026-07-08

Auto-classifies staff/employees via the periodic dedup job using
total session duration and distinct days appeared as signals.
Purchases from staff persons are excluded from analytics.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '8d1c5e2f0a9b'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade():
    # Use IF NOT EXISTS so this migration is idempotent when is_staff was
    # already included in 0001_initial (happens on fresh DB from updated initial schema).
    op.execute(
        "ALTER TABLE person_identities ADD COLUMN IF NOT EXISTS "
        "is_staff BOOLEAN NOT NULL DEFAULT FALSE"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_person_identities_is_staff "
        "ON person_identities (is_staff)"
    )


def downgrade():
    op.drop_index('idx_person_identities_is_staff', table_name='person_identities')
    op.drop_column('person_identities', 'is_staff')
