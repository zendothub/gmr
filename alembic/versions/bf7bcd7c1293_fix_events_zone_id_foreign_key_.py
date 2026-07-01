"""fix_events_zone_id_foreign_key_constraint

Revision ID: bf7bcd7c1293
Revises: 0004
Create Date: 2026-07-01 15:06:14.081885

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'bf7bcd7c1293'
down_revision: Union[str, None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the existing foreign key constraint
    op.drop_constraint('events_zone_id_fkey', 'events', type_='foreignkey')
    
    # Recreate the foreign key constraint with ON DELETE SET NULL
    op.create_foreign_key(
        'events_zone_id_fkey',
        'events',
        'zones',
        ['zone_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    # Drop the modified foreign key constraint
    op.drop_constraint('events_zone_id_fkey', 'events', type_='foreignkey')
    
    # Recreate the original foreign key constraint without ON DELETE behavior
    op.create_foreign_key(
        'events_zone_id_fkey',
        'events',
        'zones',
        ['zone_id'],
        ['id']
    )
