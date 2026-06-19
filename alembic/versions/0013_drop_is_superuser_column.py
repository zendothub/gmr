"""Drop is_superuser column from users table.

Revision ID: 0013
Revises: 40a9c0e6c353
Create Date: 2026-06-19
"""

from alembic import op
import sqlalchemy as sa

revision = '0013'
down_revision = '40a9c0e6c353'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column('users', 'is_superuser')


def downgrade():
    op.add_column(
        'users',
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default=sa.text('false')),
    )