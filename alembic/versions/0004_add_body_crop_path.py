"""add_body_crop_path

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-30 18:35:00.000000

Add body_crop_path column to person_debug for storing MinIO path
to the full body crop image.
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'person_debug',
        sa.Column('body_crop_path', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('person_debug', 'body_crop_path')