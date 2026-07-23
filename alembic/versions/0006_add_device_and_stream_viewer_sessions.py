"""add_device_and_stream_viewer_sessions

Revision ID: 0006
Revises: bf7bcd7c1293
Create Date: 2026-07-23 13:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0006'
down_revision: Union[str, None] = 'bf7bcd7c1293'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── device_sessions ─────────────────────────────────────────────────
    op.create_table(
        'device_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_hash', sa.String(128), nullable=False),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('login_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.Column('last_active_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
    )
    op.create_index('ix_device_sessions_user_id', 'device_sessions', ['user_id'])
    op.create_index('ix_device_sessions_device_hash', 'device_sessions', ['device_hash'])
    op.create_index('ix_device_sessions_is_active', 'device_sessions', ['is_active'])
    op.create_index('ix_device_sessions_last_active', 'device_sessions', ['last_active_at'])
    op.create_foreign_key(
        'fk_device_sessions_user_id',
        'device_sessions', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE',
    )

    # ── stream_viewer_sessions ──────────────────────────────────────────
    op.create_table(
        'stream_viewer_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('camera_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_hash', sa.String(128), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_heartbeat_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
    )
    op.create_index('ix_stream_viewer_camera_id', 'stream_viewer_sessions', ['camera_id'])
    op.create_index('ix_stream_viewer_device_hash', 'stream_viewer_sessions', ['device_hash'])
    op.create_index('ix_stream_viewer_ended_at', 'stream_viewer_sessions', ['ended_at'])
    op.create_foreign_key(
        'fk_stream_viewer_user_id',
        'stream_viewer_sessions', 'users',
        ['user_id'], ['id'],
        ondelete='SET NULL',
    )
    op.create_foreign_key(
        'fk_stream_viewer_camera_id',
        'stream_viewer_sessions', 'cameras',
        ['camera_id'], ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    op.drop_table('stream_viewer_sessions')
    op.drop_table('device_sessions')