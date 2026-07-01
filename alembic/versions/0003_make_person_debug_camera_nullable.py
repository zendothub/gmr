"""make_person_debug_camera_nullable

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-30 18:26:00.000000

Change camera_id and store_id in person_debug from NOT NULL + ON DELETE CASCADE
to nullable + ON DELETE SET NULL.  This prevents SQLAlchemy's ORM-level backref
from blocking camera/store deletion with NotNullViolationError.
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── camera_id: drop CASCADE FK, allow null, re-create with SET NULL ──
    op.drop_constraint(
        'person_debug_camera_id_fkey',
        'person_debug',
        type_='foreignkey',
    )
    op.alter_column(
        'person_debug',
        'camera_id',
        existing_type=op.f('UUID'),
        nullable=True,
    )
    op.create_foreign_key(
        'person_debug_camera_id_fkey',
        'person_debug',
        'cameras',
        ['camera_id'],
        ['id'],
        ondelete='SET NULL',
    )

    # ── store_id: drop CASCADE FK, re-create with SET NULL ──
    op.drop_constraint(
        'person_debug_store_id_fkey',
        'person_debug',
        type_='foreignkey',
    )
    op.create_foreign_key(
        'person_debug_store_id_fkey',
        'person_debug',
        'stores',
        ['store_id'],
        ['id'],
        ondelete='SET NULL',
    )


def downgrade() -> None:
    # ── camera_id: revert to NOT NULL + ON DELETE CASCADE ──
    op.drop_constraint(
        'person_debug_camera_id_fkey',
        'person_debug',
        type_='foreignkey',
    )
    op.alter_column(
        'person_debug',
        'camera_id',
        existing_type=op.f('UUID'),
        nullable=False,
    )
    op.create_foreign_key(
        'person_debug_camera_id_fkey',
        'person_debug',
        'cameras',
        ['camera_id'],
        ['id'],
        ondelete='CASCADE',
    )

    # ── store_id: revert to ON DELETE CASCADE ──
    op.drop_constraint(
        'person_debug_store_id_fkey',
        'person_debug',
        type_='foreignkey',
    )
    op.create_foreign_key(
        'person_debug_store_id_fkey',
        'person_debug',
        'stores',
        ['store_id'],
        ['id'],
        ondelete='CASCADE',
    )