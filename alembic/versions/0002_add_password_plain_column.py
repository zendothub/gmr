"""Add password_plain column to users table.

Revision ID: 0002_add_password_plain
Revises: 0001_initial
Create Date: 2026-06-23
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "0002_add_password_plain"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("password_plain", sa.String(128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "password_plain")
