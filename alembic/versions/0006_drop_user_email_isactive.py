"""drop users.email and users.is_active

Revision ID: 0006_user_slim
Revises: 0005_drop_views
Create Date: 2026-06-13

The user model is simplified to username + password + full_name + is_superuser.
Email and the is_active soft-disable flag are removed.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0006_user_slim"
down_revision: Union[str, None] = "0005_drop_views"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.drop_column("users", "email")
    op.drop_column("users", "is_active")


def downgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column("users", sa.Column("email", sa.String(length=255), nullable=True))
    op.create_index("ix_users_email", "users", ["email"], unique=True)
