"""add burnin_enabled column to cameras

Revision ID: 0011_burnin
Revises: 0010_feature_requests
Create Date: 2026-06-16

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0011_burnin"
down_revision: Union[str, None] = "0010_feature_requests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column("burnin_enabled", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("cameras", "burnin_enabled")