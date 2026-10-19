"""daily_analytics_frozen: customer-scale adjusted display fields.

UI display scale (2026-10-15 decision) is CUSTOMERS vs GMR Actual. Raw
columns stay immutable; the three factor columns are rewritable by
danger/refit_person_adjust.py whenever the model is re-fit.
Also renames nothing; existing total-scale fields unchanged.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-15 21:30:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "daily_analytics_frozen",
        sa.Column("adjusted_customers", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "daily_analytics_frozen",
        sa.Column("customer_factor", sa.Float(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("daily_analytics_frozen", "customer_factor")
    op.drop_column("daily_analytics_frozen", "adjusted_customers")
