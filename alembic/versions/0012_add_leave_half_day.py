"""Add is_half_day to leave_requests; widen days_count to hold 0.5 increments.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-10
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "leave_requests",
        sa.Column("is_half_day", sa.Boolean, nullable=False, server_default="false"),
    )
    op.alter_column(
        "leave_requests",
        "days_count",
        type_=sa.Float,
        existing_type=sa.Integer,
        postgresql_using="days_count::double precision",
    )


def downgrade() -> None:
    op.alter_column(
        "leave_requests",
        "days_count",
        type_=sa.Integer,
        existing_type=sa.Float,
        postgresql_using="round(days_count)::integer",
    )
    op.drop_column("leave_requests", "is_half_day")
