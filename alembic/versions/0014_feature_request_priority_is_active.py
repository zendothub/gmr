"""Add priority and is_active to feature_requests table.

Revision ID: 0014_feature_request_priority_is_active
Revises: 0013_drop_is_superuser_column
Create Date: 2026-06-19
"""
from alembic import op
import sqlalchemy as sa

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the priority enum type
    priority_enum = sa.Enum("low", "high", name="feature_priority_enum")
    priority_enum.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "feature_requests",
        sa.Column(
            "priority",
            sa.Enum("low", "high", name="feature_priority_enum"),
            nullable=False,
            server_default="low",
        ),
    )
    op.add_column(
        "feature_requests",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("feature_requests", "is_active")
    op.drop_column("feature_requests", "priority")
    op.execute("DROP TYPE IF EXISTS feature_priority_enum")
