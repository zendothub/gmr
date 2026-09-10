"""Add leave_requests table (casual/sick leave) and on_leave attendance status.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------------------------
    # 1. attendance_status — add 'on_leave' so daily/aggregate reports can
    #    reflect an approved leave day distinct from present/absent/late.
    # ---------------------------------------------------------------------------
    op.execute("ALTER TYPE attendance_status ADD VALUE IF NOT EXISTS 'on_leave'")

    # ---------------------------------------------------------------------------
    # 2. leave_requests — leave_type ENUM is created inline as part of this
    #    table (not created standalone first: SQLAlchemy's table-create hook
    #    for an unbound Enum column doesn't re-check the DB before creating,
    #    so a separate prior .create() call here causes a duplicate-type error).
    # ---------------------------------------------------------------------------
    op.create_table(
        "leave_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "employee_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "leave_type",
            sa.Enum("CASUAL", "SICK", name="leave_type"),
            nullable=False,
        ),
        sa.Column("date_from", sa.Date, nullable=False),
        sa.Column("date_to", sa.Date, nullable=False),
        sa.Column("days_count", sa.Integer, nullable=False),
        sa.Column("reason", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_leave_requests_employee_id", "leave_requests", ["employee_id"])
    op.create_index("ix_leave_requests_date_from", "leave_requests", ["date_from"])
    op.create_index("ix_leave_requests_date_to", "leave_requests", ["date_to"])


def downgrade() -> None:
    op.drop_index("ix_leave_requests_date_to", table_name="leave_requests")
    op.drop_index("ix_leave_requests_date_from", table_name="leave_requests")
    op.drop_index("ix_leave_requests_employee_id", table_name="leave_requests")
    op.drop_table("leave_requests")

    sa.Enum(name="leave_type").drop(op.get_bind(), checkfirst=True)

    # Note: Postgres doesn't support removing a value from an enum type —
    # 'on_leave' stays in attendance_status even on downgrade.
