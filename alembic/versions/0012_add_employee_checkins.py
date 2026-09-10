"""Add employee_checkins table for check-in event feed.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-10
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "employee_checkins",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("employee_id", UUID(as_uuid=True), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_name", sa.String(255), nullable=False),
        sa.Column("emp_code", sa.String(100), nullable=False),
        sa.Column("checked_in_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("check_in_date", sa.Date(), nullable=False, index=True),
        sa.Column("shift_slot_id", UUID(as_uuid=True), sa.ForeignKey("shift_slots.id", ondelete="SET NULL"), nullable=True),
        sa.Column("shift_label", sa.String(50), nullable=True),
        sa.Column("camera_id", UUID(as_uuid=True), sa.ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True),
        sa.Column("face_crop_path", sa.String(500), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="on_time"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Unique constraint: one check-in per employee per date
    op.create_unique_constraint(
        "uq_employee_checkins_employee_date",
        "employee_checkins",
        ["employee_id", "check_in_date"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_employee_checkins_employee_date", "employee_checkins", type_="unique")
    op.drop_table("employee_checkins")
