"""Add employee attendance tables: shift_slots, employees, attendance_records.

Revision ID: 0009_employee_attendance
Revises: 0008_audit_job_run_columns
Create Date: 2026-09-07
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------------------------
    # 1. attendance_status ENUM
    # ---------------------------------------------------------------------------
    attendance_status_enum = sa.Enum(
        "present", "absent", "late", "half_day",
        name="attendance_status",
    )
    attendance_status_enum.create(op.get_bind(), checkfirst=True)

    # ---------------------------------------------------------------------------
    # 2. shift_slots
    # ---------------------------------------------------------------------------
    op.create_table(
        "shift_slots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("label", sa.String(50), nullable=False, unique=True),
        sa.Column("start_time", sa.Time, nullable=False),
        sa.Column("end_time", sa.Time, nullable=False),
        sa.Column("crosses_midnight", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_shift_slots_label", "shift_slots", ["label"], unique=True)

    # ---------------------------------------------------------------------------
    # 3. employees
    # ---------------------------------------------------------------------------
    op.create_table(
        "employees",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("emp_id", sa.String(100), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "person_identity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("person_identities.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "shift_slot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("shift_slots.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("face_crop_path", sa.String(500), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_employees_emp_id", "employees", ["emp_id"], unique=True)
    op.create_index("ix_employees_person_identity_id", "employees", ["person_identity_id"])

    # ---------------------------------------------------------------------------
    # 4. attendance_records
    # ---------------------------------------------------------------------------
    op.create_table(
        "attendance_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "employee_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attendance_date", sa.Date, nullable=False),
        sa.Column(
            "shift_slot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("shift_slots.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_hours", sa.Float, nullable=True),
        sa.Column(
            "status",
            sa.Enum("present", "absent", "late", "half_day", name="attendance_status"),
            nullable=False,
            server_default="absent",
        ),
        sa.Column(
            "check_in_camera_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cameras.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "check_out_camera_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cameras.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.UniqueConstraint("employee_id", "attendance_date", name="uq_attendance_employee_date"),
    )
    op.create_index("ix_attendance_records_employee_id", "attendance_records", ["employee_id"])
    op.create_index("ix_attendance_records_attendance_date", "attendance_records", ["attendance_date"])


def downgrade() -> None:
    op.drop_index("ix_attendance_records_attendance_date", table_name="attendance_records")
    op.drop_index("ix_attendance_records_employee_id", table_name="attendance_records")
    op.drop_table("attendance_records")

    op.drop_index("ix_employees_person_identity_id", table_name="employees")
    op.drop_index("ix_employees_emp_id", table_name="employees")
    op.drop_table("employees")

    op.drop_index("ix_shift_slots_label", table_name="shift_slots")
    op.drop_table("shift_slots")

    sa.Enum(name="attendance_status").drop(op.get_bind(), checkfirst=True)
