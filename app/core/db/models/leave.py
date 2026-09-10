"""LeaveRequest model — casual/sick leave applied on an employee's behalf.

No approval workflow: admin applies on the employee's behalf (employees have
no login) and it reflects in attendance immediately. Casual + sick share one
combined 12-day/calendar-year quota — enforced in leaves/service.py.
"""

import enum
import uuid
from datetime import date
from typing import Optional

from sqlalchemy import String, Date, Integer, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LeaveType(str, enum.Enum):
    CASUAL = "CASUAL"
    SICK = "SICK"


class LeaveRequest(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "leave_requests"

    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    leave_type: Mapped[LeaveType] = mapped_column(
        SAEnum(
            LeaveType,
            name="leave_type",
            create_constraint=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
    )

    date_from: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    date_to: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Working days in [date_from, date_to] excluding the employee's own
    # weekly-off days, snapshotted at apply/edit time so a later change to
    # the employee's weekend config never retroactively alters quota already
    # consumed by this request.
    days_count: Mapped[int] = mapped_column(Integer, nullable=False)

    reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    employee: Mapped["Employee"] = relationship("Employee")
