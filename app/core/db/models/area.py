"""Area model.

An Area is an independent, named section inside the pharmacy (e.g. "Entry",
"Exit", "Billing", "Medicine Pickup"). Only a name is stored - cameras are
assigned to an area via a dropdown when they are added.
"""

from typing import List

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Area(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "areas"

    # User-provided proper name e.g. "Main Entry", "Billing Counter".
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    cameras: Mapped[List["Camera"]] = relationship("Camera", back_populates="area")
