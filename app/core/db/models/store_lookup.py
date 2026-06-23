"""Store lookup tables - Category, Level, StoreZone.

These are user-managed reference lists that populate the dropdowns when
creating/editing a Store.  StoreZone here refers to a physical airport
zone/gate label (e.g. "Gate B4") and is COMPLETELY SEPARATE from the
camera-related Zone model in camera.py.
"""

from typing import Optional

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class StoreCategory(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Retail store categories (e.g. Pharmacy, F&B, Retail, Luxury)."""
    __tablename__ = "store_categories"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class StoreLevel(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Airport levels (e.g. Level 1, Level 2)."""
    __tablename__ = "store_levels"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class StoreZone(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Physical airport zones/gates used as store location labels.

    NOT related to camera detection zones (see camera.py → Zone).
    Examples: "Gate B4", "Gate A12", "Terminal 2 Departure".
    """
    __tablename__ = "store_zones"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    terminal: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
