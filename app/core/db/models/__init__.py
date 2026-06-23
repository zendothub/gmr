"""Database models package - imports all models for easy access."""

from app.core.db.models.user import User, Role, user_roles
from app.core.db.models.area import Area


from app.core.db.models.camera import (
    Camera,
    Zone,
    CameraStatus,
    ZoneType,
    ZoneShape,
)

from app.core.db.models.rule import Rule, RuleType
from app.core.db.models.tracking import TrackSession, TrackObservation
from app.core.db.models.person import PersonIdentity, PersonEmbedding
from app.core.db.models.event import Event, EventSeverity
from app.core.db.models.billing import BillingInteraction
from app.core.db.models.analytics import DailyAnalyticsSummary
from app.core.db.models.feature_request import FeatureRequest, FeatureStatus
from app.core.db.models.storage import StorageObject, StorageType
from app.core.db.models.store import Store, StoreStatus
from app.core.db.models.store_lookup import StoreCategory, StoreLevel, StoreZone, StoreTerminal

__all__ = [
    "User",
    "Role",
    "user_roles",
    "Area",

    "Camera",
    "Zone",
    "CameraStatus",
    "ZoneType",

    "ZoneShape",
    "Rule",
    "RuleType",
    "TrackSession",
    "TrackObservation",
    "PersonIdentity",
    "PersonEmbedding",
    "Event",
    "EventSeverity",
    "BillingInteraction",
    "DailyAnalyticsSummary",
    "StorageObject",
    "StorageType",
    "FeatureRequest",
    "FeatureStatus",
    "Store",
    "StoreStatus",
    "StoreCategory",
    "StoreLevel",
    "StoreZone",
    "StoreTerminal",
]
