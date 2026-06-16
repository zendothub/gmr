"""Feature Requests service - CRUD and email notification."""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.db.models.feature_request import FeatureRequest, FeatureStatus


settings = get_settings()


def _build_email_body(title: str, description: str) -> str:
    """Build a plain-text email body for a new feature request notification."""
    return (
        f"New Feature Request Submitted\n"
        f"=============================\n\n"
        f"Title: {title}\n\n"
        f"Description:\n{description}\n"
    )


def _send_email_sync(title: str, description: str) -> None:
    """Blocking SMTP send — offloaded to a thread executor by the caller."""
    body = _build_email_body(title, description)

    msg = MIMEMultipart()
    msg["From"] = settings.SMTP_FROM
    msg["To"] = settings.NOTIFICATION_EMAIL
    msg["Subject"] = f"Feature Request: {title}"
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if settings.SMTP_USE_TLS:
        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15)
        server.starttls()
    else:
        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15)

    try:
        if settings.SMTP_USER and settings.SMTP_PASS:
            server.login(settings.SMTP_USER, settings.SMTP_PASS)
        server.sendmail(settings.SMTP_FROM, settings.NOTIFICATION_EMAIL, msg.as_string())
        logger.info(f"Feature-request notification email sent to {settings.NOTIFICATION_EMAIL}")
    finally:
        server.quit()


async def _send_notification_email(title: str, description: str) -> None:
    """Offload the synchronous SMTP send to a thread so the event loop stays free."""
    if not all([settings.SMTP_HOST, settings.SMTP_FROM, settings.NOTIFICATION_EMAIL]):
        logger.warning("SMTP not configured, skipping feature-request email notification")
        return

    try:
        await asyncio.to_thread(_send_email_sync, title, description)
    except Exception as exc:
        logger.error(f"Failed to send feature-request email: {exc}")


class FeatureRequestService:

    @staticmethod
    async def create(
        db: AsyncSession,
        title: str,
        description: str,
    ) -> FeatureRequest:
        """Create a new feature request and send email notification."""
        feature_req = FeatureRequest(
            title=title,
            description=description,
            status=FeatureStatus.QUEUED.value,
        )
        db.add(feature_req)
        await db.commit()
        await db.refresh(feature_req)

        # Fire-and-forget email notification (don't block the response)
        await _send_notification_email(title, description)

        return feature_req

    @staticmethod
    async def update(
        db: AsyncSession,
        feature_id: UUID,
        status: Optional[str] = None,
        forecast_message: Optional[str] = None,
    ) -> FeatureRequest:
        """Developer updates status and/or forecast_message on a feature request."""
        result = await db.execute(
            select(FeatureRequest).where(FeatureRequest.id == feature_id)
        )
        feature_req = result.scalar_one_or_none()
        if not feature_req:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature request not found",
            )

        if status is not None:
            feature_req.status = status
        if forecast_message is not None:
            feature_req.forecast_message = forecast_message

        await db.commit()
        await db.refresh(feature_req)
        return feature_req

    @staticmethod
    async def list_requests(
        db: AsyncSession,
        status_filter: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[FeatureRequest], int]:
        """List feature requests, optionally filtered by status, ordered newest-first.

        Returns (items, total_count).
        """
        base = select(FeatureRequest)
        count_base = select(func.count(FeatureRequest.id))

        if status_filter:
            base = base.where(FeatureRequest.status == status_filter)
            count_base = count_base.where(FeatureRequest.status == status_filter)

        # Total count
        total_result = await db.execute(count_base)
        total = total_result.scalar() or 0

        # Paginated items, newest first (descending created_at)
        query = (
            base
            .order_by(desc(FeatureRequest.created_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(query)
        items = result.scalars().all()

        return list(items), total

    @staticmethod
    async def get_live_requests(
        db: AsyncSession,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[FeatureRequest], int]:
        """Shortcut: list requests with LIVE status, newest-first."""
        return await FeatureRequestService.list_requests(
            db, status_filter=FeatureStatus.LIVE.value, page=page, page_size=page_size
        )

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        feature_id: UUID,
    ) -> FeatureRequest:
        """Get a single feature request by ID."""
        result = await db.execute(
            select(FeatureRequest).where(FeatureRequest.id == feature_id)
        )
        feature_req = result.scalar_one_or_none()
        if not feature_req:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature request not found",
            )
        return feature_req