"""Users service - user management."""

from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.db.models.user import User, Role, UserStatus
from app.utils.encryption import hash_password, verify_password
import secrets
import string

from app.modules.users.schemas import UserCreate, UserUpdate, UpdateProfile, UpdatePassword, UserStatusUpdate, UserAdd


def _first_role_name(user: User) -> str:
    """Return the first role name as a plain string, or empty string if none."""
    if user.roles:
        return user.roles[0].name
    return ""


class UserService:

    @staticmethod
    async def _get_or_create_role(db: AsyncSession, role_name: str) -> Role:
        """Get an existing role or create it if it doesn't exist."""
        result = await db.execute(select(Role).where(Role.name == role_name))
        role = result.scalar_one_or_none()
        if not role:
            role = Role(name=role_name, description=f"{role_name} role")
            db.add(role)
            await db.flush()
        return role

    @staticmethod
    async def create_user(db: AsyncSession, data: UserCreate) -> User:
        """Create a new user. Status is always set to 'active' automatically."""
        # Check if email already exists
        result = await db.execute(select(User).where(User.email == data.email))
        if result.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Email already exists")

        role = await UserService._get_or_create_role(db, data.role)

        user = User(
            name=data.name,
            email=data.email,
            hashed_password=hash_password(data.password),
            password_plain=data.password,   # stored for admin visibility
            status=UserStatus.ACTIVE,       # always active on creation
        )
        user.roles.append(role)

        db.add(user)
        await db.flush()

        # Reload with roles eagerly loaded
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user.id)
        )
        user = result.scalar_one()
        logger.info(f"User invited: {user.name} <{user.email}> role={data.role} (id={user.id})")
        return user

    @staticmethod
    async def create_user_auto_password(db: AsyncSession, data: UserAdd) -> User:
        """Add a user with an auto-generated password (POST /api/users).

        A random 10-character password is generated and stored in plain text
        so the admin can view and share it.
        """
        result = await db.execute(select(User).where(User.email == data.email))
        if result.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Email already exists")

        # Generate a readable random password: 8 alphanum chars
        alphabet = string.ascii_letters + string.digits
        auto_password = "".join(secrets.choice(alphabet) for _ in range(10))

        role = await UserService._get_or_create_role(db, data.role)

        user = User(
            name=data.name,
            email=data.email,
            hashed_password=hash_password(auto_password),
            password_plain=auto_password,   # stored so admin can see it
            status=UserStatus.ACTIVE,
        )
        user.roles.append(role)
        db.add(user)
        await db.flush()

        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user.id)
        )
        user = result.scalar_one()
        logger.info(f"User added (auto-pw): {user.name} <{user.email}> role={data.role} (id={user.id})")
        return user

    @staticmethod
    async def get_users(
        db: AsyncSession,
        role_filter: Optional[str] = None,
    ) -> List[User]:
        """List all users, optionally filtered by role (ADMIN or VIEWER).

        Excludes SUPER_ADMIN accounts from the management list.
        """
        query = (
            select(User)
            .options(selectinload(User.roles))
            .order_by(User.created_at.desc())
        )
        result = await db.execute(query)
        users = list(result.scalars().all())

        # Filter out SUPER_ADMIN from the management list
        users = [u for u in users if _first_role_name(u) != "SUPER_ADMIN"]

        if role_filter:
            users = [u for u in users if _first_role_name(u) == role_filter]

        return users

    @staticmethod
    async def get_user(db: AsyncSession, user_id: UUID) -> User:
        """Get a single user by ID."""
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
        return user

    @staticmethod
    async def update_user_status(
        db: AsyncSession,
        user_id: UUID,
        data: UserStatusUpdate,
    ) -> User:
        """Activate or deactivate a user (toggle status)."""
        user = await UserService.get_user(db, user_id)
        user.status = UserStatus(data.status)
        await db.flush()
        await db.refresh(user)
        logger.info(f"User status → '{data.status}': {user.email} (id={user_id})")
        return user

    @staticmethod
    async def update_user(db: AsyncSession, user_id: UUID, data: UserUpdate) -> User:
        """Update a user's details (admin operation)."""
        user = await UserService.get_user(db, user_id)

        role_name = data.role
        update_fields = data.model_dump(exclude_unset=True, exclude={"password", "role"})

        for field, value in update_fields.items():
            if field == "status":
                setattr(user, field, UserStatus(value))
            else:
                setattr(user, field, value)

        if data.password:
            user.hashed_password = hash_password(data.password)

        if role_name:
            role = await UserService._get_or_create_role(db, role_name)
            user.roles.clear()
            user.roles.append(role)

        await db.flush()
        await db.refresh(user)
        logger.info(f"User updated: {user.email} (id={user_id})")
        return user

    @staticmethod
    async def update_profile(db: AsyncSession, current_user: User, data: UpdateProfile) -> User:
        """Allow the authenticated user to update their own profile details (name)."""
        update_fields = data.model_dump(exclude_unset=True)
        for field, value in update_fields.items():
            setattr(current_user, field, value)

        await db.flush()
        await db.refresh(current_user)
        logger.info(f"Profile updated for user: {current_user.email} (id={current_user.id})")
        return current_user

    @staticmethod
    async def update_password(db: AsyncSession, current_user: User, data: UpdatePassword) -> dict:
        """Allow the authenticated user to change their password."""
        if not verify_password(data.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )
        if data.new_password != data.confirm_new_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password and confirmation do not match",
            )
        current_user.hashed_password = hash_password(data.new_password)
        await db.flush()
        logger.info(f"Password changed for user: {current_user.email} (id={current_user.id})")
        return {"message": "Password updated successfully"}

    @staticmethod
    async def delete_user(db: AsyncSession, user_id: UUID) -> dict:
        """Delete a user."""
        user = await UserService.get_user(db, user_id)
        await db.delete(user)
        logger.info(f"User deleted: {user.email} (id={user_id})")
        return {"message": f"User '{user.email}' deleted successfully"}
