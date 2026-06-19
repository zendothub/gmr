"""Users service - admin user and role management."""

from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.db.models.user import User, Role
from app.utils.encryption import hash_password
from app.modules.users.schemas import UserCreate, UserUpdate, RoleCreate


class UserService:

    @staticmethod
    async def create_user(db: AsyncSession, data: UserCreate) -> User:
        """Create a new user (admin operation)."""
        result = await db.execute(select(User).where(User.username == data.username))
        if result.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already exists")

        user = User(
            username=data.username,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
        )



        # Attach roles if provided
        if data.role_names:
            result = await db.execute(select(Role).where(Role.name.in_(data.role_names)))
            user.roles = list(result.scalars().all())

        db.add(user)
        await db.flush()
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user.id)
        )
        user = result.scalar_one()
        logger.info(f"User created by admin: {user.username} (id={user.id})")
        return user

    @staticmethod
    async def get_users(db: AsyncSession) -> List[User]:
        """List all users."""
        result = await db.execute(
            select(User).options(selectinload(User.roles)).order_by(User.created_at.desc())
        )
        return list(result.scalars().all())

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
    async def update_user(db: AsyncSession, user_id: UUID, data: UserUpdate) -> User:
        """Update a user."""
        user = await UserService.get_user(db, user_id)

        update_data = data.model_dump(exclude_unset=True, exclude={"password"})
        for field, value in update_data.items():
            setattr(user, field, value)

        if data.password:
            user.hashed_password = hash_password(data.password)

        await db.flush()
        await db.refresh(user)
        logger.info(f"User updated: {user.username} (id={user_id})")
        return user

    @staticmethod
    async def delete_user(db: AsyncSession, user_id: UUID) -> dict:
        """Delete a user."""
        user = await UserService.get_user(db, user_id)
        await db.delete(user)
        logger.info(f"User deleted: {user.username} (id={user_id})")
        return {"message": f"User '{user.username}' deleted successfully"}

    # --- Roles ---

    @staticmethod
    async def create_role(db: AsyncSession, data: RoleCreate) -> Role:
        """Create a new role."""
        result = await db.execute(select(Role).where(Role.name == data.name))
        if result.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Role already exists")

        role = Role(name=data.name, description=data.description)
        db.add(role)
        await db.flush()
        await db.refresh(role)
        logger.info(f"Role created: {role.name}")
        return role

    @staticmethod
    async def get_roles(db: AsyncSession) -> List[Role]:
        """List all roles."""
        result = await db.execute(select(Role).order_by(Role.name))
        return list(result.scalars().all())

    @staticmethod
    async def assign_role(db: AsyncSession, user_id: UUID, role_name: str) -> User:
        """Assign a role to a user."""
        user = await UserService.get_user(db, user_id)
        result = await db.execute(select(Role).where(Role.name == role_name))
        role = result.scalar_one_or_none()
        if not role:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Role '{role_name}' not found")

        if role not in user.roles:
            user.roles.append(role)
            await db.flush()
        logger.info(f"Role '{role_name}' assigned to user {user.username}")
        return user