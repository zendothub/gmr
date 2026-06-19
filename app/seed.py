import asyncio
import uuid
from app.core.db.session import AsyncSessionLocal
from app.core.db.models.user import User, Role
from app.utils.encryption import hash_password
from sqlalchemy import select
from sqlalchemy.orm import selectinload


async def seed():
    async with AsyncSessionLocal() as db:
        # Ensure roles exist
        admin_role = None
        user_role = None

        result = await db.execute(select(Role).where(Role.name == "admin"))
        admin_role = result.scalar_one_or_none()
        if not admin_role:
            admin_role = Role(name="admin", description="Full access to all APIs and feature requests")
            db.add(admin_role)
            await db.flush()
            print("[Role Seeded] Created 'admin' role")

        result = await db.execute(select(Role).where(Role.name == "user"))
        user_role = result.scalar_one_or_none()
        if not user_role:
            user_role = Role(name="user", description="Read-only access to GET APIs")
            db.add(user_role)
            await db.flush()
            print("[Role Seeded] Created 'user' role")

        await db.flush()

        # Create admin user
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.username == "admin")
        )
        admin_user = result.scalar_one_or_none()
        if not admin_user:
            admin_user = User(
                id=uuid.uuid4(),
                username="admin",
                hashed_password=hash_password("admin"),
                full_name="Administrator",
            )
            admin_user.roles.append(admin_role)
            db.add(admin_user)
            print("[User Seeded] Created admin user 'admin' with password 'admin'")
        else:
            # Ensure admin role is assigned
            if admin_role not in admin_user.roles:
                admin_user.roles.append(admin_role)
                print("[User Updated] Assigned 'admin' role to existing admin user")
            else:
                print("[User Exists] Admin user 'admin' already exists with admin role")

        # Create rocky user (admin)
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.username == "rocky")
        )
        rocky_user = result.scalar_one_or_none()
        if not rocky_user:
            rocky_user = User(
                id=uuid.uuid4(),
                username="rocky",
                hashed_password=hash_password("rocky"),
                full_name="Rocky",
            )
            rocky_user.roles.append(admin_role)
            db.add(rocky_user)
            print("[User Seeded] Created admin user 'rocky' with password 'rocky'")
        else:
            if admin_role not in rocky_user.roles:
                rocky_user.roles.append(admin_role)
                print("[User Updated] Assigned 'admin' role to existing user 'rocky'")
            else:
                print("[User Exists] User 'rocky' already exists with admin role")

        await db.commit()
        print("\nDatabase seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed())