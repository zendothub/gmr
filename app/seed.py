import sys
from pathlib import Path

# Ensure project root is in path when running seed.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import uuid
from app.core.db.session import AsyncSessionLocal
from app.core.db.models.user import User, Role, UserStatus
from app.utils.encryption import hash_password
from sqlalchemy import select
from sqlalchemy.orm import selectinload


async def seed():
    async with AsyncSessionLocal() as db:
        # Ensure roles exist: SUPER_ADMIN, ADMIN, VIEWER
        roles_data = [
            ("SUPER_ADMIN", "Full access to everything"),
            ("ADMIN", "Administrative access"),
            ("VIEWER", "Read-only access"),
        ]

        created_roles = {}
        for role_name, description in roles_data:
            result = await db.execute(select(Role).where(Role.name == role_name))
            role = result.scalar_one_or_none()
            if not role:
                role = Role(name=role_name, description=description)
                db.add(role)
                await db.flush()
                print(f"[Role Seeded] Created '{role_name}' role")
            else:
                print(f"[Role Exists] '{role_name}' role already exists")
            created_roles[role_name] = role

        await db.flush()

        # Seed a default SUPER_ADMIN user
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.email == "admin@gmr.com")
        )
        admin_user = result.scalar_one_or_none()
        if not admin_user:
            admin_user = User(
                id=uuid.uuid4(),
                name="Administrator",
                email="admin@gmr.com",
                hashed_password=hash_password("admin"),
                status=UserStatus.ACTIVE,
            )
            admin_user.roles.append(created_roles["SUPER_ADMIN"])
            db.add(admin_user)
            print("[User Seeded] Created SUPER_ADMIN user 'admin@gmr.com' with password 'admin'")
        else:
            if created_roles["SUPER_ADMIN"] not in admin_user.roles:
                admin_user.roles.append(created_roles["SUPER_ADMIN"])
                print("[User Updated] Assigned 'SUPER_ADMIN' role to existing admin user")
            else:
                print("[User Exists] SUPER_ADMIN user 'admin@gmr.com' already exists")

        await db.commit()
        print("\nDatabase seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed())