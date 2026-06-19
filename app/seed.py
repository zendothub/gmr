import asyncio
import uuid
from app.core.db.session import AsyncSessionLocal
from app.core.db.models.user import User
from app.utils.encryption import hash_password
from sqlalchemy import select

async def seed():
    async with AsyncSessionLocal() as db:
        # Check if admin user already exists, create if not
        result = await db.execute(select(User).where(User.username == "admin"))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                id=uuid.uuid4(),
                username="admin",
                hashed_password=hash_password("admin"),
                full_name="Administrator",
                is_superuser=True,
            )
            db.add(user)
            await db.flush()
            print(f"[User Seeded] Created admin user 'admin' with password 'admin'")
        else:
            print(f"[User Exists] Admin user 'admin' already exists.")

        await db.commit()
        print("\nDatabase seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())
