import asyncio
import uuid
from app.core.db.session import AsyncSessionLocal
from app.core.db.models.store import Store
from app.core.db.models.user import User
from app.utils.encryption import hash_password
from sqlalchemy import select

async def seed():
    async with AsyncSessionLocal() as db:
        # 1. Check if store already exists, create one if not
        result = await db.execute(select(Store))
        store = result.scalars().first()
        if not store:
            store = Store(
                id=uuid.uuid4(),
                name="Default Pharmacy Store",
                address="123 Main St",
                city="Mumbai",
                state="Maharashtra",
                country="India",
                timezone="Asia/Kolkata",
                is_active=True
            )
            db.add(store)
            await db.flush()
            print(f"\n[Store Seeded] Created Store: '{store.name}'")
            print(f"--> Store ID (UUID): {store.id}")
        else:
            print(f"\n[Store Exists] Store: '{store.name}'")
            print(f"--> Store ID (UUID): {store.id}")
            
        # 2. Check if admin user already exists, create if not
        result = await db.execute(select(User).where(User.username == "admin"))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                id=uuid.uuid4(),
                username="admin",
                hashed_password=hash_password("admin123"),
                email="admin@example.com",
                full_name="Administrator",
                is_active=True,
                is_superuser=True,
                store_id=store.id
            )
            db.add(user)
            await db.flush()
            print(f"[User Seeded] Created admin user 'admin' with password 'admin123'")
        else:
            print(f"[User Exists] Admin user 'admin' already exists.")
            
        await db.commit()
        print("\nDatabase seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())
