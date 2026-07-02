import sys
from pathlib import Path

# Ensure project root is in path when running seed.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import uuid
from app.core.db.session import AsyncSessionLocal
from app.core.db.models.user import User, Role, UserStatus
from app.core.db.models.store import Store, StoreStatus
from app.core.db.models.store_lookup import StoreCategory, StoreLevel, StoreZone, StoreTerminal
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

        # --- Seed Categories ---
        categories = ["Pharmacy", "Grocery", "F&B", "Fashion", "Electronics"]
        for cat_name in categories:
            result = await db.execute(select(StoreCategory).where(StoreCategory.name == cat_name))
            cat = result.scalar_one_or_none()
            if not cat:
                cat = StoreCategory(name=cat_name, description=f"{cat_name} stores reference")
                db.add(cat)
                print(f"[StoreCategory Seeded] Created category '{cat_name}'")
        await db.flush()

        # --- Seed Terminals ---
        terminals = ["Terminal 1", "Terminal 2", "Terminal 3"]
        for term_name in terminals:
            result = await db.execute(select(StoreTerminal).where(StoreTerminal.name == term_name))
            term = result.scalar_one_or_none()
            if not term:
                term = StoreTerminal(name=term_name)
                db.add(term)
                print(f"[StoreTerminal Seeded] Created terminal '{term_name}'")
        await db.flush()

        # --- Seed Levels ---
        levels = ["Ground Floor", "Lower Ground Floor", "First Floor", "Second Floor"]
        for lvl_name in levels:
            result = await db.execute(select(StoreLevel).where(StoreLevel.name == lvl_name))
            lvl = result.scalar_one_or_none()
            if not lvl:
                lvl = StoreLevel(name=lvl_name, description=f"{lvl_name} level reference")
                db.add(lvl)
                print(f"[StoreLevel Seeded] Created level '{lvl_name}'")
        await db.flush()

        # --- Seed Zones ---
        zones = [
            ("Zone 1", "Terminal 1"),
            ("Zone 2", "Terminal 2"),
            ("Zone 3", "Terminal 3")
        ]
        for zone_name, term_name in zones:
            result = await db.execute(select(StoreZone).where(StoreZone.name == zone_name))
            zone = result.scalar_one_or_none()
            if not zone:
                zone = StoreZone(name=zone_name, terminal=term_name, description=f"Gate/Zone {zone_name} reference")
                db.add(zone)
                print(f"[StoreZone Seeded] Created zone '{zone_name}' linked to '{term_name}'")
        await db.flush()

        # --- Seed Stores ---
        stores_data = [
            ("Apollo Pharmacy", "Pharmacy", "Terminal 1", "Ground Floor", "Zone 1", "Default pharmacy outlet"),
            ("Daily Grocery", "Grocery", "Terminal 2", "First Floor", "Zone 2", "Daily essentials and grocery"),
            ("Costa Coffee", "F&B", "Terminal 3", "Second Floor", "Zone 3", "Premium coffee and beverages")
        ]
        for name, category, terminal, level, zone_gate, desc in stores_data:
            result = await db.execute(select(Store).where(Store.name == name))
            store = result.scalar_one_or_none()
            if not store:
                store = Store(
                    id=uuid.uuid4(),
                    name=name,
                    category=category,
                    terminal=terminal,
                    level=level,
                    zone_gate=zone_gate,
                    description=desc,
                    status=StoreStatus.ACTIVE
                )
                db.add(store)
                print(f"[Store Seeded] Created store '{name}' ({category})")
        
        await db.commit()
        print("\nDatabase seeded successfully with stores and lookup reference data!")


if __name__ == "__main__":
    asyncio.run(seed())