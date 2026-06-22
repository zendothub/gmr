#!/usr/bin/env python3
"""
Reset Alembic version to base and rerun all migrations.
This is useful when the database schema is corrupted or out of sync.
"""
import asyncio
import sys
from sqlalchemy import text
from app.core.db.session import async_engine


async def reset_alembic():
    """Reset Alembic to base version and recreate all tables"""
    
    print("🔍 Checking database state...")
    
    async with async_engine.begin() as conn:
        # Check if alembic_version table exists
        result = await conn.execute(
            text(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'alembic_version'"
                ")"
            )
        )
        alembic_exists = result.scalar()
        
        if alembic_exists:
            # Get current version
            result = await conn.execute(
                text("SELECT version_num FROM alembic_version")
            )
            current_version = result.scalar()
            print(f"📌 Current Alembic version: {current_version}")
            
            # Check if any tables exist
            result = await conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                    "AND table_name NOT IN ('alembic_version', 'spatial_ref_sys')"
                )
            )
            table_count = result.scalar()
            print(f"📊 Found {table_count} application tables in database")
            
            if current_version and current_version != 'base' and table_count == 0:
                print("\n⚠️  Database is corrupted:")
                print(f"   - Alembic version: {current_version}")
                print(f"   - Application tables: {table_count}")
                print("\n🔧 Resetting Alembic version to base...")
                
                # Delete the alembic version (reset to base)
                await conn.execute(text("DELETE FROM alembic_version"))
                print("✓ Alembic version reset to base")
                
                print("\n📝 Next steps:")
                print("   1. Exit this script")
                print("   2. Run: alembic upgrade head")
                print("   3. Run: python app/seed.py")
                return True
            elif table_count > 0:
                print("\n✓ Database appears to have tables")
                print("   This script is only for completely corrupted databases")
                print("   Consider using 'alembic downgrade base' then 'alembic upgrade head'")
                return False
        else:
            print("✓ Alembic version table doesn't exist yet")
            print("   Run: alembic upgrade head")
            return True


async def main():
    try:
        success = await reset_alembic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
