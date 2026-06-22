#!/usr/bin/env python3
"""
Fix migration state by manually stamping the migration as applied.
This script should be run when the database already has the changes
but Alembic thinks the migration hasn't been applied yet.
"""
import asyncio
import sys
from sqlalchemy import text
from app.core.db.session import async_engine


async def fix_migration():
    """Manually stamp the migration as applied"""
    
    async with async_engine.begin() as conn:
        # Check current alembic version
        result = await conn.execute(
            text("SELECT version_num FROM alembic_version")
        )
        current_version = result.scalar()
        print(f"Current Alembic version: {current_version}")
        
        # Check if best_crop_path column exists
        result = await conn.execute(
            text(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'track_sessions' AND column_name = 'best_crop_path'"
                ")"
            )
        )
        column_exists = result.scalar()
        print(f"best_crop_path column exists: {column_exists}")
        
        if column_exists and current_version == 'b872230f22bc':
            # Update alembic version to mark the migration as applied
            print("Updating Alembic version to 8b8b559916d9...")
            await conn.execute(
                text("UPDATE alembic_version SET version_num = '8b8b559916d9'")
            )
            print("✓ Migration state fixed!")
            return True
        elif current_version == '8b8b559916d9':
            print("Migration is already marked as applied.")
            return True
        else:
            print(f"Warning: Unexpected state. Current version: {current_version}")
            print("You may need to manually resolve the migration state.")
            return False


async def main():
    try:
        success = await fix_migration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
