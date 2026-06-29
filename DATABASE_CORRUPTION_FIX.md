# Database Corruption Fix Guide

## 🔴 Critical Issue Detected

Your database is in a **severely corrupted state**:

- ✅ Alembic version: `8b8b559916d9` (appears "applied")
- ❌ Tables: `track_sessions`, `roles`, and others **DO NOT EXIST**
- ❌ Result: All migrations and seed scripts fail

**Root Cause:** The Alembic version tracker is completely out of sync with the actual database schema. The database thinks migrations are applied when the tables don't even exist.

---

## 🚨 Choose Your Fix Strategy

### Strategy 1: Complete Database Reset (RECOMMENDED - Fastest)

**Use when:** Starting fresh is acceptable (development/staging environments)

```bash
cd /home/ubuntu/gmr
source venv/bin/activate

# Make script executable
chmod +x reset_database.sh

# Run the reset (will prompt for confirmation)
./reset_database.sh
```

This script will:
1. Drop the entire database
2. Recreate it from scratch
3. Install required extensions (uuid-ossp, vector)
4. Run all migrations from the beginning
5. Seed the database with initial data

**⚠️ WARNING:** This **DELETES ALL DATA**. Only use in dev/staging!

---

### Strategy 2: Reset Alembic Version Only (Safer)

**Use when:** You want to preserve any existing data (if any tables exist)

```bash
cd /home/ubuntu/gmr
source venv/bin/activate

# Run the Alembic reset script
python alembic_reset.py

# If it detects corruption and resets, then run:
alembic upgrade head
python app/seed.py
```

This script:
1. Checks if your database is corrupted
2. Resets Alembic version to "base" if corrupted
3. Lets you re-run migrations from scratch
4. Preserves any existing tables (if any)

---

### Strategy 3: Manual Alembic Reset

**Use when:** You want full control over the process

```bash
cd /home/ubuntu/gmr
source venv/bin/activate

# Connect to your database
psql $DATABASE_URL

# Inside psql, run:
DELETE FROM alembic_version;
\q

# Back to shell - run migrations from scratch
alembic upgrade head
python app/seed.py
```

---

### Strategy 4: Manual Database Recreation (Script-Free)

**Use when:** Scripts don't work or you need to execute step by step

```bash
cd /home/ubuntu/gmr
source venv/bin/activate

# Get your database details from .env
cat .env | grep DATABASE_URL

# Connect to postgres (not your app database)
psql -h <host> -U <user> -d postgres

# In psql:
DROP DATABASE IF EXISTS <your_db_name>;
CREATE DATABASE <your_db_name>;
\c <your_db_name>
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
\q

# Run migrations
alembic upgrade head
python app/seed.py
```

---

## 🔍 Verification Steps

After running any fix, verify everything is working:

```bash
# 1. Check Alembic version is at the latest
alembic current

# 2. Verify tables exist
psql $DATABASE_URL -c "\dt"

# 3. Check specific critical tables
psql $DATABASE_URL -c "SELECT COUNT(*) FROM roles;"
psql $DATABASE_URL -c "SELECT COUNT(*) FROM cameras;"

# 4. Try running the seed script again
python app/seed.py
```

Expected results:
- `alembic current` should show the latest migration
- Tables should be listed (roles, users, cameras, track_sessions, etc.)
- Seed script should run without errors

---

## 📝 What Was Fixed

The following files were created/modified to prevent and fix this issue:

### 1. **alembic/versions/8b8b559916d9_add_best_crop_path_to_tracksession.py**
   - Improved column existence checking
   - Better compatibility with asyncpg driver
   - Graceful handling of existing columns

### 2. **reset_database.sh** (NEW)
   - Automated complete database reset
   - Handles extensions and migrations
   - Safe with confirmation prompts

### 3. **alembic_reset.py** (NEW)
   - Intelligent corruption detection
   - Resets Alembic version table only
   - Preserves existing data when possible

### 4. **fix_migration.py**
   - Original script for version stamp fixes
   - Less useful for current corruption level

---

## 🛡️ Prevention Tips

To avoid this issue in the future:

1. **Never manually modify the database schema** - always use Alembic migrations

2. **Before pulling new migrations:**
   ```bash
   alembic current  # Note your current version
   git pull
   alembic upgrade head
   ```

3. **If a migration fails mid-execution:**
   ```bash
   # Don't panic! Check what happened:
   alembic current
   psql $DATABASE_URL -c "\dt"
   
   # If tables are missing, downgrade and retry:
   alembic downgrade -1
   alembic upgrade +1
   ```

4. **Always backup production before migrations:**
   ```bash
   pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
   ```

5. **Test migrations in staging first** before applying to production

---

## 🆘 Still Having Issues?

### Common Errors & Solutions

**Error:** `psql: FATAL: database does not exist`
- **Solution:** Use Strategy 1 or 4 to recreate the database

**Error:** `relation "alembic_version" does not exist`
- **Solution:** Run `alembic upgrade head` - Alembic will create it

**Error:** `permission denied` on scripts
- **Solution:** Run `chmod +x reset_database.sh` before executing

**Error:** Scripts can't connect to database
- **Solution:** Check `.env` file has correct `DATABASE_URL`

**Error:** Extensions not available (uuid-ossp, vector)
- **Solution:** Install PostgreSQL contrib and pgvector:
  ```bash
  sudo apt-get install postgresql-contrib
  # For pgvector, follow: https://github.com/pgvector/pgvector
  ```

---

## 📞 Need More Help?

1. Check PostgreSQL logs: `tail -f /var/log/postgresql/*.log`
2. Verify database connection: `psql $DATABASE_URL -c "SELECT version();"`
3. Check Alembic migration history: `alembic history`
4. Review migration file revisions for conflicts

---

## ✅ Quick Decision Tree

```
Is this a production database with important data?
├─ YES → Strategy 2 (Alembic reset only) or Strategy 3 (Manual)
└─ NO  → Strategy 1 (Complete reset) ← FASTEST

Can you run bash scripts?
├─ YES → Use reset_database.sh or alembic_reset.py
└─ NO  → Follow Strategy 4 (Manual commands)

Do you understand what happened?
├─ YES → Pick any strategy that fits your needs
└─ NO  → Use Strategy 1 (it's fully automated)
```

---

**TL;DR for Development:**
```bash
cd /home/ubuntu/gmr
source venv/bin/activate
chmod +x reset_database.sh
./reset_database.sh   # Type 'yes' when prompted
```

Done! ✅
