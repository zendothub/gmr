#!/bin/bash
# Script to completely reset the database and re-run all migrations from scratch

set -e  # Exit on error

echo "🔄 Database Reset Script"
echo "========================"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Extract database connection details from DATABASE_URL
DB_NAME=$(echo $DATABASE_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')
DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
DB_USER=$(echo $DATABASE_URL | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
DB_PASS=$(echo $DATABASE_URL | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')

echo "Database: $DB_NAME"
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo "User: $DB_USER"
echo ""

# Set PGPASSWORD for psql commands
export PGPASSWORD=$DB_PASS

echo "⚠️  WARNING: This will DROP and RECREATE the database!"
echo "All existing data will be LOST."
echo ""
read -p "Are you sure you want to continue? (type 'yes' to proceed): " -r
echo ""

if [[ ! $REPLY == "yes" ]]; then
    echo "❌ Operation cancelled."
    exit 1
fi

echo "🗑️  Dropping existing database..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>&1 || {
    echo "⚠️  Could not drop database. It may not exist or you may not have permissions."
}

echo "📦 Creating new database..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;" || {
    echo "❌ Failed to create database"
    exit 1
}

echo "🔌 Installing extensions..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" || {
    echo "⚠️  Failed to install uuid-ossp extension"
}

psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;" || {
    echo "⚠️  Failed to install vector extension"
}

echo "🔄 Running Alembic migrations..."
alembic upgrade head || {
    echo "❌ Alembic migrations failed"
    exit 1
}

echo "🌱 Running seed script..."
python app/seed.py || {
    echo "❌ Seed script failed"
    exit 1
}

echo ""
echo "✅ Database reset and migrations completed successfully!"
echo ""
