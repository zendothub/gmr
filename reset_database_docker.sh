#!/bin/bash
# Script to reset the database running in Docker container

set -e  # Exit on error

echo "🔄 Database Reset Script (Docker Version)"
echo "=========================================="
echo ""

# Database details from docker-compose.yml
DB_CONTAINER="retail_ai_postgres"
DB_NAME="retail_ai_db"
DB_USER="retail_user"
DB_PASS="retail_pass"

echo "Container: $DB_CONTAINER"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo ""

# Check if Docker container is running
if ! docker ps | grep -q $DB_CONTAINER; then
    echo "❌ Error: PostgreSQL container '$DB_CONTAINER' is not running!"
    echo ""
    echo "Start it with:"
    echo "  docker-compose up -d postgres"
    exit 1
fi

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
docker exec -e PGPASSWORD=$DB_PASS $DB_CONTAINER psql -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>&1 || {
    echo "⚠️  Could not drop database. It may not exist."
}

echo "📦 Creating new database..."
docker exec -e PGPASSWORD=$DB_PASS $DB_CONTAINER psql -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;" || {
    echo "❌ Failed to create database"
    exit 1
}

echo "🔌 Installing extensions..."
docker exec -e PGPASSWORD=$DB_PASS $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" || {
    echo "⚠️  Failed to install uuid-ossp extension"
}

docker exec -e PGPASSWORD=$DB_PASS $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;" || {
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
