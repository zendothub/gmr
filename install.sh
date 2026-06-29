#!/bin/bash
# Installation script for Retail AI systemd service
# This script installs the retail-ai.service from the systemd/ folder

set -e

echo "=== Retail AI Service Installation ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run with sudo: sudo bash install.sh"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$PROJECT_ROOT/systemd/retail-ai.service"
DOCKER_SERVICE_FILE="$PROJECT_ROOT/systemd/retail-ai-docker.service"

# Check if service files exist
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Error: Service file not found at $SERVICE_FILE"
    exit 1
fi

if [ ! -f "$DOCKER_SERVICE_FILE" ]; then
    echo "❌ Error: Docker service file not found at $DOCKER_SERVICE_FILE"
    exit 1
fi

# Stop any existing services
echo "🛑 Stopping existing services (if running)..."
systemctl stop retail-ai.service 2>/dev/null || true
systemctl stop retail-ai-docker.service 2>/dev/null || true

# Copy service files
echo "📋 Installing service files..."
cp "$SERVICE_FILE" /etc/systemd/system/retail-ai.service
cp "$DOCKER_SERVICE_FILE" /etc/systemd/system/retail-ai-docker.service

# Set permissions
chmod 644 /etc/systemd/system/retail-ai.service
chmod 644 /etc/systemd/system/retail-ai-docker.service

# Reload systemd
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Ensure logs directory exists
echo "📁 Creating logs directory..."
mkdir -p /gmr/gmr/logs
chown retaileye:retaileye /gmr/gmr/logs

echo ""
echo "✅ Installation Complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Available commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Start the service:"
echo "    sudo systemctl start retail-ai"
echo ""
echo "  Enable auto-start on boot:"
echo "    sudo systemctl enable retail-ai"
echo ""
echo "  Start and enable in one command:"
echo "    sudo systemctl enable --now retail-ai"
echo ""
echo "  Check status:"
echo "    sudo systemctl status retail-ai"
echo ""
echo "  View logs:"
echo "    sudo journalctl -u retail-ai -f"
echo "    tail -f /gmr/gmr/logs/retail-ai.log"
echo ""
echo "  Stop the service:"
echo "    sudo systemctl stop retail-ai"
echo ""
echo "  Restart the service:"
echo "    sudo systemctl restart retail-ai"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
