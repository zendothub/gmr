#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup-local.sh  — configure retail-ai-platform for a local LAN deployment
#
# Usage:
#   ./scripts/setup-local.sh <LOCAL_IP>
#
# Example:
#   ./scripts/setup-local.sh 192.168.1.50
#
# What it does:
#   1. Writes LOCAL_IP into .env  (used as MEDIAMTX public host for WHEP/HLS URLs)
#   2. Updates MTX_WEBRTCADDITIONALHOSTS in docker-compose.yml (WebRTC ICE candidates)
#   3. Restarts the MediaMTX container so the new ICE host takes effect
#   4. Verifies the WHEP endpoint is reachable
#
# After running this script, start the backend with:
#   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── argument check ────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    error "No IP provided."
    echo ""
    echo "  Usage:  ./scripts/setup-local.sh <LOCAL_IP>"
    echo "  Example: ./scripts/setup-local.sh 192.168.1.50"
    exit 1
fi

LOCAL_IP="$1"

# Basic IP format validation
if ! echo "$LOCAL_IP" | grep -qE '^([0-9]{1,3}\.){3}[0-9]{1,3}$'; then
    error "Invalid IP address: $LOCAL_IP"
    exit 1
fi

# ── locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
COMPOSE_FILE="$REPO_ROOT/docker-compose.yml"

info "Repo root  : $REPO_ROOT"
info "Local IP   : $LOCAL_IP"
echo ""

# ── 1. Update .env — set MEDIAMTX_PUBLIC_URL to local IP:port ────────────────
# When MEDIAMTX_PUBLIC_URL is set, ALL WHEP/HLS URLs in API responses will use
# this base, so the frontend always gets the right LAN address regardless of
# which host it sent the request from.
# Format: http://<LOCAL_IP>  (no trailing slash; port is appended by the code)
MEDIAMTX_PUBLIC_URL_VALUE="http://${LOCAL_IP}"

if grep -q "^MEDIAMTX_PUBLIC_URL=" "$ENV_FILE"; then
    # Replace existing line (handles both empty and previously set values)
    sed -i.bak "s|^MEDIAMTX_PUBLIC_URL=.*|MEDIAMTX_PUBLIC_URL=${MEDIAMTX_PUBLIC_URL_VALUE}|" "$ENV_FILE"
else
    # Append if not present at all
    echo "" >> "$ENV_FILE"
    echo "MEDIAMTX_PUBLIC_URL=${MEDIAMTX_PUBLIC_URL_VALUE}" >> "$ENV_FILE"
fi
ok ".env  → MEDIAMTX_PUBLIC_URL=${MEDIAMTX_PUBLIC_URL_VALUE}"

# ── 2. Update docker-compose.yml — MTX_WEBRTCADDITIONALHOSTS ─────────────────
# Keep any existing IPs and prepend/replace the dynamic local IP at position 1
# (first entry wins for ICE candidate selection).
# Strategy: extract current value, replace leading IP (if it looks like an old
# local IP) OR prepend, then write back with sed.

CURRENT_HOSTS=$(grep "MTX_WEBRTCADDITIONALHOSTS" "$COMPOSE_FILE" | sed 's/.*"\(.*\)".*/\1/')

# Remove any existing IPs that look like they were set by a previous run of this
# script — i.e., the first comma-separated token (before the first comma).
# We always want the user-supplied LOCAL_IP as the first entry.
REMAINING_HOSTS=$(echo "$CURRENT_HOSTS" | sed 's/^[^,]*,//')

if [[ "$REMAINING_HOSTS" == "$CURRENT_HOSTS" ]]; then
    # No comma found — only one value; replace entirely
    NEW_HOSTS="${LOCAL_IP}"
else
    NEW_HOSTS="${LOCAL_IP},${REMAINING_HOSTS}"
fi

# Escape for sed replacement
NEW_HOSTS_ESCAPED=$(echo "$NEW_HOSTS" | sed 's/[\/&]/\\&/g')
sed -i.bak "s|MTX_WEBRTCADDITIONALHOSTS: \".*\"|MTX_WEBRTCADDITIONALHOSTS: \"${NEW_HOSTS_ESCAPED}\"|" "$COMPOSE_FILE"
ok "docker-compose.yml → MTX_WEBRTCADDITIONALHOSTS=\"${NEW_HOSTS}\""

# ── 3. Restart MediaMTX ───────────────────────────────────────────────────────
info "Restarting MediaMTX container..."
cd "$REPO_ROOT"
if ! docker compose up -d --no-deps mediamtx 2>&1; then
    error "Failed to restart MediaMTX. Is Docker running?"
    exit 1
fi
ok "MediaMTX restarted"

# Give it a moment to boot
sleep 2

# ── 4. Verify WHEP endpoint ───────────────────────────────────────────────────
info "Verifying WHEP endpoint on http://${LOCAL_IP}:8889 ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "http://${LOCAL_IP}:8889/cam_test/whep" 2>/dev/null; true)
# curl always writes the HTTP code (or "000" on connect fail) to stdout;
# the "; true" prevents set -e from aborting on curl exit != 0.

if [[ "$HTTP_CODE" == "405" ]]; then
    ok "WHEP endpoint is responding correctly (HTTP 405 — correct for GET; WHEP needs POST)"
elif [[ "$HTTP_CODE" == "404" ]]; then
    warn "WHEP returned 404 — MediaMTX is up but no stream is publishing yet. This is normal — start a stream first."
elif [[ "$HTTP_CODE" == "000" ]]; then
    warn "Could not reach http://${LOCAL_IP}:8889 — make sure this machine has IP ${LOCAL_IP} active."
    warn "Run:  ifconfig | grep 'inet '"
else
    warn "Unexpected HTTP ${HTTP_CODE} from WHEP endpoint."
fi

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Local setup complete for IP: ${LOCAL_IP}${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Services:"
echo "    Backend API  : http://${LOCAL_IP}:8000          (start manually below)"
echo "    API Docs     : http://${LOCAL_IP}:8000/docs"
echo "    WebRTC/WHEP  : http://${LOCAL_IP}:8889/<path>/whep"
echo "    HLS          : http://${LOCAL_IP}:8888/<path>/index.m3u8"
echo "    MinIO        : http://${LOCAL_IP}:9001  (console)"
echo ""
echo "  Start the backend:"
echo "    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"
echo ""
echo "  Add your camera via API (Swagger at /docs):"
echo "    POST /api/cameras  →  rtsp_url: rtsp://admin:admin@<camera-ip>:<port>"
echo ""
echo "  Then start streaming:"
echo "    POST /api/cameras/{id}/stream/start"
echo "    → returns WHEP URL: http://${LOCAL_IP}:8889/cam_<id>/whep"
echo ""
