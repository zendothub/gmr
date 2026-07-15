#!/usr/bin/env bash
# Network recovery watchdog for Retail Eye host.
# Hard fail  = cannot reach gateway (host/LAN stuck) → escalate recoveries + reboot.
# Soft fail  = gateway OK but internet dead → log only (likely ISP).
set -euo pipefail

GW="${NET_WATCHDOG_GW:-192.168.1.1}"
INET="${NET_WATCHDOG_INET:-8.8.8.8}"
IFACE="${NET_WATCHDOG_IFACE:-eno1}"
CONN="${NET_WATCHDOG_CONN:-Wired connection 1}"

CHECK_INTERVAL_SEC=120
FAIL_THRESHOLD_SEC=360          # 6 minutes
REBOOT_COOLDOWN_SEC=5400        # 1.5 hours
POST_ACTION_WAIT_SEC=25
NM_WAIT_SEC=35

STATE_DIR="${NET_WATCHDOG_STATE_DIR:-/var/lib/net-watchdog}"
LOG_FILE="${NET_WATCHDOG_LOG:-/var/log/net-watchdog.log}"
STATE_FILE="${STATE_DIR}/state"
LOCK_FILE="${STATE_DIR}/lock"

mkdir -p "$STATE_DIR"
touch "$LOG_FILE" 2>/dev/null || true

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"
  echo "$msg" | tee -a "$LOG_FILE" >/dev/null 2>&1 || echo "$msg"
}

ping_ok() {
  local target="$1"
  ping -c1 -W2 "$target" >/dev/null 2>&1
}

load_state() {
  HARD_FAIL_SINCE=0
  LAST_REBOOT_TS=0
  LAST_ACTION=""
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE" || true
  fi
  HARD_FAIL_SINCE=${HARD_FAIL_SINCE:-0}
  LAST_REBOOT_TS=${LAST_REBOOT_TS:-0}
  LAST_ACTION=${LAST_ACTION:-}
}

save_state() {
  cat >"$STATE_FILE" <<EOF
HARD_FAIL_SINCE=${HARD_FAIL_SINCE:-0}
LAST_REBOOT_TS=${LAST_REBOOT_TS:-0}
LAST_ACTION=${LAST_ACTION:-}
EOF
}

clear_hard_fail() {
  if [[ "${HARD_FAIL_SINCE:-0}" -ne 0 ]]; then
    log "OK: connectivity restored (gw=${GW} inet=${INET}); clearing hard-fail streak"
  fi
  HARD_FAIL_SINCE=0
  LAST_ACTION=""
  save_state
}

recover_level1() {
  log "RECOVERY L1: neigh flush + nmcli bounce '${CONN}' on ${IFACE}"
  ip neigh flush dev "$IFACE" 2>/dev/null || true
  nmcli connection down "$CONN" 2>/dev/null || true
  sleep 2
  nmcli connection up "$CONN" 2>/dev/null || true
  sleep "$POST_ACTION_WAIT_SEC"
}

recover_level2() {
  log "RECOVERY L2: systemctl restart NetworkManager"
  systemctl restart NetworkManager || true
  sleep "$NM_WAIT_SEC"
  nmcli connection up "$CONN" 2>/dev/null || true
  sleep "$POST_ACTION_WAIT_SEC"
}

maybe_reboot() {
  local now
  now=$(date +%s)
  local elapsed=$((now - LAST_REBOOT_TS))
  if [[ "$LAST_REBOOT_TS" -gt 0 && "$elapsed" -lt "$REBOOT_COOLDOWN_SEC" ]]; then
    log "RECOVERY L3: reboot deferred (cooldown ${REBOOT_COOLDOWN_SEC}s, last reboot ${elapsed}s ago)"
    return 0
  fi
  log "RECOVERY L3: system reboot (gateway still unreachable after recoveries)"
  LAST_REBOOT_TS=$now
  LAST_ACTION=reboot
  save_state
  # Give log a chance to flush
  sync || true
  systemctl reboot || reboot
}

check_after_recovery() {
  if ping_ok "$GW"; then
    log "RECOVERY: gateway ${GW} reachable after action"
    if ping_ok "$INET"; then
      clear_hard_fail
      return 0
    fi
    log "SOFT: gateway OK, internet ${INET} still down after recovery (likely ISP)"
    clear_hard_fail
    return 0
  fi
  return 1
}

# --- main ---
# Serialize concurrent timer fires
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "SKIP: another watchdog run holds lock"
  exit 0
fi

load_state
now=$(date +%s)

gw_ok=0
inet_ok=0
ping_ok "$GW" && gw_ok=1
ping_ok "$INET" && inet_ok=1

if [[ "$gw_ok" -eq 1 && "$inet_ok" -eq 1 ]]; then
  clear_hard_fail
  exit 0
fi

if [[ "$gw_ok" -eq 1 && "$inet_ok" -eq 0 ]]; then
  log "SOFT_FAIL: gateway ${GW} OK, internet ${INET} down (likely Airtel/upstream — no reboot)"
  # Do not treat as hard host failure; keep streak clear so ISP outages don't escalate
  HARD_FAIL_SINCE=0
  save_state
  exit 0
fi

# Hard fail: gateway unreachable
if [[ "$HARD_FAIL_SINCE" -eq 0 ]]; then
  HARD_FAIL_SINCE=$now
  save_state
  log "HARD_FAIL: gateway ${GW} unreachable (streak started)"
  exit 0
fi

streak=$((now - HARD_FAIL_SINCE))
log "HARD_FAIL: gateway ${GW} still down for ${streak}s (threshold ${FAIL_THRESHOLD_SEC}s)"

if [[ "$streak" -lt "$FAIL_THRESHOLD_SEC" ]]; then
  save_state
  exit 0
fi

# Escalate
if [[ "${LAST_ACTION:-}" != "l1" && "${LAST_ACTION:-}" != "l2" && "${LAST_ACTION:-}" != "reboot" ]]; then
  LAST_ACTION=l1
  save_state
  recover_level1
  if check_after_recovery; then
    exit 0
  fi
fi

if [[ "${LAST_ACTION:-}" != "l2" && "${LAST_ACTION:-}" != "reboot" ]]; then
  LAST_ACTION=l2
  save_state
  recover_level2
  if check_after_recovery; then
    exit 0
  fi
fi

# Still hard-fail after L1+L2
maybe_reboot
exit 0
