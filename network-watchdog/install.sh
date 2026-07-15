#!/usr/bin/env bash
# Install / update network-watchdog from this repo copy into the host systemd paths.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
need_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run as root: sudo $0 $*"
    exit 1
  fi
}

need_root

install -d -m 755 /var/lib/net-watchdog
install -d -m 755 /var/log
touch /var/log/net-watchdog.log
chmod 644 /var/log/net-watchdog.log

install -m 755 "$SRC/net-watchdog.sh" /usr/local/sbin/net-watchdog.sh
install -m 644 "$SRC/net-watchdog.service" /etc/systemd/system/net-watchdog.service
install -m 644 "$SRC/net-watchdog.timer" /etc/systemd/system/net-watchdog.timer

systemctl daemon-reload
systemctl enable --now net-watchdog.timer
systemctl restart net-watchdog.timer

echo "Installed. Status:"
systemctl status net-watchdog.timer --no-pager || true
systemctl list-timers --all | grep net-watchdog || true
echo "Manual run: systemctl start net-watchdog.service"
echo "Logs:       tail -f /var/log/net-watchdog.log"
