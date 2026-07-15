# Network Watchdog

Host-side recovery for stuck WAN/LAN after ISP (Airtel) blips or NIC/NetworkManager hangs.

Repo copy lives here; install copies into system paths (see `install.sh`).

## Behaviour

| Interval | Every **2 minutes** (`net-watchdog.timer`) |
|---|---|
| Soft fail | Gateway `192.168.1.1` OK, `8.8.8.8` fail → log only (likely ISP). **No reboot.** |
| Hard fail | Gateway unreachable |
| Threshold | Hard fail continuous **≥ 6 minutes** before recovery |
| L1 | `ip neigh flush` + `nmcli` down/up **Wired connection 1** (`eno1`) |
| L2 | `systemctl restart NetworkManager` + re-up connection |
| L3 | `systemctl reboot` |
| Reboot cooldown | **1.5 hours** (no reboot loop during long Airtel outages) |

State: `/var/lib/net-watchdog/state`  
Log: `/var/log/net-watchdog.log`

## Static camera RTSP IPs

**Not controlled by this host or this script.** The PC only *consumes* RTSP URLs stored in Postgres (`cameras.rtsp_url`).

To keep camera IPs stable (`.5` counter, `.6` entry):

1. **Best on Airtel AirFibre:** Router admin → DHCP / LAN → **DHCP reservation / static lease** by MAC  
   - Counter: `00:1c:27:26:84:ec` → `192.168.1.5`  
   - Entry: `00:1c:27:26:84:28` → `192.168.1.6`  
2. **Or** set static IPv4 **in each camera’s web UI** (same subnet, gateway `192.168.1.1`, DNS optional).  
3. From this machine alone you **cannot** force cameras to keep an IP after DHCP reassign (unless you use a local DHCP server you control — not Airtel’s).

After any IP change, update DB `cameras.rtsp_url` (password path stays the same).

## Install / update on host

```bash
cd /gmr/gmr/network-watchdog
sudo ./install.sh
```

## Ops

```bash
systemctl status net-watchdog.timer
systemctl start net-watchdog.service    # run once now
journalctl -u net-watchdog.service -n 50
tail -f /var/log/net-watchdog.log
```

## Files

| Repo | Installed |
|---|---|
| `net-watchdog.sh` | `/usr/local/sbin/net-watchdog.sh` |
| `net-watchdog.service` | `/etc/systemd/system/net-watchdog.service` |
| `net-watchdog.timer` | `/etc/systemd/system/net-watchdog.timer` |

Tunable env vars (optional override in a drop-in unit):  
`NET_WATCHDOG_GW`, `NET_WATCHDOG_INET`, `NET_WATCHDOG_IFACE`, `NET_WATCHDOG_CONN`.
