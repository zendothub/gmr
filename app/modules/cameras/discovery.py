"""MAC-based camera IP re-discovery.

Cameras are addressed by rtsp_url (which bakes in an IP). When a camera's
network identity changes underneath it - DHCP lease renewal after a power
cycle, Ethernet renegotiation, router reboot - the stored rtsp_url points at
a dead address and the stream never recovers on its own.

A camera's MAC address does not change. This module scans the camera's own
/24 subnet, matches the OS ARP table against the camera's known MAC, and
returns its current IP so the caller can patch rtsp_url and reconnect -
without touching which zone/location the camera row represents.

The scan itself is plain synchronous subprocess calls (parallelized over a
thread pool for the sweep), not asyncio. It used to spin up a fresh asyncio
event loop per call via asyncio.run() from frame_buffer's dedicated capture
thread; on macOS that leaks file descriptors across repeated loop creation,
and enough failed reconnect retries during an extended camera outage would
exhaust the whole process's fd limit and take the entire server down (not
just this camera). Plain subprocess.run() + a thread pool needs no event
loop at all, so there's nothing to leak.
"""

import platform
import re
import socket
import time
import uuid as uuidlib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from ipaddress import ip_address, ip_network
from typing import Optional
from urllib.parse import urlparse
import asyncio
import subprocess

import psutil
from loguru import logger

_MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$")
_ARP_LINE_RE = re.compile(r"\(([\d.]+)\)\s+at\s+([0-9a-fA-F:]{17})")
_PING_TIMEOUT_S = 1
_MAX_CONCURRENT_PINGS = 50


def normalize_mac(mac: str) -> str:
    return mac.strip().lower().replace("-", ":")


def is_valid_mac(mac: str) -> bool:
    return bool(_MAC_RE.match(mac.strip()))


def subnet_from_ip(ip: str, prefix: int = 24) -> str:
    """Derive the /24 (by default) containing `ip`, e.g. '192.168.0.84' -> '192.168.0.0/24'."""
    return str(ip_network(f"{ip}/{prefix}", strict=False))


def local_subnet_for_ip(ip: str) -> str:
    """Return the real CIDR of whichever local network interface's subnet
    contains `ip`, read from this machine's actual netmask.

    A camera network isn't always a /24 - it's whatever the router's DHCP
    pool is (a /23 spanning two /24s is common). Assuming /24 means a camera
    that lands outside that boundary after an IP change is permanently
    unfindable even though it's on the same physical network. Reading the
    real netmask off the interface that's actually on this network removes
    the assumption entirely.

    Falls back to a /24 guess if no local interface's subnet contains the
    IP (e.g. this process doesn't have a direct L2 view of the camera's LAN).
    """
    try:
        target = ip_address(ip)
    except ValueError:
        return subnet_from_ip(ip)

    try:
        addrs = psutil.net_if_addrs()
    except Exception as e:
        logger.error(f"Could not read local network interfaces: {e}")
        return subnet_from_ip(ip)

    for snics in addrs.values():
        for snic in snics:
            if snic.family != socket.AF_INET or not snic.netmask:
                continue
            try:
                network = ip_network(f"{snic.address}/{snic.netmask}", strict=False)
            except ValueError:
                continue
            if target in network:
                return str(network)

    return subnet_from_ip(ip)


def extract_host(rtsp_url: str) -> Optional[str]:
    """Pull the IP/host out of an rtsp:// URL (rtsp://user:pass@host:port/path)."""
    match = re.match(r"^rtsp://(?:[^@/]+@)?([^:/]+)", rtsp_url)
    return match.group(1) if match else None


def rebuild_rtsp_url(old_rtsp_url: str, new_ip: str) -> str:
    """Swap the host portion of an rtsp:// URL for `new_ip`, keeping creds/port/path intact."""
    match = re.match(r"^(rtsp://(?:[^@/]+@)?)([^:/]+)(.*)$", old_rtsp_url)
    if not match:
        raise ValueError(f"Unrecognized RTSP URL format: {old_rtsp_url}")
    prefix, _old_host, suffix = match.groups()
    return f"{prefix}{new_ip}{suffix}"


def _ping_sync(host: str) -> None:
    ping_count_flag = "-n" if platform.system() == "Windows" else "-c"
    try:
        subprocess.run(
            ["ping", ping_count_flag, "1", "-W", str(_PING_TIMEOUT_S), host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_PING_TIMEOUT_S + 1,
        )
    except Exception:
        pass  # unreachable hosts are expected; we only care about ARP side-effects


def _ping_sweep_sync(subnet: str) -> None:
    """Populate the OS ARP cache by pinging every host in `subnet`, in parallel."""
    network = ip_network(subnet, strict=False)
    hosts = [str(host) for host in network.hosts()]
    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_PINGS) as pool:
        list(pool.map(_ping_sync, hosts))


def _read_arp_table_sync() -> dict:
    """Return {normalized_mac: ip} from the OS ARP cache."""
    try:
        result = subprocess.run(
            ["arp", "-a"],
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        logger.error(f"Failed to read ARP table: {e}")
        return {}
    mac_to_ip = {}
    for line in result.stdout.decode(errors="ignore").splitlines():
        m = _ARP_LINE_RE.search(line)
        if m:
            ip, mac = m.group(1), normalize_mac(m.group(2))
            mac_to_ip[mac] = ip
    return mac_to_ip


def resolve_ip_by_mac_sync(mac_address: str, subnet: str) -> Optional[str]:
    """Scan `subnet` for `mac_address`, return its current IP or None if not found.

    Not found means the camera is actually offline (powered off / unplugged) -
    caller should keep normal crash-retry/backoff behavior in that case rather
    than treating it as a simple IP change.

    Synchronous and creates no event loop - safe to call directly from a
    plain background thread (e.g. frame_buffer's dedicated capture thread)
    on every reconnect retry without leaking resources.
    """
    if not mac_address or not is_valid_mac(mac_address):
        return None

    target_mac = normalize_mac(mac_address)
    try:
        _ping_sweep_sync(subnet)
        arp_table = _read_arp_table_sync()
    except Exception as e:
        logger.error(f"MAC discovery scan failed for subnet {subnet}: {e}")
        return None

    ip = arp_table.get(target_mac)
    if ip:
        logger.info(f"Camera MAC {target_mac} resolved to IP {ip} on subnet {subnet}")
    else:
        logger.warning(f"Camera MAC {target_mac} not found on subnet {subnet} - camera likely offline")
    return ip


def resolve_mac_by_ip_sync(ip: str) -> Optional[str]:
    """Look up the MAC address currently answering at `ip`, or None if it
    can't be determined (camera unreachable, ARP entry not yet populated).
    Synchronous - see resolve_ip_by_mac_sync for why.
    """
    try:
        _ping_sync(ip)
        arp_table = _read_arp_table_sync()
    except Exception as e:
        logger.error(f"MAC lookup failed for {ip}: {e}")
        return None

    for mac, mapped_ip in arp_table.items():
        if mapped_ip == ip:
            logger.info(f"Camera IP {ip} resolved to MAC {mac}")
            return mac

    logger.warning(f"Could not resolve MAC for {ip} - camera may be unreachable or ARP entry not yet populated")
    return None


async def resolve_ip_by_mac(mac_address: str, subnet: str) -> Optional[str]:
    """Async wrapper for callers already inside a running event loop (e.g. a
    FastAPI request handler) - runs the sync scan on the loop's default
    thread-pool executor so it never blocks the loop. A caller on a plain
    background thread with no running loop should call
    resolve_ip_by_mac_sync directly instead (see frame_buffer.py).
    """
    return await asyncio.to_thread(resolve_ip_by_mac_sync, mac_address, subnet)


async def resolve_mac_by_ip(ip: str) -> Optional[str]:
    """Async wrapper - see resolve_ip_by_mac."""
    return await asyncio.to_thread(resolve_mac_by_ip_sync, ip)


# ---------------------------------------------------------------------------
# ONVIF (WS-Discovery) - fallback identity for cameras whose MAC rotates.
#
# A Wi-Fi camera with "private address" / MAC randomization enabled presents
# a new MAC every time it reconnects, which breaks MAC-based re-discovery by
# design - there's no stable MAC to search for. Most ONVIF-compliant IP
# cameras (the large majority of commercial ones) also answer WS-Discovery,
# a UDP multicast probe/response protocol, with a permanent device endpoint
# UUID assigned at the firmware/hardware level. That UUID doesn't rotate
# with the MAC, so it survives exactly the case MAC-based lookup can't
# handle. It also doesn't need a subnet/prefix guess at all: multicast
# naturally reaches only the local network segment, and the response
# includes the device's current address directly - one broadcast instead of
# a per-host ping sweep.
# ---------------------------------------------------------------------------

_WSD_MULTICAST_ADDR = "239.255.255.250"
_WSD_MULTICAST_PORT = 3702
_WSD_NS = {
    "wsa": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
    "wsd": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
}


def _ws_discovery_probe(timeout_s: float = 3.0) -> dict:
    """Send a WS-Discovery Probe over UDP multicast and collect responses.

    Returns {onvif_uuid: xaddr_url} for every ONVIF device that answers on
    this network segment within `timeout_s`.
    """
    message_id = f"urn:uuid:{uuidlib.uuid4()}"
    probe = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:wsd="http://schemas.xmlsoap.org/ws/2005/04/discovery">'
        "<soap:Header>"
        "<wsa:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>"
        "<wsa:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>"
        f"<wsa:MessageID>{message_id}</wsa:MessageID>"
        "</soap:Header>"
        "<soap:Body>"
        '<wsd:Probe><wsd:Types>dn:NetworkVideoTransmitter</wsd:Types></wsd:Probe>'
        "</soap:Body>"
        "</soap:Envelope>"
    ).encode("utf-8")

    results: dict = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout_s)
        sock.sendto(probe, (_WSD_MULTICAST_ADDR, _WSD_MULTICAST_PORT))

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, _addr = sock.recvfrom(65535)
            except (socket.timeout, OSError):
                break
            parsed = _parse_probe_match(data)
            if parsed:
                dev_uuid, xaddr = parsed
                results[dev_uuid] = xaddr
    finally:
        sock.close()
    return results


def _parse_probe_match(data: bytes) -> Optional[tuple]:
    """Extract (device_uuid, xaddr_url) from a WS-Discovery ProbeMatch."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    addr_el = root.find(".//wsa:Address", _WSD_NS)
    xaddrs_el = root.find(".//wsd:XAddrs", _WSD_NS)
    if addr_el is None or xaddrs_el is None or not addr_el.text or not xaddrs_el.text:
        return None
    dev_uuid = addr_el.text.strip()
    xaddr = xaddrs_el.text.strip().split()[0]
    if not dev_uuid.startswith("urn:uuid:"):
        return None
    return dev_uuid, xaddr


def resolve_ip_by_onvif_id_sync(onvif_id: str, timeout_s: float = 3.0) -> Optional[str]:
    """Find the current IP of the ONVIF device whose stable endpoint UUID
    matches `onvif_id`, via a WS-Discovery probe on the local network
    segment. Falls back path when MAC-based lookup finds nothing - covers
    the case a camera's MAC has rotated (Wi-Fi privacy addressing) but its
    ONVIF identity hasn't.
    """
    if not onvif_id:
        return None
    try:
        matches = _ws_discovery_probe(timeout_s=timeout_s)
    except Exception as e:
        logger.error(f"WS-Discovery probe failed: {e}")
        return None

    xaddr = matches.get(onvif_id)
    if not xaddr:
        logger.warning(f"ONVIF device {onvif_id} did not answer WS-Discovery probe")
        return None

    host = urlparse(xaddr).hostname
    if host:
        logger.info(f"ONVIF device {onvif_id} resolved to IP {host} via WS-Discovery")
    return host


def resolve_onvif_id_by_ip_sync(ip: str, timeout_s: float = 3.0) -> Optional[str]:
    """Best-effort: find the ONVIF endpoint UUID of whatever device is
    currently at `ip`, by probing and matching on XAddrs host. Used to
    auto-capture a camera's onvif_id at creation time (mirrors
    resolve_mac_by_ip_sync for MAC) - returns None for non-ONVIF cameras,
    never raises.
    """
    try:
        matches = _ws_discovery_probe(timeout_s=timeout_s)
    except Exception as e:
        logger.error(f"WS-Discovery probe failed: {e}")
        return None

    for dev_uuid, xaddr in matches.items():
        if urlparse(xaddr).hostname == ip:
            logger.info(f"Camera IP {ip} resolved to ONVIF id {dev_uuid}")
            return dev_uuid
    return None


async def resolve_ip_by_onvif_id(onvif_id: str, timeout_s: float = 3.0) -> Optional[str]:
    """Async wrapper - see resolve_ip_by_mac."""
    return await asyncio.to_thread(resolve_ip_by_onvif_id_sync, onvif_id, timeout_s)


async def resolve_onvif_id_by_ip(ip: str, timeout_s: float = 3.0) -> Optional[str]:
    """Async wrapper - see resolve_ip_by_mac."""
    return await asyncio.to_thread(resolve_onvif_id_by_ip_sync, ip, timeout_s)
