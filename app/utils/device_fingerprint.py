"""Device fingerprinting — derives a stable device ID from request headers."""

from __future__ import annotations

import hashlib


def fingerprint(user_agent: str, ip_address: str) -> str:
    """Return a SHA-256 hash of User-Agent + /24 IP prefix.

    Two requests from the same device (same browser + same subnet) produce the
    same hash. Different browsers, different subnets, or VPN changes produce
    different hashes.
    """
    ip_prefix = _ip_slash_24(ip_address)
    raw = f"{user_agent}|{ip_prefix}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def device_label(user_agent: str) -> str:
    """Derive a human-readable label from the User-Agent string.

    Returns something like ``"Chrome 131 on macOS"`` or ``"Firefox 130 on Windows"``
    or ``"Unknown Device"`` if parsing fails.
    """
    ua = user_agent.lower()

    # Browser detection
    browser = "Unknown"
    if "edg/" in ua:
        browser = "Edge"
    elif "chrome/" in ua:
        browser = "Chrome"
    elif "firefox/" in ua:
        browser = "Firefox"
    elif "safari/" in ua:
        browser = "Safari"
    elif "opr/" in ua or "opera/" in ua:
        browser = "Opera"

    # OS detection
    os_name = ""
    if "mac os" in ua or "macintosh" in ua:
        os_name = "macOS"
    elif "windows nt" in ua:
        os_name = "Windows"
    elif "linux" in ua or "x11" in ua:
        os_name = "Linux"
    elif "android" in ua:
        os_name = "Android"
    elif "iphone" in ua or "ipad" in ua:
        os_name = "iOS"

    if os_name:
        return f"{browser} on {os_name}"
    return browser


def _ip_slash_24(ip: str) -> str:
    """Return the /24 subnet prefix of an IP (first 3 octets for IPv4)."""
    # Handle IPv4
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3])
    # Handle IPv6 — use /48 prefix
    parts = ip.split(":")
    if len(parts) >= 3:
        return ":".join(parts[:3])
    return ip  # fallback — shouldn't happen