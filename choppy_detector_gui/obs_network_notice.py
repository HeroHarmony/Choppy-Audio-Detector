"""Helpers for OBS network warning visibility in bundled macOS builds."""

from __future__ import annotations

import ipaddress
import sys


def is_local_lan_host(host: str) -> bool:
    host_text = str(host or "").strip().lower()
    if not host_text or host_text == "localhost":
        return False
    try:
        addr = ipaddress.ip_address(host_text)
    except ValueError:
        return False
    return bool(addr.is_private and not addr.is_loopback)


def should_show_unsigned_bundle_notice(host: str) -> bool:
    if sys.platform != "darwin":
        return False
    if not bool(getattr(sys, "frozen", False)):
        return False
    return is_local_lan_host(host)
