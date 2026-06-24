"""Live current air-raid state via the alerts.in.ua API (optional).

Design: the project is fully wired for live data, but runs WITHOUT a token —
`get_live_active_regions()` returns None when no token / on any error, and the
dashboard then falls back to the latest snapshot from the historical series.

A developer enables live mode at deploy time by setting ALERTS_IN_UA_TOKEN in the
environment (free, non-commercial token from https://devs.alerts.in.ua/).

API etiquette: alerts.in.ua asks clients to cache and poll no more than ~once per
15-30s. The dashboard wraps this with a 30s TTL cache.
"""
from __future__ import annotations

import os

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from . import geo

ACTIVE_ALERTS_URL = "https://api.alerts.in.ua/v1/alerts/active.json"
AIR_RAID = "air_raid"


def parse_active_regions(payload: dict) -> set[str]:
    """Map an alerts.in.ua /active.json payload to canonical oblasts under an
    active AIR-RAID alert. Raion/hromada alerts roll up to their oblast.
    Pure function (no I/O) so it is unit-testable.
    """
    active: set[str] = set()
    for a in payload.get("alerts", []):
        if a.get("alert_type") != AIR_RAID:
            continue
        if a.get("finished_at"):  # defensive; active.json returns only active
            continue
        name = a.get("location_oblast") or a.get("location_title")
        region = geo.normalize_region(name)
        if region:
            active.add(region)
    return active


def fetch_active(token: str, timeout: int = 10) -> dict:
    resp = requests.get(
        ACTIVE_ALERTS_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def get_live_active_regions(token: str | None = None) -> set[str] | None:
    """Set of oblasts currently under an air-raid alert, or None if live mode is
    unavailable (no token, network/API error) — caller should fall back."""
    token = token or os.getenv("ALERTS_IN_UA_TOKEN")
    if not token:
        return None
    try:
        return parse_active_regions(fetch_active(token))
    except Exception as exc:  # noqa: BLE001 - any failure => graceful fallback
        print(f"[live] alerts.in.ua unavailable ({exc!r}); falling back to snapshot")
        return None


def is_live_enabled() -> bool:
    return bool(os.getenv("ALERTS_IN_UA_TOKEN"))


if __name__ == "__main__":
    if not is_live_enabled():
        print("[live] ALERTS_IN_UA_TOKEN not set -> snapshot fallback mode")
    regions = get_live_active_regions()
    if regions is None:
        print("[live] live mode OFF (no token or API error)")
    else:
        print(f"[live] active air-raid oblasts now ({len(regions)}):")
        for r in sorted(regions):
            print(f"  - {r}")
