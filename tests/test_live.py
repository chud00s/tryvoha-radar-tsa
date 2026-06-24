"""Tests for alerts.in.ua live-payload parsing (pure, no network)."""
from src.live import get_live_active_regions, parse_active_regions

SAMPLE = {
    "alerts": [
        {"alert_type": "air_raid", "finished_at": None,
         "location_oblast": "Харківська область", "location_title": "Харківська область",
         "location_type": "oblast"},
        # raion-level air-raid rolls up to its oblast
        {"alert_type": "air_raid", "finished_at": None,
         "location_oblast": "Одеська область", "location_title": "Ізмаїльський район",
         "location_type": "raion"},
        # non air-raid alert is ignored
        {"alert_type": "artillery", "finished_at": None,
         "location_oblast": "Сумська область", "location_title": "Сумська область",
         "location_type": "oblast"},
        # finished air-raid is ignored
        {"alert_type": "air_raid", "finished_at": "2026-06-22T01:00:00+00:00",
         "location_oblast": "Львівська область", "location_title": "Львівська область",
         "location_type": "oblast"},
    ]
}


def test_parse_active_regions_filters_and_maps():
    active = parse_active_regions(SAMPLE)
    assert active == {"Kharkivska oblast", "Odeska oblast"}


def test_parse_empty_payload():
    assert parse_active_regions({}) == set()


def test_live_off_without_token(monkeypatch):
    monkeypatch.delenv("ALERTS_IN_UA_TOKEN", raising=False)
    # no token -> graceful None (snapshot fallback), no network call
    assert get_live_active_regions() is None
