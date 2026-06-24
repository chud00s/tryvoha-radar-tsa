"""Tests for interval -> hourly time-series transformation."""
import pandas as pd

from src.transform import build_series


def _alerts():
    return pd.DataFrame({
        "region": ["Kyiv City", "Kyiv City", "Lvivska oblast"],
        "start": pd.to_datetime(
            ["2022-03-01 10:15", "2022-03-01 12:00", "2022-03-01 10:00"], utc=True),
        "end": pd.to_datetime(
            ["2022-03-01 11:30", "2022-03-01 12:20", "2022-03-01 10:50"], utc=True),
    })


def test_grid_is_gap_free_and_complete():
    s = build_series(_alerts())
    # hours 10, 11, 12 across 25 regions
    assert s["ts"].nunique() == 3
    assert s["region"].nunique() == 25
    assert len(s) == 75
    # exactly 3 hours per region (no gaps, no dupes)
    assert (s.groupby("region").size() == 3).all()


def test_active_flags_and_counts():
    s = build_series(_alerts()).set_index(["region", "ts"])
    h = lambda x: pd.Timestamp(x, tz="UTC")

    # Kyiv: 10:15-11:30 covers hours 10 & 11; 12:00-12:20 covers hour 12
    assert s.loc[("Kyiv City", h("2022-03-01 10:00")), "active"] == 1
    assert s.loc[("Kyiv City", h("2022-03-01 11:00")), "active"] == 1
    assert s.loc[("Kyiv City", h("2022-03-01 12:00")), "active"] == 1
    # two alerts started: one at hour 10, one at hour 12
    assert s.loc[("Kyiv City", h("2022-03-01 10:00")), "starts"] == 1
    assert s.loc[("Kyiv City", h("2022-03-01 11:00")), "starts"] == 0
    assert s.loc[("Kyiv City", h("2022-03-01 12:00")), "starts"] == 1

    # Lviv: only hour 10 active
    assert s.loc[("Lvivska oblast", h("2022-03-01 10:00")), "active"] == 1
    assert s.loc[("Lvivska oblast", h("2022-03-01 11:00")), "active"] == 0


def test_inactive_region_is_all_zero():
    s = build_series(_alerts())
    quiet = s[s["region"] == "Odeska oblast"]
    assert quiet["active"].sum() == 0
    assert quiet["starts"].sum() == 0


def test_timestamps_are_utc_aware():
    s = build_series(_alerts())
    assert str(s["ts"].dt.tz) == "UTC"
    assert set(s["active"].unique()) <= {0, 1}
