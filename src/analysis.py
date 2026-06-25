"""Descriptive time-series analysis: temporal patterns, regional summaries,
duration statistics, national intensity, mass-attack anomalies, and
neighbour propagation lead/lag.

All functions take the hourly long series (and/or raw alerts) and return tidy
frames ready for plotting in the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import NEIGHBORS, REGIONS


# --- local-time helpers -----------------------------------------------------
def add_local_time(series: pd.DataFrame) -> pd.DataFrame:
    """Add Kyiv-local hour / day-of-week / month / date columns."""
    out = series.copy()
    local = out["ts"].dt.tz_convert(config.DISPLAY_TZ)
    out["hour"] = local.dt.hour
    out["dow"] = local.dt.dayofweek          # 0=Mon
    out["month"] = local.dt.strftime("%Y-%m")
    out["date"] = local.dt.date
    return out


def _filter_regions(series: pd.DataFrame, regions: list[str] | None) -> pd.DataFrame:
    if regions:
        return series[series["region"].isin(regions)]
    return series


# --- temporal patterns ------------------------------------------------------
def hourly_pattern(series: pd.DataFrame, regions: list[str] | None = None) -> pd.DataFrame:
    """Share of hours under alert by Kyiv-local hour of day."""
    s = add_local_time(_filter_regions(series, regions))
    return (
        s.groupby("hour")["active"].mean()
        .rename("alert_rate").reset_index()
    )


def dow_pattern(series: pd.DataFrame, regions: list[str] | None = None) -> pd.DataFrame:
    """Share of hours under alert by day of week."""
    s = add_local_time(_filter_regions(series, regions))
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out = s.groupby("dow")["active"].mean().rename("alert_rate").reset_index()
    out["day"] = out["dow"].map(dict(enumerate(names)))
    return out


def monthly_counts(series: pd.DataFrame, regions: list[str] | None = None) -> pd.DataFrame:
    """Number of alerts started per month (trend over the war)."""
    s = add_local_time(_filter_regions(series, regions))
    return (
        s.groupby("month")["starts"].sum()
        .rename("alerts").reset_index()
    )


def hour_dow_heatmap(series: pd.DataFrame, regions: list[str] | None = None) -> pd.DataFrame:
    """2-D pattern: alert rate by (day-of-week x hour). Returns long frame."""
    s = add_local_time(_filter_regions(series, regions))
    return (
        s.groupby(["dow", "hour"])["active"].mean()
        .rename("alert_rate").reset_index()
    )


# --- regional summary -------------------------------------------------------
def region_summary(series: pd.DataFrame, alerts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-region totals: #alerts, hours under alert, share of time, avg duration."""
    g = series.groupby("region")
    out = pd.DataFrame({
        "alerts": g["starts"].sum(),
        "active_hours": g["active"].sum(),
        "alert_rate": g["active"].mean(),
    })
    if alerts is not None:
        dur = (alerts["end"] - alerts["start"]).dt.total_seconds() / 60.0
        out["avg_duration_min"] = alerts.assign(dur=dur).groupby("region")["dur"].mean()
    return out.reindex(REGIONS).reset_index().rename(columns={"index": "region"})


def duration_stats(alerts: pd.DataFrame) -> pd.DataFrame:
    """Per-alert duration in minutes (for distribution plots)."""
    dur = (alerts["end"] - alerts["start"]).dt.total_seconds() / 60.0
    return pd.DataFrame({"region": alerts["region"].values, "duration_min": dur.values})


# --- national intensity & mass-attack anomalies -----------------------------
def national_intensity(series: pd.DataFrame) -> pd.DataFrame:
    """Per hour: how many regions are simultaneously under alert."""
    return (
        series.groupby("ts")["active"].sum()
        .rename("regions_active").reset_index()
    )


def mass_attack_episodes(
    series: pd.DataFrame, min_regions: int = 18, gap_h: int = 3
) -> pd.DataFrame:
    """Discrete nationwide-attack episodes: stretches where >= ``min_regions``
    oblasts are simultaneously under alert (gaps <= ``gap_h`` merged into one
    episode). Returns start, end, duration_h, peak_regions — far more
    interpretable than per-hour flags.
    """
    inten = national_intensity(series).sort_values("ts")
    vals = inten["regions_active"].to_numpy()
    idx = inten["ts"].to_numpy()
    gap = np.timedelta64(gap_h, "h")
    over = vals >= min_regions

    eps, start_i, last_i, peak = [], None, None, 0
    for i in range(len(vals)):
        if over[i]:
            if start_i is None:
                start_i, peak = i, vals[i]
            last_i = i
            peak = max(peak, vals[i])
        elif start_i is not None and (idx[i] - idx[last_i]) > gap:
            eps.append((idx[start_i], idx[last_i], int(peak)))
            start_i, peak = None, 0
    if start_i is not None:
        eps.append((idx[start_i], idx[last_i], int(peak)))

    df = pd.DataFrame(eps, columns=["start", "end", "peak_regions"])
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    df["duration_h"] = ((df["end"] - df["start"]) / pd.Timedelta(hours=1) + 1).astype(int)
    return df


def detect_mass_attacks(
    series: pd.DataFrame, window_hours: int = 24 * 14, z_thresh: float = 3.0
) -> pd.DataFrame:
    """Flag hours where simultaneous nationwide activity spikes far above its
    rolling baseline (signature of a mass missile/drone attack).

    Uses a rolling mean/std baseline and a z-score; returns flagged hours.
    """
    inten = national_intensity(series).set_index("ts")
    roll = inten["regions_active"].rolling(window_hours, min_periods=48)
    mu, sd = roll.mean(), roll.std()
    z = (inten["regions_active"] - mu) / sd.replace(0, np.nan)
    inten["baseline"] = mu
    inten["z"] = z
    inten["is_anomaly"] = (z >= z_thresh) & (inten["regions_active"] >= 10)
    return inten.reset_index()


# --- neighbour propagation (lead/lag) ---------------------------------------
def propagation_lead_lag(series: pd.DataFrame, max_lag_h: int = 6) -> pd.DataFrame:
    """For each ordered neighbour pair (A -> B), estimate how often a NEW alert
    in A is followed by a new alert in B within ``max_lag_h`` hours.

    Quantifies how threats cascade across adjacent oblasts (early-warning value).
    """
    starts_wide = (
        series.pivot(index="ts", columns="region", values="starts")
        .fillna(0).astype(int).clip(upper=1)   # 1 if a new alert started this hour
    )
    n_hours = len(starts_wide)
    rows = []
    for a in REGIONS:
        a_events = starts_wide[a]
        n_a = int(a_events.sum())
        if n_a == 0:
            continue
        a_idx = a_events.to_numpy().astype(bool)
        for b in sorted(NEIGHBORS[a]):
            # smallest lag (1..max_lag_h h) at which B started after an A start
            b_events = starts_wide[b].to_numpy().astype(bool)
            min_lag = np.full(len(a_idx), np.inf)
            for lag in range(max_lag_h, 0, -1):
                shifted = np.r_[np.zeros(lag, dtype=bool), b_events[:-lag]]
                min_lag[a_idx & shifted] = lag
            leads = min_lag[np.isfinite(min_lag)]
            hits = int(len(leads))
            follow_rate = hits / n_a
            lead_h = float(np.median(leads)) if hits else float("nan")
            # Baseline: chance B starts in *any* random max_lag_h-hour window.
            p_b = b_events.sum() / n_hours
            base = 1 - (1 - p_b) ** max_lag_h
            rows.append({
                "from": a, "to": b, "n_from": n_a,
                "followed": hits, "follow_rate": follow_rate,
                "base_rate": base, "lift": follow_rate / base if base else np.nan,
                "lead_h": lead_h,
            })
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def neighbor_correlation(series: pd.DataFrame) -> pd.DataFrame:
    """Correlation of hourly 'active' between every pair of regions (long frame)."""
    wide = series.pivot(index="ts", columns="region", values="active")
    corr = wide.corr()
    return corr.reset_index().melt(id_vars="region", var_name="region_b", value_name="corr")


if __name__ == "__main__":
    from .transform import load_series

    s = load_series()
    print("hourly peak:", hourly_pattern(s).sort_values("alert_rate").tail(3).to_string(index=False))
    print("\nmass-attack hours flagged:",
          int(detect_mass_attacks(s)["is_anomaly"].sum()))
    print("\ntop propagation pairs:")
    print(propagation_lead_lag(s).head(8).to_string(index=False))
