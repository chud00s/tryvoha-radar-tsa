"""Feature engineering for the per-oblast risk forecast.

Target: y = 1 if an air-raid alert is active at ANY hour in (t, t+H], per region.
Features use only information available at or before time t (no leakage):
  - calendar (Kyiv-local hour / day-of-week / month / night / weekend)
  - current and lagged alert state for the region
  - rolling activity sums over recent windows
  - hours since the last alert
  - neighbouring-oblast activity now and recently (threats cascade across borders)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import NEIGHBORS, REGIONS

FEATURE_COLUMNS: list[str] = []  # filled by build_features()


def _calendar(ts_local: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "hour": ts_local.dt.hour,
        "dow": ts_local.dt.dayofweek,
        "month": ts_local.dt.month,
        "is_weekend": (ts_local.dt.dayofweek >= 5).astype("int8"),
        "is_night": ts_local.dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype("int8"),
    })


def build_features(series: pd.DataFrame, horizon: int = config.FORECAST_HORIZON_HOURS) -> pd.DataFrame:
    """Return a model-ready frame: region, ts, <features...>, y."""
    df = series.sort_values(["region", "ts"]).reset_index(drop=True).copy()

    # --- target: any alert active in the next `horizon` hours, per region ---
    g_active = df.groupby("region")["active"]
    future = np.zeros(len(df), dtype="float64")
    for h in range(1, horizon + 1):
        future = np.maximum(future, g_active.shift(-h).to_numpy())
    df["y"] = future  # NaN near the tail (last `horizon` hours per region)

    # --- calendar (Kyiv local) ---
    local = df["ts"].dt.tz_convert(config.DISPLAY_TZ)
    df = pd.concat([df, _calendar(local)], axis=1)

    # --- per-region lag / rolling features (past only) ---
    g_active = df.groupby("region")["active"]
    g_starts = df.groupby("region")["starts"]
    for lag in config.LAG_HOURS:
        df[f"lag_active_{lag}"] = g_active.shift(lag)
    # group BEFORE rolling so windows never leak across region boundaries
    for w in config.ROLLING_WINDOWS:
        df[f"roll_active_{w}"] = df.groupby("region")["active"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).sum())
        df[f"roll_starts_{w}"] = df.groupby("region")["starts"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).sum())
    df["active_now"] = df["active"]

    # hours since the region's last active hour
    df["_last_active_ts"] = df["ts"].where(df["active"] == 1)
    df["_last_active_ts"] = df.groupby("region")["_last_active_ts"].ffill()
    df["hours_since_active"] = (
        (df["ts"] - df["_last_active_ts"]).dt.total_seconds() / 3600.0
    ).fillna(9999.0)
    df = df.drop(columns="_last_active_ts")

    # --- neighbour activity (cross-border cascade) ---
    active_wide = df.pivot(index="ts", columns="region", values="active")
    nb_now = pd.DataFrame(index=active_wide.index)
    nb_cnt = pd.DataFrame(index=active_wide.index)
    for r in REGIONS:
        nb = sorted(NEIGHBORS[r])
        if nb:
            nb_now[r] = active_wide[nb].mean(axis=1)
            nb_cnt[r] = active_wide[nb].sum(axis=1)
        else:
            nb_now[r] = 0.0
            nb_cnt[r] = 0.0
    nb_now_long = nb_now.reset_index().melt(id_vars="ts", var_name="region", value_name="nb_active_now")
    nb_cnt_long = nb_cnt.reset_index().melt(id_vars="ts", var_name="region", value_name="nb_active_count")
    df = df.merge(nb_now_long, on=["ts", "region"], how="left").merge(
        nb_cnt_long, on=["ts", "region"], how="left"
    )
    # recent neighbour pressure (past 6h mean), per region
    df["nb_active_6h"] = df.groupby("region")["nb_active_now"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).mean())

    feature_cols = (
        ["hour", "dow", "month", "is_weekend", "is_night", "active_now",
         "hours_since_active", "nb_active_now", "nb_active_count", "nb_active_6h"]
        + [f"lag_active_{l}" for l in config.LAG_HOURS]
        + [f"roll_active_{w}" for w in config.ROLLING_WINDOWS]
        + [f"roll_starts_{w}" for w in config.ROLLING_WINDOWS]
    )
    FEATURE_COLUMNS.clear()
    FEATURE_COLUMNS.extend(feature_cols)

    keep = ["region", "ts", "y"] + feature_cols
    # Keep rows with complete FEATURES even if y is NaN (the latest `horizon`
    # hours per region) — those are exactly what we forecast "now".
    out = df[keep].dropna(subset=feature_cols).reset_index(drop=True)
    out["region"] = out["region"].astype("category")
    return out


def feature_names() -> list[str]:
    return list(FEATURE_COLUMNS)


if __name__ == "__main__":
    from .transform import load_series

    feat = build_features(load_series())
    print(f"rows: {len(feat):,}  features: {len(feature_names())}")
    print(f"positive rate (y): {feat['y'].mean():.3f}")
    print(feat.head().to_string())
