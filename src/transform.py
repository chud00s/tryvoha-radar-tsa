"""Turn alert intervals (region, start, end) into a per-oblast HOURLY time series.

Output (long format, one row per region-hour):
    region, ts (UTC, hour), active (0/1), n_active (#alerts overlapping the hour),
    starts (#alerts that began in the hour).

This regular, gap-free hourly grid is the substrate for pattern analysis and the
forecasting model.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config
from .geo import REGIONS
from .ingest import load_alerts

# Guard against corrupt mega-intervals (a few weeks is already implausible).
_MAX_HOURS_PER_ALERT = 24 * 21


def build_series(alerts: pd.DataFrame) -> pd.DataFrame:
    """Build the gap-free hourly long series from alert intervals."""
    df = alerts.copy()
    df = df[df["region"].isin(REGIONS)]

    # Work in tz-naive UTC for fast datetime64 arithmetic; re-localize at the end.
    start_naive = df["start"].dt.tz_localize(None)
    end_naive = df["end"].dt.tz_localize(None)
    fs = start_naive.dt.floor("h")
    fe = end_naive.dt.floor("h")
    n_hours = ((fe - fs) // pd.Timedelta(hours=1)).astype("int64") + 1

    clipped = int((n_hours > _MAX_HOURS_PER_ALERT).sum())
    if clipped:
        print(f"[transform] clipping {clipped} implausibly long alerts to {_MAX_HOURS_PER_ALERT}h")
    n_hours = n_hours.clip(upper=_MAX_HOURS_PER_ALERT)

    # Vectorised expansion of each interval into the hours it touches.
    region_rep = np.repeat(df["region"].to_numpy(), n_hours.to_numpy())
    fs_rep = np.repeat(fs.to_numpy(), n_hours.to_numpy())
    offsets = np.concatenate([np.arange(n) for n in n_hours.to_numpy()])
    hours = fs_rep + offsets * np.timedelta64(1, "h")

    exploded = pd.DataFrame({"region": region_rep, "ts": hours})
    # n_active = number of distinct alerts overlapping each region-hour.
    n_active = (
        exploded.groupby(["region", "ts"]).size().rename("n_active").reset_index()
    )

    # starts = number of alerts that began in each region-hour.
    starts = (
        df.assign(ts=fs)
        .groupby(["region", "ts"]).size().rename("starts").reset_index()
    )

    # Build the full region x hour grid (gap-free), tz-naive for the merge.
    hour_index = pd.date_range(start=fs.min(), end=fe.max(), freq="h")
    grid = pd.MultiIndex.from_product(
        [REGIONS, hour_index], names=["region", "ts"]
    ).to_frame(index=False)

    out = (
        grid.merge(n_active, on=["region", "ts"], how="left")
        .merge(starts, on=["region", "ts"], how="left")
    )
    out[["n_active", "starts"]] = out[["n_active", "starts"]].fillna(0).astype("int32")
    out["active"] = (out["n_active"] > 0).astype("int8")
    out = out.sort_values(["region", "ts"]).reset_index(drop=True)
    # Re-localize to UTC for downstream consumers.
    out["ts"] = out["ts"].dt.tz_localize("UTC")
    return out


def save_series(series: pd.DataFrame) -> None:
    series.to_parquet(config.SERIES_PARQUET, index=False)
    print(f"[transform] saved {config.SERIES_PARQUET} ({len(series):,} rows)")


def load_series(rebuild: bool = False, dataset_key: str = config.DEFAULT_DATASET) -> pd.DataFrame:
    """Load cached hourly series, building it from raw alerts if needed."""
    if config.SERIES_PARQUET.exists() and not rebuild:
        df = pd.read_parquet(config.SERIES_PARQUET)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    series = build_series(load_alerts(dataset_key))
    save_series(series)
    return series


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build hourly alert time series")
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=list(config.DATASET_URLS))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    series = load_series(rebuild=args.rebuild, dataset_key=args.dataset)
    print("\n=== hourly series ===")
    print(series.head().to_string())
    print(f"\nregions: {series['region'].nunique()}  hours/region: {series['ts'].nunique():,}")
    print(f"overall active rate: {series['active'].mean():.3f}")
    top = (
        series.groupby("region")["active"].mean().sort_values(ascending=False).head(5)
    )
    print("\ntop-5 regions by share of hours under alert:")
    for r, v in top.items():
        print(f"  {r:28s} {v:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
