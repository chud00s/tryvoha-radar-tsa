"""Download and load the raw air-raid alert dataset.

Source: Vadimkin/ukrainian-air-raid-sirens-dataset (oblast-level alert intervals).
Each row is one alert with a start and end timestamp (UTC) for a region.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import requests

from . import config


def download_alerts(dataset_key: str = config.DEFAULT_DATASET, force: bool = False) -> "config.Path":
    """Download the chosen CSV to data/raw/ (cached unless ``force``)."""
    url = config.DATASET_URLS[dataset_key]
    dest = config.raw_csv_path(dataset_key)
    if dest.exists() and not force:
        print(f"[ingest] cached: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    print(f"[ingest] downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    print(f"[ingest] saved {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


# Candidate column names across dataset variants -> canonical name.
_COLUMN_ALIASES = {
    "region": {"region", "oblast", "area", "region_title", "location"},
    "start": {"start", "started_at", "start_time", "begin", "from"},
    "end": {"end", "finished_at", "end_time", "finish", "to"},
}


def _canonical_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map actual columns to canonical {region,start,end}; raise if missing."""
    lower = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canon, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[lower[alias]] = canon
                break
        else:
            raise ValueError(
                f"Could not find a '{canon}' column. Available: {list(df.columns)}"
            )
    return mapping


def load_alerts(dataset_key: str = config.DEFAULT_DATASET, force: bool = False) -> pd.DataFrame:
    """Return a tidy frame with columns: region, start, end (tz-aware UTC)."""
    path = download_alerts(dataset_key, force=force)
    df = pd.read_csv(path)
    df = df.rename(columns=_canonical_columns(df))[["region", "start", "end"]]

    df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df["region"] = df["region"].astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["region", "start"])
    # Open alerts (no end yet) -> treat as ending now-ish is risky; drop for history.
    df = df.dropna(subset=["end"])
    # Sanity: end must be >= start.
    df = df[df["end"] >= df["start"]]
    df = df.sort_values("start").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[ingest] dropped {dropped} malformed/open rows")
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n=== columns ===")
    print(list(df.columns))
    print("\n=== head ===")
    print(df.head(5).to_string())
    print("\n=== date range (UTC) ===")
    print(f"{df['start'].min()}  ->  {df['end'].max()}")
    print(f"rows: {len(df):,}")
    print("\n=== regions ===")
    regions = sorted(df["region"].unique())
    print(f"count: {len(regions)}")
    for r in regions:
        print(f"  - {r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download & inspect air-raid alert data")
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=list(config.DATASET_URLS))
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args(argv)

    df = load_alerts(args.dataset, force=args.force)
    summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
