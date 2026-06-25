"""Live OSINT collector (optional, deploy-time).

Listens to public Telegram monitoring channels in real time via Telethon, runs
each new message through the same extractor (LLM if ANTHROPIC_API_KEY is set,
else rule-based), and appends structured events to data/processed/events.parquet
— which the dashboard's OSINT tab reads (15s cache) and auto-refreshes.

Run:
    python -m src.collector              # live stream (needs Telegram creds)
    python -m src.collector --backfill 200   # seed from recent history, then exit

Required env (set at deploy time; never committed):
    TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS=chan1,chan2
Optional:
    ANTHROPIC_API_KEY  -> richer LLM extraction instead of rules
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from . import config
from .ai_extractor import extract_events, fetch_live_telethon, load_events, save_events

# Keep a rolling window so the live store stays small and relevant.
RETENTION_HOURS = 48


def _channels() -> list[str]:
    return [c.strip() for c in os.getenv("TELEGRAM_CHANNELS", "").split(",") if c.strip()]


def _require_creds() -> tuple[str, str, list[str]]:
    api_id, api_hash, channels = os.getenv("TELEGRAM_API_ID"), os.getenv("TELEGRAM_API_HASH"), _channels()
    if not (api_id and api_hash and channels):
        raise SystemExit("Set TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS in .env")
    return api_id, api_hash, channels


def append_events(new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new events into the store (dedup by id, drop stale, persist)."""
    if new_df.empty:
        return load_events()
    cur = load_events()
    merged = pd.concat([cur, new_df], ignore_index=True) if not cur.empty else new_df
    merged = merged.drop_duplicates(subset="id", keep="last").sort_values("timestamp")
    cutoff = merged["timestamp"].max() - pd.Timedelta(hours=RETENTION_HOURS)
    merged = merged[merged["timestamp"] >= cutoff].reset_index(drop=True)
    save_events(merged)
    return merged


def backfill(limit_per_channel: int = 200) -> None:
    """Seed the store from recent channel history (one-shot)."""
    msgs = fetch_live_telethon(limit_per_channel=limit_per_channel)
    print(f"[collector] backfill: {len(msgs)} messages")
    append_events(extract_events(msgs))


def run_live() -> None:
    api_id, api_hash, channels = _require_creds()
    from telethon import TelegramClient, events  # lazy import

    client = TelegramClient("tryvoha_radar", int(api_id), api_hash)

    @client.on(events.NewMessage(chats=channels))
    async def _handler(event):  # noqa: ANN001
        text = (event.message.message or "").strip()
        if not text:
            return
        msg = {"id": f"{event.chat_id}:{event.id}",
               "date": event.message.date.isoformat(), "text": text}
        df = extract_events([msg])
        append_events(df)
        e = df.iloc[0]
        print(f"[collector] + {e['event_type']:9} {e['weapon']:11} {e['region']}")

    print(f"[collector] listening to: {', '.join(channels)}")
    client.start()
    client.run_until_disconnected()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live Telegram OSINT collector")
    ap.add_argument("--backfill", type=int, metavar="N",
                    help="seed from last N messages per channel, then exit")
    args = ap.parse_args(argv)
    if args.backfill:
        backfill(args.backfill)
    else:
        run_live()
    return 0


if __name__ == "__main__":
    sys.exit(main())
