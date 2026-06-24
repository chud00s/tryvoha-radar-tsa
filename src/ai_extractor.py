"""Turn raw Telegram OSINT text into STRUCTURED threat events.

This is the agentic-AI core of the project: free-form Ukrainian monitoring
messages ("Група ударних дронів курсом на Дніпропетровщину") become typed,
geolocated events that we can map and align with the official alert series.

Two interchangeable backends:
  * LLM (Claude, structured tool-use)  -> used when ANTHROPIC_API_KEY is set
  * deterministic rules (keywords)     -> always-available offline fallback

Both emit the same schema, so the dashboard + cache work with or without a key.
Purpose is DEFENSIVE situational awareness / early warning only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv optional
    pass

from . import config, geo

# Canonical vocabularies
EVENT_TYPES = ["загроза", "пуск", "рух", "влучання", "збиття", "відбій", "інше"]
WEAPONS = ["дрон/шахед", "ракета", "КАБ", "авіація", "невідомо"]

LLM_MODEL = "claude-haiku-4-5"


# --------------------------------------------------------------------------- #
# Telegram message loading
# --------------------------------------------------------------------------- #
def load_sample(path: "config.Path | str | None" = None) -> list[dict]:
    """Load the committed sample messages (simple [{id,date,text}] list)."""
    p = config.DATA_RAW / "sample_messages.json" if path is None else path
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_telegram_export(path: "config.Path | str") -> list[dict]:
    """Parse a Telegram Desktop JSON export (result.json) into {id,date,text}.

    Handles the official export where ``text`` may be a list of fragments
    (strings and {"type","text"} dicts).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("messages", data) if isinstance(data, dict) else data
    out = []
    for m in raw:
        if m.get("type") not in (None, "message"):
            continue
        text = m.get("text", "")
        if isinstance(text, list):
            text = "".join(t if isinstance(t, str) else t.get("text", "") for t in text)
        text = text.strip()
        if not text:
            continue
        out.append({"id": m.get("id"), "date": m.get("date"), "text": text})
    return out


def fetch_live_telethon(limit_per_channel: int = 100) -> list[dict]:
    """STRETCH: pull recent messages from public channels via Telethon.

    Requires TELEGRAM_API_ID / TELEGRAM_API_HASH / TELEGRAM_CHANNELS in env.
    Returns the same {id,date,text} shape. Imported lazily so the core pipeline
    has no hard Telethon dependency.
    """
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    channels = [c.strip() for c in os.getenv("TELEGRAM_CHANNELS", "").split(",") if c.strip()]
    if not (api_id and api_hash and channels):
        raise RuntimeError("Set TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS in .env")

    from telethon.sync import TelegramClient  # lazy import

    out: list[dict] = []
    with TelegramClient("tryvoha_radar", int(api_id), api_hash) as client:
        for ch in channels:
            for msg in client.iter_messages(ch, limit=limit_per_channel):
                if msg.message:
                    out.append({
                        "id": f"{ch}:{msg.id}",
                        "date": msg.date.isoformat(),
                        "text": msg.message,
                    })
    return out


# --------------------------------------------------------------------------- #
# Rule-based extractor (offline fallback)
# --------------------------------------------------------------------------- #
# Order matters: outcomes (intercept/impact) before causes (launch/threat).
_RULES = [
    ("збиття", ["збито", "збили", "знищ", "ппо прац", "робота ппо", "спрацювала ппо", "перехоп"]),
    ("влучання", ["влучан", "влучил", "приліт", "вибух", "уражен", "попадан", "пожеж"]),
    ("відбій", ["відбій"]),
    ("пуск", ["пуск", "запуск", "зліт", "злетів", "застосуванн", "балістик"]),
    ("рух", ["курс", "напрямку", "рух", "пролет", "тримают", "змінили курс", "у бік", "на підльоті"]),
    ("загроза", ["загроза", "увага", "небезпек", "укритт", "обстріл", "ціл"]),
]
_WEAPON_RULES = [
    ("дрон/шахед", ["шахед", "shahed", "бпла", "дрон", "герань", "мопед"]),
    ("ракета", ["ракет", "калібр", "кинджал", "іскандер", "балістик", "крилат", "міг-31"]),
    ("КАБ", ["каб", "кабів", "авіабомб", "керован"]),
    ("авіація", ["авіаці", "літак", "тактичн", "бомбардув"]),
]


def _classify_rules(text: str) -> tuple[str, str]:
    t = text.lower()
    event = "інше"
    for label, kws in _RULES:
        if any(k in t for k in kws):
            event = label
            break
    weapon = "невідомо"
    for label, kws in _WEAPON_RULES:
        if any(k in t for k in kws):
            weapon = label
            break
    return event, weapon


def _extract_rules(messages: list[dict]) -> list[dict]:
    events = []
    for m in messages:
        event_type, weapon = _classify_rules(m["text"])
        region, alias = geo.find_region_mention(m["text"])
        events.append(_make_event(m, event_type, weapon, alias, region, 0.5, "rules"))
    return events


# --------------------------------------------------------------------------- #
# LLM extractor (Claude, structured tool-use)
# --------------------------------------------------------------------------- #
_TOOL = {
    "name": "record_events",
    "description": "Record structured air-threat events extracted from monitoring messages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string"},
                        "event_type": {"type": "string", "enum": EVENT_TYPES},
                        "weapon": {"type": "string", "enum": WEAPONS},
                        "location": {"type": "string", "description": "City/oblast mentioned, '' if none"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["message_id", "event_type", "weapon", "location", "confidence"],
                },
            }
        },
        "required": ["events"],
    },
}

_SYSTEM = (
    "Ти аналітик з ОSINT-моніторингу повітряних загроз в Україні. Завдання — суто "
    "ОБОРОННЕ: ситуаційна обізнаність і раннє попередження населення. Тобі дають "
    "пронумеровані повідомлення з моніторингових Telegram-каналів. Для КОЖНОГО "
    "повідомлення визнач тип події (загроза/пуск/рух/влучання/збиття/відбій/інше), "
    "тип засобу, згадану локацію (місто чи область) і впевненість 0..1. "
    "Якщо локації немає — порожній рядок. Поверни результат ВИКЛЮЧНО через інструмент record_events."
)


def _extract_llm(messages: list[dict], batch_size: int = 15) -> list[dict]:
    import anthropic  # lazy import

    client = anthropic.Anthropic()
    by_id = {str(m["id"]): m for m in messages}
    events: list[dict] = []

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        listing = "\n".join(f'[{m["id"]}] {m["text"]}' for m in batch)
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "record_events"},
            messages=[{"role": "user", "content": f"Повідомлення:\n{listing}"}],
        )
        payload = next((b.input for b in resp.content if b.type == "tool_use"), {"events": []})
        for e in payload.get("events", []):
            m = by_id.get(str(e.get("message_id")))
            if m is None:
                continue
            region, alias = geo.find_region_mention(e.get("location") or m["text"])
            events.append(_make_event(
                m, e.get("event_type", "інше"), e.get("weapon", "невідомо"),
                e.get("location") or alias, region, float(e.get("confidence", 0.7)), "llm",
            ))
    return events


# --------------------------------------------------------------------------- #
# Shared helpers + dispatcher
# --------------------------------------------------------------------------- #
def _make_event(m, event_type, weapon, location_raw, region, confidence, method) -> dict:
    coords = geo.region_coords(region) if region else None
    return {
        "id": m["id"],
        "timestamp": pd.to_datetime(m["date"], utc=True),
        "event_type": event_type if event_type in EVENT_TYPES else "інше",
        "weapon": weapon if weapon in WEAPONS else "невідомо",
        "location_raw": location_raw or "",
        "region": region,
        "lat": coords[0] if coords else None,
        "lon": coords[1] if coords else None,
        "confidence": confidence,
        "method": method,
        "text": m["text"],
    }


def extract_events(messages: list[dict], prefer_llm: bool = True) -> pd.DataFrame:
    """Extract events, using the LLM when a key is present, else rules."""
    use_llm = prefer_llm and bool(os.getenv("ANTHROPIC_API_KEY"))
    if use_llm:
        try:
            print(f"[ai_extractor] using LLM backend ({LLM_MODEL})")
            events = _extract_llm(messages)
        except Exception as exc:  # graceful fallback
            print(f"[ai_extractor] LLM failed ({exc!r}); falling back to rules")
            events = _extract_rules(messages)
    else:
        print("[ai_extractor] no ANTHROPIC_API_KEY -> using rule-based fallback")
        events = _extract_rules(messages)
    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)


def save_events(df: pd.DataFrame) -> None:
    df.to_parquet(config.EVENTS_PARQUET, index=False)
    print(f"[ai_extractor] saved {config.EVENTS_PARQUET} ({len(df)} events)")


def load_events() -> pd.DataFrame:
    if config.EVENTS_PARQUET.exists():
        df = pd.read_parquet(config.EVENTS_PARQUET)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    return pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract structured events from Telegram OSINT text")
    ap.add_argument("--input", help="Telegram export JSON; default = bundled sample")
    ap.add_argument("--live", action="store_true", help="fetch live via Telethon (needs creds)")
    ap.add_argument("--rules", action="store_true", help="force rule-based backend")
    args = ap.parse_args(argv)

    if args.live:
        messages = fetch_live_telethon()
    elif args.input:
        messages = parse_telegram_export(args.input)
    else:
        messages = load_sample()
    print(f"[ai_extractor] loaded {len(messages)} messages")

    df = extract_events(messages, prefer_llm=not args.rules)
    save_events(df)
    print("\n=== event-type counts ===")
    print(df["event_type"].value_counts().to_string())
    print("\n=== sample ===")
    cols = ["timestamp", "event_type", "weapon", "region", "confidence", "method"]
    print(df[cols].head(12).to_string(index=False))
    geocoded = df["region"].notna().mean()
    print(f"\ngeocoded: {geocoded:.0%} of events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
