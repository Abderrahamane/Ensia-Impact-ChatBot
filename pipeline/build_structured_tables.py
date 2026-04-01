"""Build structured partnership and event tables from processed messages."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PROCESSED_DIR, STRUCTURED_EVENTS_PATH, STRUCTURED_PARTNERSHIPS_PATH

PARTNER_PATTERNS = [
    r"\bsonatrach\b",
    r"\bsonelgaz\b",
    r"\bmobilis\b",
    r"\bdjezzy\b",
    r"\bhuawei\b",
    r"\bbomare\b",
    r"\bcpa\b",
    r"\balgerie telecom\b",
    r"\basal\b",
]

EVENT_HINTS = ["summit", "conference", "hackathon", "event", "registration", "inscription", "forum", "expo"]


def _load_messages() -> list[dict]:
    path = PROCESSED_DIR / "messages.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_partnerships(messages: list[dict]) -> dict:
    partners: dict[str, dict] = {}
    for msg in messages:
        text = (msg.get("raw_content") or msg.get("text") or "").lower()
        for pattern in PARTNER_PATTERNS:
            match = re.search(pattern, text)
            if not match:
                continue
            name = match.group(0).title()
            prev = partners.get(name)
            record = {
                "name": name,
                "first_seen": msg.get("date", ""),
                "last_seen": msg.get("date", ""),
                "message_ids": [msg.get("id")],
            }
            if prev:
                record["first_seen"] = min(prev["first_seen"], record["first_seen"]) if prev["first_seen"] and record["first_seen"] else prev["first_seen"] or record["first_seen"]
                record["last_seen"] = max(prev["last_seen"], record["last_seen"]) if prev["last_seen"] and record["last_seen"] else prev["last_seen"] or record["last_seen"]
                record["message_ids"] = list({*prev["message_ids"], *record["message_ids"]})
            partners[name] = record
    return {"partners": sorted(partners.values(), key=lambda x: x["name"])}


def build_events(messages: list[dict]) -> dict:
    events: list[dict] = []
    for msg in messages:
        text = (msg.get("raw_content") or msg.get("text") or "")
        text_l = text.lower()
        if not any(h in text_l for h in EVENT_HINTS):
            continue
        year_match = re.search(r"\b(20\d{2})\b", text)
        events.append(
            {
                "date": msg.get("date", ""),
                "year": year_match.group(1) if year_match else "",
                "preview": re.sub(r"\s+", " ", text).strip()[:240],
                "message_id": msg.get("id"),
            }
        )
    events.sort(key=lambda x: (x.get("date", ""), x.get("message_id") or 0), reverse=True)
    dedup: dict[str, dict] = {}
    for e in events:
        key = f"{e.get('date','')}::{e.get('preview','')[:80]}"
        if key not in dedup:
            dedup[key] = e
    return {"events": list(dedup.values())[:200]}


def main() -> None:
    msgs = _load_messages()
    if not msgs:
        print("No processed messages found. Run parse/extract first.")
        return

    STRUCTURED_PARTNERSHIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    partnerships = build_partnerships(msgs)
    events = build_events(msgs)

    STRUCTURED_PARTNERSHIPS_PATH.write_text(json.dumps(partnerships, ensure_ascii=False, indent=2), encoding="utf-8")
    STRUCTURED_EVENTS_PATH.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved partnerships table -> {STRUCTURED_PARTNERSHIPS_PATH}")
    print(f"Saved events timeline -> {STRUCTURED_EVENTS_PATH}")


if __name__ == "__main__":
    main()



