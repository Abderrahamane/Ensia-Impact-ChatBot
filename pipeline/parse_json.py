"""
Phase 1: Parse & clean result.json from the Telegram export.
Produces: data/processed/messages.json  (clean knowledge base)
          data/processed/stats.json      (summary for inspection)

Expected folder layout (adjust RAW_DIR / FILES_DIR if needed):
  raw/
    result.json
    chats/
      chat_1/
        files/
        photos/
"""

import json
import os
from pathlib import Path
from datetime import datetime


# PATHS  (edit these to match the project structure)

BASE_DIR    = Path(__file__).resolve().parent.parent   # project root
RAW_DIR     = BASE_DIR / "data" / "raw"
JSON_FILE   = RAW_DIR / "result.json"
FILES_DIR   = RAW_DIR / "chats" / "chat_1" / "files"
OUTPUT_DIR  = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# HELPERS

def flatten_text(text_field) -> str:
    """
    Telegram's 'text' can be:
      - a plain string   → return as-is
      - a list of mixed  → join string parts + link texts
    """
    if isinstance(text_field, str):
        return text_field.strip()
    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                # entity types: plain, link, bold, italic, mention, hashtag, code, …
                parts.append(item.get("text", "").strip())
        return " ".join(p for p in parts if p)
    return ""


def extract_links(text_entities: list) -> list[str]:
    """Pull all URLs from text_entities."""
    return [
        e["text"]
        for e in (text_entities or [])
        if e.get("type") == "link" and e.get("text", "").startswith("http")
    ]


def resolve_file(relative_path: str) -> str | None:
    """Return absolute path if the file exists on disk, else None."""
    if not relative_path:
        return None
    abs_path = RAW_DIR / relative_path
    return str(abs_path) if abs_path.exists() else None


def parse_date(date_str: str) -> str:
    """Normalise to ISO-8601 date string (date only, no time)."""
    try:
        return datetime.fromisoformat(date_str).strftime("%Y-%m-%d")
    except Exception:
        return date_str


def reaction_count(reactions: list) -> int:
    """Sum all reaction counts (proxy for 'importance')."""
    return sum(r.get("count", 0) for r in (reactions or []))



# MAIN PARSER

def parse_messages(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_messages = data["chats"]["list"][0]["messages"]

    cleaned = []
    skipped = 0

    for msg in raw_messages:

        # ── Skip service events (joins, leaves, pins, …) ──
        if msg.get("type") != "message":
            skipped += 1
            continue

        # ── Skip stickers (no useful text) ──
        if msg.get("media_type") == "sticker":
            skipped += 1
            continue

        text      = flatten_text(msg.get("text", ""))
        links     = extract_links(msg.get("text_entities", []))
        has_photo = "photo" in msg
        has_file  = "file" in msg

        #  Determine content type
        if has_file:
            content_type = "file"
        elif has_photo:
            content_type = "photo"
        elif links:
            content_type = "link"
        else:
            content_type = "text"

        #  File / photo paths
        file_path  = resolve_file(msg.get("file", ""))
        photo_path = resolve_file(msg.get("photo", ""))

        #  Skip truly empty messages
        if not text and not file_path and not photo_path and not links:
            skipped += 1
            continue

        record = {
            "id":           msg["id"],
            "date":         parse_date(msg.get("date", "")),
            "from":         msg.get("from", ""),
            "content_type": content_type,
            "text":         text,
            "links":        links,
            "file_path":    file_path,
            "file_name":    msg.get("file_name", ""),
            "photo_path":   photo_path,
            "reply_to":     msg.get("reply_to_message_id"),
            "reactions":    reaction_count(msg.get("reactions", [])),
            # raw text is what goes into the vector store
            "raw_content":  build_raw_content(text, links, msg.get("file_name", ""), photo_path),
        }
        cleaned.append(record)

    return cleaned, skipped


def build_raw_content(text: str, links: list, file_name: str, photo_path: str | None) -> str:
    """
    Combine all text signals into one string for embedding.
    This is what the RAG system will index.
    """
    parts = []
    if text:
        parts.append(text)
    if links:
        parts.append("Links: " + " | ".join(links))
    if file_name:
        parts.append(f"[Attached file: {file_name}]")
    if photo_path:
        parts.append("[Contains image]")
    return "\n".join(parts)



# STATS REPORT

def build_stats(messages: list[dict], skipped: int) -> dict:
    from collections import Counter
    types    = Counter(m["content_type"] for m in messages)
    senders  = Counter(m["from"] for m in messages)
    top_msgs = sorted(messages, key=lambda m: m["reactions"], reverse=True)[:10]

    return {
        "total_kept":    len(messages),
        "total_skipped": skipped,
        "by_type":       dict(types),
        "top_senders":   dict(senders.most_common(10)),
        "top_reacted":   [
            {"id": m["id"], "date": m["date"], "reactions": m["reactions"],
             "preview": m["text"][:120]}
            for m in top_msgs
        ],
    }



# ENTRY POINT

def main():
    print(f"  Reading {JSON_FILE} …")
    messages, skipped = parse_messages(JSON_FILE)

    stats = build_stats(messages, skipped)

    #  Save cleaned messages
    out_messages = OUTPUT_DIR / "messages.json"
    with open(out_messages, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(messages)} messages → {out_messages}")

    #  Save stats
    out_stats = OUTPUT_DIR / "stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Saved stats → {out_stats}")
    print()
    print("── Summary ──────────────────────────────")
    print(f"  Kept:    {stats['total_kept']}")
    print(f"  Skipped: {stats['total_skipped']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Top sender: {list(stats['top_senders'].items())[0]}")


if __name__ == "__main__":
    main()