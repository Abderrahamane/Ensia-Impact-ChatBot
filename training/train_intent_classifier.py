"""Build tiny intent classifier seed data used by IntentRouter centroids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CLASSES = {
    "smalltalk": [
        "hello",
        "hi",
        "how are you",
        "what can you do",
        "thanks",
        "bonjour",
        "salut",
        "مرحبا",
    ],
    "ensia_query": [
        "what are current school partnerships",
        "give me internship opportunities in ensia",
        "search resources about datacamp",
        "what are ensia impact events",
        "ما هي شراكات المدرسة",
    ],
    "admin_op": [
        "/health",
        "/stats",
        "/backup_now",
        "check bot health",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create intent classifier seed JSON")
    parser.add_argument("--output", default="assets/intent_classifier_seed.json")
    parser.add_argument("--merge", default="", help="Optional JSON file with extra class examples")
    args = parser.parse_args()

    classes = {k: list(v) for k, v in DEFAULT_CLASSES.items()}
    if args.merge:
        extra = json.loads(Path(args.merge).read_text(encoding="utf-8"))
        for label, examples in extra.get("classes", {}).items():
            classes.setdefault(label, [])
            classes[label].extend([str(x) for x in examples if str(x).strip()])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved intent seed data to {out_path}")


if __name__ == "__main__":
    main()

