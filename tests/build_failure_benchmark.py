"""Build failure benchmark cases from bot logs and wrong-answer feedback snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "logs" / "bot.log"
FEEDBACK_PATH = ROOT_DIR / "data" / "processed" / "feedback.jsonl"
OUT_PATH = ROOT_DIR / "tests" / "failure_benchmark_cases.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def build_cases(max_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 1) Parse bot failures / degradation hints from logs.
    for rec in _read_jsonl(LOG_PATH):
        event = rec.get("event")
        meta = rec.get("meta") or {}
        query = str(meta.get("query") or "").strip()
        if event == "bot_query_failed" and query:
            cases.append(
                {
                    "id": f"fail_log_{len(cases)+1:03d}",
                    "language": "unknown",
                    "query": query,
                    "expected_any": [],
                    "expected_intent": "ensia_query",
                    "origin": "bot_query_failed",
                }
            )
        if event == "bot_query_ok" and meta.get("mode") == "needs-clarification" and query:
            cases.append(
                {
                    "id": f"clarify_log_{len(cases)+1:03d}",
                    "language": "unknown",
                    "query": query,
                    "expected_any": [],
                    "expected_intent": "ensia_query",
                    "origin": "needs_clarification",
                }
            )

    # 2) Parse wrong-answer button feedback snapshots.
    for rec in _read_jsonl(FEEDBACK_PATH):
        if rec.get("text") != "wrong_answer_button":
            continue
        snap = rec.get("snapshot") or {}
        query = str(snap.get("query") or "").strip()
        if not query:
            continue
        cases.append(
            {
                "id": f"wrong_fb_{len(cases)+1:03d}",
                "language": "unknown",
                "query": query,
                "expected_any": [],
                "expected_intent": "ensia_query",
                "origin": "wrong_answer_button",
            }
        )

    # Deduplicate by query text while preserving order.
    dedup: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in cases:
        q = c["query"].strip().lower()
        if q in seen:
            continue
        seen.add(q)
        dedup.append(c)

    return dedup[:max_cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build failure benchmark cases from logs")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_cases(max_cases=args.max_cases)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(payload)} failure benchmark cases -> {out_path}")


if __name__ == "__main__":
    main()

