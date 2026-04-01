"""Fail CI/local checks if evaluation metrics regress versus baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT_DIR / "tests" / "eval_baseline.json"
CURRENT_PATH = ROOT_DIR / "tests" / "eval_rag_report.json"

# Maximum allowed drop (current must be >= baseline - tolerance)
DEFAULT_TOLERANCES = {
    "intent_accuracy": 0.03,
    "hit@1": 0.03,
    "hit@3": 0.03,
    "hit@5": 0.02,
    "groundedness": 0.03,
    "citation_usefulness": 0.03,
}


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression gate for eval metrics")
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--current", default=str(CURRENT_PATH))
    args = parser.parse_args()

    baseline = _load(Path(args.baseline))
    current = _load(Path(args.current))

    b = (baseline.get("summary") or {}).get("overall") or {}
    c = (current.get("summary") or {}).get("overall") or {}

    failed: list[str] = []
    for metric, tol in DEFAULT_TOLERANCES.items():
        bv = float(b.get(metric, 0.0))
        cv = float(c.get(metric, 0.0))
        if cv < (bv - tol):
            failed.append(f"{metric}: baseline={bv:.4f}, current={cv:.4f}, tol={tol:.4f}")

    if failed:
        print("Regression gate FAILED:")
        for msg in failed:
            print(f"- {msg}")
        raise SystemExit(1)

    print("Regression gate passed.")


if __name__ == "__main__":
    main()

