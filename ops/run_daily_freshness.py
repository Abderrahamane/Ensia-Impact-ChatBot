"""Daily data freshness pipeline runner.

Runs in order:
1) parse_json
2) extract_files
3) reindex_incremental
4) build_structured_tables
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _run_step(cmd: list[str], dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily data freshness pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    py = sys.executable
    steps = [
        [py, "pipeline/parse_json.py"],
        [py, "pipeline/extract_files.py"],
        [py, "pipeline/reindex_incremental.py"],
        [py, "pipeline/build_structured_tables.py"],
    ]

    for step in steps:
        _run_step(step, dry_run=args.dry_run)

    print("Daily freshness pipeline complete.")


if __name__ == "__main__":
    main()

