"""Run full quality pipeline: build failures, evaluate, and enforce regression gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> None:
    py = sys.executable
    _run([py, "tests/build_failure_benchmark.py", "--max-cases", "100"])
    _run([py, "tests/eval_rag.py", "--top-k", "5"])
    _run([py, "tests/eval_regression_gate.py"])
    print("Quality pipeline passed.")


if __name__ == "__main__":
    main()

