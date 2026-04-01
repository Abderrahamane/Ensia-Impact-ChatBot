"""Run local deployment stack: web chat + Telegram bot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    py = sys.executable
    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen([py, "web/app.py"], cwd=ROOT_DIR))
        procs.append(subprocess.Popen([py, "bot/telegram_bot.py"], cwd=ROOT_DIR))
        print("Started web + telegram processes. Press Ctrl+C to stop both.")
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()

