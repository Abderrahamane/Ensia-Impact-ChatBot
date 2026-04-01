"""Create daily backups of vectorstore and prune old backups.

Run:
    python ops/backup_data.py
    python ops/backup_data.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BACKUP_RETENTION_DAYS, BACKUP_VECTORSTORE_DIR, VECTORSTORE_DIR


def make_backup_name(now: datetime) -> str:
    return f"vectorstore_{now.strftime('%Y%m%d_%H%M%S')}"


def remove_old_backups(backup_dir: Path, retention_days: int, dry_run: bool) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0
    for item in backup_dir.iterdir():
        if not item.is_dir() and not item.name.endswith(".zip"):
            continue
        mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=UTC)
        if mtime < cutoff:
            deleted += 1
            if not dry_run:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
    return deleted


def run_backup(dry_run: bool = False) -> dict[str, str | int | None]:
    if not VECTORSTORE_DIR.exists():
        raise RuntimeError(f"Vectorstore directory does not exist: {VECTORSTORE_DIR}")

    BACKUP_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    base = BACKUP_VECTORSTORE_DIR / make_backup_name(now)

    archive_path = None
    if not dry_run:
        archive_path = shutil.make_archive(str(base), "zip", root_dir=str(VECTORSTORE_DIR))

    removed = remove_old_backups(BACKUP_VECTORSTORE_DIR, BACKUP_RETENTION_DAYS, dry_run)
    return {
        "archive_path": archive_path,
        "planned_archive_path": str(base) + ".zip",
        "removed": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup data/vectorstore")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions only")
    args = parser.parse_args()

    result = run_backup(dry_run=args.dry_run)
    if result.get("archive_path"):
        print(f"Backup created: {result['archive_path']}")
    else:
        print(f"Would archive {VECTORSTORE_DIR} to {result['planned_archive_path']}")
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {result['removed']} backup(s) older than {BACKUP_RETENTION_DAYS} days")


if __name__ == "__main__":
    main()


