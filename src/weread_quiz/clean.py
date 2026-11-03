"""清理指定保留期外的运行记录，支持只预览；不处理配置和凭据。"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from .config import DATA_DIR

# 默认保留天数。
#
# 要临时改：weread-quiz clean --keep-days 7，或启动时 --keep-days 7。
CLEAN_KEEP_DAYS = 30

# 永远不碰的目录名（相对于 data/）。
#
PROTECTED = frozenset({"templates"})

# 永远不碰的文件名（相对于 data/）。
PROTECTED_FILES = frozenset({"window.json"})

# 调试工具产生的目录，保留期内也删 —— 它们不是运行记录。
_DEBUG_DIRS = ("watch", "verify")


def _looks_like_timestamp(name: str) -> bool:
    """目录名是不是时间戳（20261020-083303）。"""
    return len(name) == 15 and name[8] == "-" and name[:8].isdigit() and name[9:].isdigit()


def _is_protected(path: Path, root: Path) -> bool:
    """这个路径或其任何父目录是否受保护。"""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False

    parts = relative.parts
    if any(part in PROTECTED for part in parts):
        return True
    if parts and parts[0] in PROTECTED_FILES:
        return True
    return False


def _directory_age_days(path: Path, now: datetime) -> float:
    """目录最后修改时间距今多少天。"""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return 0.0
    return (now - mtime).total_seconds() / 86400


def plan_clean(
    data_dir: Path = DATA_DIR,
    keep_days: int | None = None,
    now: datetime | None = None,
) -> tuple[list[Path], list[str]]:
    """算出要删什么。返回 (待删目录, 说明行)。"""
    now = now or datetime.now()
    cutoff = (now - timedelta(days=CLEAN_KEEP_DAYS if keep_days is None else keep_days)).timestamp()
    targets: list[Path] = []
    notes: list[str] = []

    if not data_dir.exists():
        return targets, notes

    # 1) 运行记录：按时间戳目录判断，优先用目录 mtime。
    #
    runs_dir = data_dir / "runs"
    if runs_dir.exists():
        for group in sorted(runs_dir.iterdir()):
            if not group.is_dir() or _is_protected(group, data_dir):
                continue
            if _looks_like_timestamp(group.name):
                runs = [group]
            else:
                try:
                    runs = [d for d in group.iterdir()
                            if d.is_dir() and not _is_protected(d, data_dir)]
                except OSError:
                    continue
            for entry in runs:
                try:
                    if entry.stat().st_mtime < cutoff:
                        targets.append(entry)
                        label = entry.relative_to(data_dir).as_posix()
                        notes.append(f"{label}（{_directory_age_days(entry, now):.1f} 天前）")
                except OSError:
                    continue

    # 2) 调试目录：不分时间，一律删
    for name in _DEBUG_DIRS:
        debug = data_dir / name
        if debug.exists() and not _is_protected(debug, data_dir):
            targets.append(debug)
            notes.append(f"{name}/（调试目录）")

    return targets, notes


def clean(
    data_dir: Path = DATA_DIR,
    keep_days: int | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> tuple[int, int]:
    """执行清理。返回 (删除的文件数, 释放的字节数)。"""
    effective = CLEAN_KEEP_DAYS if keep_days is None else keep_days
    targets, notes = plan_clean(data_dir, keep_days=keep_days, now=now)
    if not targets:
        return 0, 0

    print(f"清理 data/（保留最近 {effective} 天的运行记录，保留 templates/ 和 window.json）：")
    freed = 0
    files = 0

    for target in targets:
        size = _directory_size(target)
        label = next((n for n in notes if target.name in n), target.name)
        print(f"  {label}  {_human(size)}")
        if dry_run:
            freed += size
            files += _count_files(target)
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as exc:
            print(f"  ! 删除失败 {target}: {exc}")
            continue
        freed += size
        files += _count_files(target)

    if dry_run:
        print(f"\n试运行：将会删除 {files} 个文件，释放 {_human(freed)}")
    else:
        print(f"\n已删除 {files} 个文件，释放 {_human(freed)}")
    return files, freed


def _directory_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _count_files(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
