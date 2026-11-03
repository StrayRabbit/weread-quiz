"""按帧差把连续截图切成"每题一段"。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 倒计时区域：每 0.1 秒刷新，会掩盖真实变化
TIMER_BAND = (200, 245)

SAME_QUESTION_MAX = 1.0
NEW_QUESTION_MIN = 2.0


def frame_diff(a: Image.Image, b: Image.Image) -> float:
    """两张图的平均像素差（排除倒计时区）。"""
    arr_a = np.asarray(a.convert("RGB")).astype(np.int16)
    arr_b = np.asarray(b.convert("RGB")).astype(np.int16)

    top = TIMER_BAND[0]
    bottom = TIMER_BAND[1]
    diff = np.abs(
        np.concatenate([arr_a[:top], arr_a[bottom:]], axis=0)
        - np.concatenate([arr_b[:top], arr_b[bottom:]], axis=0)
    )
    return float(diff.mean())


def load_image(path: Path) -> Image.Image | None:
    """读图并强制解码。抓图过程中文件可能只写了一半。"""
    try:
        image = Image.open(path)
        image.load()
        return image
    except (OSError, ValueError):
        return None


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "data/watch")
    files = sorted(directory.glob("*.png"))
    if not files:
        print(f"{directory} 里没有截图")
        return 1

    print(f"共 {len(files)} 张截图，逐帧算差异（排除倒计时区 y{TIMER_BAND[0]}-{TIMER_BAND[1]}）\n")

    transitions = []
    previous = None
    previous_name = ""
    same_count = 0
    unreadable = 0

    for path in files:
        image = load_image(path)
        if image is None:
            unreadable += 1
            continue
        if previous is not None:
            delta = frame_diff(previous, image)
            if SAME_QUESTION_MAX <= delta < NEW_QUESTION_MIN:
                transitions.append((previous_name, path.name, delta, "模糊"))
            elif delta >= NEW_QUESTION_MIN:
                transitions.append((previous_name, path.name, delta, "换题"))
            else:
                same_count += 1
        previous = image
        previous_name = path.name

    print(f"同一题（帧差<1）：{same_count} 次")
    print(f"换题（帧差>=2）：{sum(1 for t in transitions if t[3] == '换题')} 次")
    fuzzy = [t for t in transitions if t[3] == "模糊"]
    print(f"模糊区（1<=帧差<2）：{len(fuzzy)} 次")
    print(f"读图失败（半截截图）：{unreadable} 张")
    print()

    if transitions:
        print("所有状态切换点：")
        for start, end, delta, kind in transitions:
            print(f"  {start} → {end}  帧差={delta:6.2f}  {kind}")
    else:
        print("全程没有检测到换题 —— 检查截图目录是否正确")

    return 0


if __name__ == "__main__":
    sys.exit(main())
