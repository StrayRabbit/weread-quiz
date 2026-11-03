"""监视微信小程序窗口，检测到答题界面就自动存图。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from weread_quiz.capture import grab_screen
from weread_quiz.dpi import init_dpi_awareness
from weread_quiz.window.finder import find_wechat_window

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/watch")

POLL_SECONDS = 1.5
MIN_BUTTON_ROWS = 2
MIN_ROW_WIDTH = 40


def cyan_row_bands(rgb: np.ndarray) -> list[tuple[int, int]]:
    """找出横向的青色按钮条，返回 [(y_top, y_bottom), ...]。"""
    r = rgb[:, :, 0].astype(int)
    g = rgb[:, :, 1].astype(int)
    b = rgb[:, :, 2].astype(int)
    cyan = (b > 100) & (g > 90) & (b - r > 30)

    row_counts = cyan.sum(axis=1)
    bands = []
    start = None
    for y, count in enumerate(row_counts):
        if count >= MIN_ROW_WIDTH and start is None:
            start = y
        elif count < MIN_ROW_WIDTH and start is not None:
            bands.append((start, y - 1))
            start = None
    if start is not None:
        bands.append((start, len(row_counts) - 1))

    # 只保留有厚度（>=10px）的条，太窄的是边框线
    return [band for band in bands if band[1] - band[0] >= 10]


def main() -> int:
    init_dpi_awareness()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"监视中，检测到答题界面会存到 {OUT_DIR}/")
    print("现在去小程序里点『自动匹配』或『邀请好友』。")
    print("按 Ctrl+C 停止。")
    print()

    last_hwnd = None
    while True:
        try:
            window = find_wechat_window()
        except Exception:  # noqa: BLE001 - 窗口还没开就继续等
            time.sleep(POLL_SECONDS)
            continue

        moved = window.hwnd != last_hwnd
        last_hwnd = window.hwnd

        image = grab_screen(window.to_rect())
        bands = cyan_row_bands(np.array(image.convert("RGB")))

        if len(bands) >= MIN_BUTTON_ROWS:
            suffix = f"_hwnd{window.hwnd}" if moved else ""
            path = OUT_DIR / f"{time.strftime('%H%M%S')}{suffix}.png"
            image.save(path)
            print(f"[{time.strftime('%H:%M:%S')}] 检测到 {len(bands)} 个按钮条 "
                  f"{bands} → {path}")
        elif moved:
            print(f"[{time.strftime('%H:%M:%S')}] 窗口变化 {window.title!r} "
                  f"{window.width}x{window.height} @({window.left},{window.top})")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n停止监视。")
        sys.exit(0)
