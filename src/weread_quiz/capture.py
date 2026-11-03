"""屏幕截图与运行记录。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image

from .config import RUNS_DIR
from .dpi import BASELINE_WIDTH
from .models import Rect

_UNSAFE = re.compile(r"[^0-9A-Za-z一-鿿_-]")


def _slug(text: str) -> str:
    cleaned = _UNSAFE.sub("_", text).strip("_")
    return cleaned or "shot"


class RunRecorder:
    """一次运行的截图与日志落盘器。"""

    def __init__(self, base_dir: Path = RUNS_DIR, timestamp: str | None = None,
                 mode: str | None = None, *, echo: bool = False) -> None:
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.mode = mode or ""
        # 有 mode 时按 mode/时间戳 分目录，截图一眼能看出是哪条流程。
        parent = Path(base_dir) / self.mode if self.mode else Path(base_dir)
        self.run_dir = parent / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        # 调试模式：日志同步打到终端（见类文档）。
        self.echo = echo

    def next_index(self) -> int:
        self._counter += 1
        return self._counter

    def shot(self, label: str, image: Image.Image | None = None) -> Path:
        """保存一张截图。传入 image 则用它，否则现场抓屏。"""
        index = self.next_index()
        path = self.run_dir / f"{index:03d}_{_slug(label)}.png"
        if image is None:
            image = grab_screen(None)
        image.save(path)
        return path

    def log(self, message: str) -> None:
        """追加一行日志，带时间戳和子命令前缀。"""
        stamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{self.mode}] " if self.mode else ""
        with (self.run_dir / "run.log").open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {prefix}{message}\n")
        if self.echo:
            try:
                print(f"[{stamp}] {prefix}{message}")
            except Exception:  # noqa: BLE001 - 控制台关闭不能影响答题
                pass


def normalize_to_baseline(image: Image.Image) -> Image.Image:
    """把窗口截图归一化到标定宽度。

    所有像素判据（按钮带、题号带、题干区、答题卡片）都是按
    BASELINE_WIDTH 标定的 —— 归一化之后下游一律按标定坐标工作，
    系统显示缩放 125%/150%/200% 都不用改常量。
    """
    if image.width <= 0 or image.width == BASELINE_WIDTH:
        return image
    height = max(1, round(image.height * BASELINE_WIDTH / image.width))
    return image.resize((BASELINE_WIDTH, height), Image.LANCZOS)


def grab_screen(region: Rect | None = None) -> Image.Image:
    """抓屏幕。

    带 region 时抓的是小程序窗口，并**归一化到标定宽度**
    （BASELINE_WIDTH）。整屏抓取（region=None）不归一化 —— 那是给人看的
    留证截图。
    """
    with mss.MSS() as sct:
        monitor = sct.monitors[1] if region is None else {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        raw = sct.grab(monitor)
        image = Image.frombytes("RGB", raw.size, raw.rgb)
    if region is not None:
        image = normalize_to_baseline(image)
    return image
