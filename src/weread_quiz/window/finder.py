"""定位微信读书小程序窗口，找不到精确标题时才按微信窗口面积回退。"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from ..errors import WindowNotFound
from ..models import Rect

_WECHAT_TITLE_KEYWORDS = ("微信", "WeChat")

# 宿主窗口标题的长度上限。超过这个长度的一定是浏览器标签页，不是宿主窗口。
_MAX_TITLE_LEN = 12


@dataclass(frozen=True)
class WindowInfo:
    """一个顶层窗口的快照。"""

    hwnd: int
    title: str
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_rect(self) -> Rect:
        return Rect(left=self.left, top=self.top, width=self.width, height=self.height)


def _is_visible(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.IsWindowVisible(hwnd))


def _get_title(hwnd: int) -> str:
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _get_rect(hwnd: int) -> Rect | None:
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        return None
    return Rect(
        left=rect.left,
        top=rect.top,
        width=rect.right - rect.left,
        height=rect.bottom - rect.top,
    )


def _callback(hwnd: int, _param) -> bool:
    if not _is_visible(hwnd):
        return True
    title = _get_title(hwnd)
    if not title:
        return True
    rect = _get_rect(hwnd)
    if rect is None or rect.width <= 0 or rect.height <= 0:
        return True
    _COLLECTED.append(
        WindowInfo(
            hwnd=hwnd,
            title=title,
            left=rect.left,
            top=rect.top,
            width=rect.width,
            height=rect.height,
        )
    )
    return True


_COLLECTED: list[WindowInfo] = []


def list_windows() -> list[WindowInfo]:
    """枚举所有可见且有标题的顶层窗口。"""
    global _COLLECTED
    _COLLECTED = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    ctypes.windll.user32.EnumWindows(enum_proc(_callback), 0)
    return list(_COLLECTED)


_MINIPROGRAM_TITLE = "微信读书"


def pick_largest_wechat(windows: list[WindowInfo]) -> WindowInfo | None:
    """优先精确标题「微信读书」的小程序窗口；否则从可用微信窗口中选面积最大的。"""
    candidates = [
        w
        for w in windows
        if w.title
        and len(w.title.strip()) <= _MAX_TITLE_LEN
        and any(k in w.title for k in _WECHAT_TITLE_KEYWORDS)
    ]
    if not candidates:
        return None
    miniprogram = [w for w in candidates if w.title.strip() == _MINIPROGRAM_TITLE]
    if miniprogram:
        return max(miniprogram, key=lambda w: w.area)
    return max(candidates, key=lambda w: w.area)


def find_wechat_window() -> WindowInfo:
    """找不到就抛 WindowNotFound。"""
    found = pick_largest_wechat(list_windows())
    if found is None:
        raise WindowNotFound(
            "未找到微信窗口。请确认微信 PC 客户端已启动、已登录，且窗口可见。"
        )
    return found
