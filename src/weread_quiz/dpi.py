"""DPI 感知与坐标换算。"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from .models import Rect

# PER_MONITOR_AWARE_V2 = -4
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
_BASE_DPI = 96.0

# 截图的标定宽度：全部像素判据（按钮带、题号带、题干区、答题卡片）
# 都是按这个宽度标定的，对应 100% 显示缩放下 430px 宽的小程序窗口。
#
# 抓屏时会把窗口归一化到这个宽度（见 capture.grab_screen），所以
# 下游一律按标定坐标工作，系统缩放 125%/150%/200% 都不用改常量。
BASELINE_WIDTH = 430


class DpiError(RuntimeError):
    """DPI 感知设置失败。"""


def init_dpi_awareness() -> float:
    """必须在导入 mss / pyautogui 之前调用，且整个进程只调一次。"""
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        )
    except AttributeError:
        # Windows 8.1 及更早没有这个 API，退回 system DPI aware
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception as exc:  # noqa: BLE001
            raise DpiError(f"无法设置 DPI 感知: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise DpiError(f"无法设置 DPI 感知: {exc}") from exc

    return get_scale_for_window(None)


def get_scale_for_window(hwnd: int | None) -> float:
    """查询缩放比。hwnd 为 None 时用主显示器。"""
    user32 = ctypes.windll.user32
    try:
        if hwnd is None:
            hwnd = user32.GetDesktopWindow()
        dpi = user32.GetDpiForWindow(wintypes.HWND(hwnd))
    except AttributeError:
        # 老系统没有 GetDpiForWindow，退回 GetDeviceCaps
        dpi = ctypes.windll.gdi32.GetDeviceCaps(0, 88)  # LOGPIXELSX
    if not dpi:
        dpi = int(_BASE_DPI)
    return dpi / _BASE_DPI


def box_center_to_screen(region: Rect, box: Rect) -> tuple[float, float]:
    """截图（已归一化到标定宽度）里的框 → 屏幕物理坐标。

    进程是 PER_MONITOR_AWARE_V2，Windows 不做 DPI 虚拟化：
    GetWindowRect、mss 抓屏、SetCursorPos 用的都是**物理像素**。
    截图被归一化到 BASELINE_WIDTH 之后，框在屏幕上的位置是：

        窗口原点 + 框位置 × (物理窗口宽 / 标定宽)

    早先这里除的是系统缩放比 —— 那等于把物理坐标又缩回去，缩放非
    100% 时点击点会落到窗口外面（150% 时整窗偏到窗口左侧 1/3 处，
    点的根本不是按钮）。
    """
    factor = region.width / BASELINE_WIDTH if region.width > 0 else 1.0
    return (
        region.left + (box.left + box.width / 2) * factor,
        region.top + (box.top + box.height / 2) * factor,
    )
