"""DPI 感知与坐标换算。"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from .models import Rect

# PER_MONITOR_AWARE_V2 = -4
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
_BASE_DPI = 96.0


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


def box_center_to_screen(region: Rect, box: Rect, scale: float) -> tuple[float, float]:
    """region 内某个矩形框的中心点 → 点击坐标。"""
    if scale <= 0:
        raise ValueError(f"scale 必须为正数，收到 {scale}")
    return (
        (region.left + box.left + box.width / 2) / scale,
        (region.top + box.top + box.height / 2) / scale,
    )
