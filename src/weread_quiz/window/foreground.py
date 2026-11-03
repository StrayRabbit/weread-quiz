"""把小程序窗口置顶：抓屏按它在屏幕上的位置来，窗口被盖住就抓错画面。

用户常常是**先打开小程序答题页、再双击启动器** —— 启动器弹出的控制台
窗口会把小程序盖住，程序抓到的就是控制台而不是答题页。这里在开跑前把
小程序窗口置顶（TOPMOST），跑完再还原。

置顶用 `SWP_NOACTIVATE`：窗口抬到最上层，但**不抢键盘焦点** —— 焦点留在
控制台上，用户还能看着题号、题干、答案一行行打出来。

控制台如果正好压着小程序窗口，置顶之后它反而被盖住了 —— 所以顺手把控制台
挪到旁边，跑完再挪回原位。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from ..models import Rect

# Win32 常量
SW_RESTORE = 9
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008

# RedrawWindow 的标志
RDW_INVALIDATE = 0x0001
RDW_ALLCHILDREN = 0x0080
RDW_UPDATENOW = 0x0100
RDW_FRAME = 0x0400

# 抬起来之后给应用重绘的时间（秒）。
_SETTLE_SECONDS = 0.25

# 虚拟屏幕的 GetSystemMetrics 索引
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


def is_topmost(hwnd: int) -> bool:
    """窗口当前是不是置顶状态。"""
    user32 = ctypes.windll.user32
    return bool(user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)


def set_topmost(hwnd: int) -> bool:
    """置顶窗口，并把最小化的窗口恢复出来。成功返回 True。"""
    user32 = ctypes.windll.user32
    if not hwnd:
        return False
    if user32.IsIconic(hwnd):
        # 最小化的窗口在屏幕上没有画面 —— 抓屏只会抓到它底下的东西。
        user32.ShowWindow(hwnd, SW_RESTORE)
    # 句柄一律按 64 位指针传：HWND_TOPMOST 是 (HWND)-1，
    # 用普通 int 传会被 ctypes 截成 32 位（高位清零），SetWindowPos 直接失败。
    return bool(user32.SetWindowPos(
        ctypes.c_void_p(hwnd), ctypes.c_void_p(HWND_TOPMOST), 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
    ))


def clear_topmost(hwnd: int) -> bool:
    """取消置顶（跑完还原窗口原来的状态）。成功返回 True。"""
    user32 = ctypes.windll.user32
    if not hwnd:
        return False
    return bool(user32.SetWindowPos(
        ctypes.c_void_p(hwnd), ctypes.c_void_p(HWND_NOTOPMOST), 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
    ))


def _dwm_flush() -> None:
    """等 DWM 把当前帧合成完 —— 紧接着抓屏才不会抓到还没画出来的空白。"""
    try:
        ctypes.windll.dwmapi.DwmFlush()
    except (AttributeError, OSError):
        pass


def redraw(hwnd: int) -> None:
    """强制窗口重绘，并等合成器把这一帧刷出来。"""
    if not hwnd:
        return
    ctypes.windll.user32.RedrawWindow(
        ctypes.c_void_p(hwnd), None, None,
        RDW_INVALIDATE | RDW_UPDATENOW | RDW_FRAME | RDW_ALLCHILDREN,
    )
    _dwm_flush()


def bring_to_front(hwnd: int) -> bool:
    """把小程序窗口抬到前面、激活，并置顶。

    **激活是关键**：小程序是 GPU 合成窗口，焦点丢了就停止渲染 —— 只用
    HWND_TOPMOST 抬层而不激活，窗口会留在白屏上（内容根本没画出来）。
    激活才让它把界面重绘。

    置顶是第二道保险：用户点控制台看输出时，小程序不会被压回去。

    返回是否成功激活（激活失败也仍然置顶，照样能盖住控制台）。
    """
    user32 = ctypes.windll.user32
    if not hwnd:
        return False
    if user32.IsIconic(hwnd):
        # 最小化的窗口在屏幕上没有画面 —— 抓屏只会抓到它底下的东西。
        user32.ShowWindow(hwnd, SW_RESTORE)
    activated = bool(user32.SetForegroundWindow(ctypes.c_void_p(hwnd)))
    user32.BringWindowToTop(ctypes.c_void_p(hwnd))
    set_topmost(hwnd)
    redraw(hwnd)
    # 给应用一点时间把内容画出来。
    time.sleep(_SETTLE_SECONDS)
    return activated


def refresh_window(hwnd: int) -> None:
    """白屏时的急救：重新激活 + 强制重绘。"""
    if not hwnd:
        return
    ctypes.windll.user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
    redraw(hwnd)
    time.sleep(0.15)


def console_hwnd() -> int:
    """本进程的控制台窗口句柄（没有控制台时是 0）。"""
    return int(ctypes.windll.kernel32.GetConsoleWindow() or 0)


def window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """窗口矩形 (left, top, width, height)。"""
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(ctypes.c_void_p(hwnd),
                                              ctypes.byref(rect)):
        return None
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def _virtual_screen() -> tuple[int, int, int, int]:
    """虚拟屏幕 (left, top, width, height) —— 多显示器时是它们的并集。"""
    user32 = ctypes.windll.user32
    return (
        user32.GetSystemMetrics(_SM_XVIRTUALSCREEN),
        user32.GetSystemMetrics(_SM_YVIRTUALSCREEN),
        user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN),
        user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN),
    )


def _move(hwnd: int, left: int, top: int) -> bool:
    """移动窗口（尺寸不动）。"""
    return bool(ctypes.windll.user32.SetWindowPos(
        ctypes.c_void_p(hwnd), ctypes.c_void_p(0), int(left), int(top), 0, 0,
        SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
    ))


def rects_overlap(rect: tuple[int, int, int, int], avoid: Rect) -> bool:
    """两个矩形是否相交。rect 是 (left, top, width, height)。"""
    left, top, width, height = rect
    return not (left >= avoid.left + avoid.width
                or left + width <= avoid.left
                or top >= avoid.top + avoid.height
                or top + height <= avoid.top)


def aside_left(avoid: Rect, width: int, screen: tuple[int, int, int, int]) -> int:
    """控制台挪到哪个横坐标才不压着 avoid（screen 是虚拟屏幕矩形）。"""
    screen_left, _screen_top, screen_width, _screen_height = screen
    for candidate in (avoid.left - width - 8, avoid.left + avoid.width + 8):
        if screen_left <= candidate and candidate + width <= screen_left + screen_width:
            return candidate
    return screen_left  # 实在没地方放就贴屏幕左边


def move_console_aside(avoid: Rect) -> tuple[bool, tuple[int, int, int, int] | None]:
    """控制台压着小程序窗口时，把它挪到旁边。

    返回 (是否挪过, 控制台原来的位置)。没压着就什么都不做。
    """
    hwnd = console_hwnd()
    if not hwnd:
        return False, None
    current = window_rect(hwnd)
    if current is None:
        return False, None
    _left, top, width, height = current
    if not rects_overlap(current, avoid):
        return False, None  # 没压着

    screen = _virtual_screen()
    new_top = min(max(top, screen[1]),
                  max(screen[1], screen[1] + screen[3] - height))
    if not _move(hwnd, aside_left(avoid, width, screen), new_top):
        return False, None
    return True, current


def restore_console(rect: tuple[int, int, int, int] | None) -> None:
    """把控制台挪回原位。"""
    if rect is None:
        return
    hwnd = console_hwnd()
    if hwnd:
        _move(hwnd, rect[0], rect[1])
