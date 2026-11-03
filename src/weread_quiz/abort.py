"""紧急中止：Esc 键监听。

pyautogui 的 FAILSAFE（把鼠标推到屏幕左上角）保留作兜底，Esc 是主入口 ——
「按 Esc」比「把鼠标猛地推到左上角」好理解得多。

用低级键盘钩子（WH_KEYBOARD_LL）而不是全局热键：热键会把 Esc 从系统里吃掉
（微信自己就收不到这个 Esc 了），低级钩子只是旁听，不抢键。
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

_VK_ESCAPE = 0x1B
_WH_KEYBOARD_LL = 13
_WM_KEYDOWN = 0x0100
_WM_SYSKEYDOWN = 0x0104


class KbdllHookStruct(ctypes.Structure):
    """低级键盘钩子的回调参数。"""

    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


_HookProc = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


class Aborted(Exception):
    """用户按下 Esc 要求中止。"""


_event = threading.Event()
_lock = threading.Lock()
_installed = threading.Event()
_state = {"started": False, "hooked": False}
# 回调必须一直有引用，否则会被 GC 掉，钩子随之失效。
_proc_ref = None


def _hook(n_code: int, w_param: int, l_param: int) -> int:
    """低级键盘钩子回调：wParam 是消息类型，lParam 才是按键结构体指针。"""
    try:
        if w_param in (_WM_KEYDOWN, _WM_SYSKEYDOWN):
            info = ctypes.cast(l_param, ctypes.POINTER(KbdllHookStruct)).contents
            if info.vkCode == _VK_ESCAPE:
                _event.set()
    except Exception:  # noqa: BLE001 - 回调里抛异常会被 ctypes 吞掉，只能自己兜
        pass
    return 0  # CallNextHookEx：继续传给下一个钩子，不拦键


def _watch(user32, kernel32) -> None:
    """装钩子并跑消息循环；钩子回调就发生在消息循环里。"""
    global _proc_ref
    user32.SetWindowsHookExW.argtypes = [
        ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE

    _proc_ref = _HookProc(_hook)
    hmod = kernel32.GetModuleHandleW(None)
    hook = user32.SetWindowsHookExW(_WH_KEYBOARD_LL, _proc_ref, hmod, 0)
    if not hook:
        # 钩子装不上（被安全软件拦、或环境不允许）——角落 failsafe 仍兜底。
        return
    _state["hooked"] = True
    _installed.set()
    try:
        msg = wintypes.MSG()
        user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG), wintypes.HWND,
            wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int
        # 阻塞在消息循环里 —— 钩子回调靠它被调用。
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
    finally:
        user32.UnhookWindowsHookEx(hook)
        _state["hooked"] = False


def start(timeout: float = 0.5) -> bool:
    """启动监听线程（幂等），返回低级钩子是否装上。

    非 Windows 或钩子装不上时返回 False —— 角落 failsafe 仍然兜底。
    """
    with _lock:
        if not _state["started"]:
            _state["started"] = True
            try:
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            except (AttributeError, OSError):
                return False
            threading.Thread(target=_watch, args=(user32, kernel32),
                             name="esc-watcher", daemon=True).start()
    # 等钩子装上再返回：调用方要据此告诉用户 Esc 能不能用。
    return _installed.wait(timeout)


def hooked() -> bool:
    """低级钩子是否已装上（装不上时只剩角落 failsafe 可用）。"""
    return _state["hooked"]


def requested() -> bool:
    """用户是否已经按下 Esc。"""
    return _event.is_set()


def check() -> None:
    """已按 Esc 就抛 Aborted，调用方据此停下。"""
    if _event.is_set():
        raise Aborted("用户按下 Esc")


def reset() -> None:
    """清掉中止标记（仅测试用）。"""
    _event.clear()
