"""小程序窗口置顶：抓屏按窗口在屏幕上的位置来，被盖住就抓错画面。"""

from __future__ import annotations

import pytest

from weread_quiz.models import Rect
from weread_quiz.window import foreground


def test_invalid_handle_is_reported_not_raised():
    """句柄无效时返回 False —— 置顶失败不该把整局带崩。"""
    assert foreground.set_topmost(0) is False
    assert foreground.clear_topmost(0) is False


def test_roundtrip_on_a_real_window():
    """置顶 → 查询 → 还原，状态要真的变回去。"""
    from weread_quiz.window.finder import list_windows

    windows = [w for w in list_windows() if w.width > 0 and w.height > 0]
    if not windows:
        pytest.skip("这个环境里没有可见窗口")
    host = windows[0]
    original = foreground.is_topmost(host.hwnd)
    try:
        assert foreground.set_topmost(host.hwnd) is True
        assert foreground.is_topmost(host.hwnd) is True
    finally:
        # 还原成原来的状态（原来就是置顶的，就还回置顶）。
        if original:
            foreground.set_topmost(host.hwnd)
        else:
            foreground.clear_topmost(host.hwnd)
    assert foreground.is_topmost(host.hwnd) == original


def test_console_is_moved_only_when_it_covers_the_window():
    """控制台压着小程序窗口就挪开；压不着就别动它。"""
    hwnd = foreground.console_hwnd()
    if not hwnd:
        pytest.skip("这个环境里没有控制台窗口")
    original = foreground.window_rect(hwnd)
    assert original is not None
    left, top, width, height = original

    # 压着：把小程序窗口摆在控制台上。
    moved, saved = foreground.move_console_aside(
        Rect(left=left, top=top, width=width, height=height))
    try:
        assert moved is True
        assert saved == original
        current = foreground.window_rect(hwnd)
        assert current is not None
        assert current[:2] != original[:2], "控制台应该被挪开了"
    finally:
        foreground.restore_console(saved)

    assert foreground.window_rect(hwnd) == original

    # 压不着：屏幕另一头放一个窗口，控制台不该动。
    moved, saved = foreground.move_console_aside(
        Rect(left=original[0] + 100000, top=original[1] + 100000,
             width=10, height=10))
    assert moved is False
    assert saved is None
    assert foreground.window_rect(hwnd) == original


def test_overlap_and_aside_placement():
    """挪动位置的判断是纯逻辑，单独测。"""
    screen = (0, 0, 2560, 1440)
    avoid = Rect(left=1878, top=125, width=642, height=1181)

    # 压着 → 挪到左边。
    assert foreground.rects_overlap((2000, 200, 300, 200), avoid) is True
    assert foreground.aside_left(avoid, 300, screen) == 1878 - 300 - 8

    # 没压着 → 不动。
    assert foreground.rects_overlap((0, 0, 300, 200), avoid) is False

    # 左边放不下 → 挪到右边。
    wide = Rect(left=100, top=0, width=1200, height=800)
    assert foreground.aside_left(wide, 500, screen) == 100 + 1200 + 8

    # 两边都放不下 → 贴屏幕左边。
    full = Rect(left=0, top=0, width=2560, height=1440)
    assert foreground.aside_left(full, 300, screen) == 0
