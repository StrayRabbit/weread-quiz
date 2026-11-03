"""Esc 中止的单元测试。

钩子回调的判定逻辑在这里直接喂结构体验证 —— 真机上的按键投递无法在测试里
复现（需要真实桌面会话），所以只测「回调收到什么 → 是否触发」。
"""

from __future__ import annotations

import ctypes

import pytest

from weread_quiz import abort


@pytest.fixture(autouse=True)
def _clear_abort():
    abort.reset()
    yield
    abort.reset()


def _struct(vk_code: int):
    """返回 (结构体, 地址) —— 结构体必须活到回调读完，否则是野指针。"""
    info = abort.KbdllHookStruct()
    info.vkCode = vk_code
    return info, ctypes.addressof(info)


def test_esc_keydown_triggers_abort():
    info, address = _struct(0x1B)
    abort._hook(0, abort._WM_KEYDOWN, address)
    assert abort.requested()


def test_other_key_does_not_trigger():
    info, address = _struct(0x41)
    abort._hook(0, abort._WM_KEYDOWN, address)
    assert not abort.requested()


def test_keyup_does_not_trigger():
    info, address = _struct(0x1B)
    abort._hook(0, 0x0101, address)  # WM_KEYUP
    assert not abort.requested()


def test_check_raises_after_esc():
    info, address = _struct(0x1B)
    abort._hook(0, abort._WM_KEYDOWN, address)
    with pytest.raises(abort.Aborted):
        abort.check()


def test_check_is_silent_before_esc():
    abort.check()  # 不抛异常


def test_click_box_checks_abort(monkeypatch):
    """按了 Esc 就不能再点 —— 点击前必须拦下。"""
    from weread_quiz import click

    monkeypatch.setattr(abort, "_event", abort.threading.Event())
    abort._event.set()
    clicked = []
    monkeypatch.setattr(click.pyautogui, "click", lambda *a, **k: clicked.append(a))
    with pytest.raises(abort.Aborted):
        click.click_box(None, None, 1.0)
    assert not clicked
