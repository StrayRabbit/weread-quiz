"""空白帧：窗口还没把内容画出来时的识别与自愈。"""

from __future__ import annotations

from PIL import Image

from weread_quiz import advance
from weread_quiz.capture import is_blank_frame


def _blank():
    return Image.new("RGB", (430, 788), "white")


def _with_content():
    image = Image.new("RGB", (430, 788), "white")
    image.paste((0, 0, 0), (10, 10, 60, 40))
    return image


def test_blank_frame_detection():
    assert is_blank_frame(_blank()) is True
    assert is_blank_frame(_with_content()) is False


def test_blank_frames_trigger_window_refresh(monkeypatch):
    """连续空白帧要把窗口叫醒；空白帧本身不能被跳过（跳过会干等到超时）。"""
    frames = [_blank(), _blank(), _blank(), _with_content()]
    refreshed = []
    clicked = []

    def fake_grab(region):
        return frames.pop(0) if frames else _with_content()

    def fake_lines(ocr, image):
        if is_blank_frame(image):
            return []
        return [("自动匹配", 1.0, ((10, 10), (60, 10), (60, 30), (10, 30)))]

    monkeypatch.setattr(advance, "grab_screen", fake_grab)
    monkeypatch.setattr(advance, "_read_all_lines", fake_lines)
    monkeypatch.setattr(advance, "click_box", lambda *a, **k: clicked.append(a))

    ok = advance.advance_to_question(
        region=(0, 0, 430, 788),
        ocr=object(),
        recorder=None,
        targets=("自动匹配",),
        is_question_page=None,
        timeout=5.0,
        refresh=lambda: refreshed.append(1),
    )

    assert ok is True
    assert refreshed, "连续空白帧应该触发一次窗口刷新"
    assert clicked, "读到目标就该点它"
