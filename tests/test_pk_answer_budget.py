"""PK 识别预算、确认时刻锚点及本地预读保护回归。"""

from __future__ import annotations

import time

import pytest

from weread_quiz.pk import runner as pk
from weread_quiz.errors import AnswerUnavailable
from weread_quiz.models import Rect
from weread_quiz.ocr_core import OcrRead
from weread_quiz.vision import VisionQuestion


# ---- 测试替身 ----------------------------------------------------------


class FakeBox(Rect):
    """带 center_y 的按钮矩形。"""

    @property
    def center_y(self) -> int:
        return self.top + self.height // 2


def _boxes(count: int) -> list[FakeBox]:
    return [FakeBox(left=60, top=400 + 80 * i, width=310, height=44)
            for i in range(count)]


def _blank():
    from PIL import Image
    return Image.new("RGB", (430, 788), (20, 35, 70))


class FakeVision:
    """只实现答题阶段用到的那几个接口。"""

    def __init__(self, index: int | None = 1):
        self._index = index
        self.consulted = False
        self.deadlines: list[float | None] = []

    def answer_from_ocr_read(self, read: OcrRead, *, timeout=None, deadline=None):
        self.consulted = True
        self.deadlines.append(deadline)
        if deadline is not None and time.monotonic() >= deadline:
            raise pk.RecognitionTimeout("deadline_exceeded")
        if self._index is None:
            return None
        return VisionQuestion(
            stem=read.stem,
            options=list(read.options),
            answer_index=self._index,
            screen="question",
            local_read=True,
        )

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        raise AssertionError("本用例不该走图片路径")

    @property
    def usage(self):
        return None


@pytest.fixture
def patched(monkeypatch):
    """把抓屏、按钮定位、点击、身份校验全部换成确定性的替身。"""
    state = {"clicks": [], "grabs": 0}

    def fake_grab(region):
        state["grabs"] += 1
        return _blank()

    def fake_find(image):
        return _boxes(2)

    def fake_pick(boxes, index):
        if not (0 <= index < len(boxes)):
            raise pk.WqError("下标越界")
        return boxes[index]

    def fake_click(box, region, pause=0.0):
        state["clicks"].append(box.center_y)

    monkeypatch.setattr(pk, "grab_screen", fake_grab)
    monkeypatch.setattr(pk, "find_option_buttons", fake_find)
    monkeypatch.setattr(pk, "pick_button", fake_pick)
    monkeypatch.setattr(pk, "click_box", fake_click)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    return state


def _ocr_read() -> OcrRead:
    return OcrRead(
        stem="驳___走蛇",
        options=["龙", "凤", "虎"],
        confidences=[0.99, 0.99, 0.99],
        button_count=3,
        has_progress=True,
        progress_text="第1题·共10题",
    )


# ---- 用例 1：deadline 锚点 ----------------------------------------------


def test_deadline_anchors_to_confirmed_at_not_stale_capture(monkeypatch):
    """等待阶段确认答题页的时刻，比采集时刻晚很多时，用确认时刻锚定。"""
    now = 1000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    # 采集在 1000.0，但确认答题页直到 1009.0 才发生。
    context = pk.PKCaptureContext()
    context.captured_at = 1000.0
    context.confirmed_at = 1009.0

    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)

    vision = FakeVision(1)
    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        vision=vision,
        image=_blank(),
        question=None,
        captured_at=1000.0,
        ocr_read=_ocr_read(),
        capture_context=context,
        verbose=False,
    )

    assert vision.consulted, "文本路径根本没有被调用 —— deadline 一进来就过期了"
    assert result["guessed"] is False, "本地读数被丢掉、变成蒙题了"
    assert result["answer"] == 1


def test_deadline_uses_capture_when_no_confirmation(monkeypatch):
    """没有确认时刻（纯图片路径）时，仍然锚在采集时刻上。"""
    now = 2000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)

    vision = FakeVision(1)
    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        vision=vision,
        image=_blank(),
        question=None,
        captured_at=now,
        ocr_read=_ocr_read(),
        verbose=False,
    )

    assert vision.consulted, "文本路径没有被调用"
    # 传给模型的是**夹紧后**的 deadline：min(锚点+6.0, now+3.0)。
    # 关键是它**没有过期**（> now）—— 过期就是用例 2/3 要修的病。
    assert vision.deadlines and vision.deadlines[0] > now, (
        f"deadline 已过期（{vision.deadlines[0]} <= {now}）")
    assert result["guessed"] is False


# ---- 用例 2：本地读数对 deadline 的豁免 --------------------------------


def test_local_preread_is_exempt_from_deadline(monkeypatch):
    """完整的本地读数存在时，deadline 过期**不该**直接判死。"""
    now = 3000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)

    class Vision(FakeVision):
        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            # 即使 deadline 已过期，也照常给答案 —— 这正是要保证的。
            self.consulted = True
            self.deadlines.append(deadline)
            return VisionQuestion(
                stem=read.stem,
                options=list(read.options),
                answer_index=1,
                screen="question",
                local_read=True,
            )

    vision = Vision(1)

    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        vision=vision,
        image=_blank(),
        question=None,
        captured_at=2900.0,
        ocr_read=_ocr_read(),
        verbose=False,
    )

    assert vision.consulted, "本地读数在 deadline 过期时根本没被送去求答案"
    assert result["guessed"] is False, "读了又丢掉，改成蒙题了"
    assert result["answer"] == 1


# ---- 用例 3：deadline 过期也不跳过本地读数 ----------------------------


def test_preread_path_still_consults_model_when_deadline_passed(monkeypatch):
    """即使 deadline 已过，有完整读数也该先问一次模型再蒙。"""
    now = 4000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)

    seen = []

    class Vision(FakeVision):
        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            seen.append(read.stem)
            self.consulted = True
            return None  # 模型没给出合法下标

        def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
            # 图片路径也失败 —— 于是最后蒙题。
            raise pk.AnswerUnavailable("invalid_response")

    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        vision=Vision(None),
        image=_blank(),
        question=None,
        captured_at=3900.0,
        ocr_read=_ocr_read(),
        verbose=False,
    )

    assert seen == ["驳___走蛇"], "本地读数被 deadline 直接掐掉了"
    assert result["guessed"] is True, "模型没答出下标时应该蒙题"
