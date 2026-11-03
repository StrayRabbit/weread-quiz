"""PK 首题预读、等待阶段题号缓存与末题文本路径回归。"""

from __future__ import annotations

import pytest

from weread_quiz import pk
from weread_quiz.models import Rect
from weread_quiz.ocr_core import OcrRead
from weread_quiz.vision import VisionQuestion


class FakeBox(Rect):
    @property
    def center_y(self) -> int:
        return self.top + self.height // 2


def _boxes(count: int) -> list[FakeBox]:
    return [FakeBox(left=60, top=400 + 80 * i, width=310, height=44)
            for i in range(count)]


def _blank():
    from PIL import Image
    return Image.new("RGB", (430, 788), (20, 35, 70))


class _StubVision:
    usage = None
    last_question_usage = None

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        raise AssertionError("本用例不该调模型")


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_real_run_scenario_q1_is_not_guessed(monkeypatch):
    """首题使用本地预读求答案，不因等待耗时直接蒙答。"""
    clock = FakeClock()
    monkeypatch.setattr(pk.time, "monotonic", clock)

    # 等待阶段：10.1 秒。题号行 OCR 每次 1.5 秒。
    ocr_calls = {"n": 0}

    class SlowOcr:
        def read_progress_text(self, image):
            ocr_calls["n"] += 1
            clock.advance(1.5)
            return (None, "")

        def read_progress(self, image):
            return None

    monkeypatch.setattr(pk, "looks_like_question_template", lambda image: False)
    monkeypatch.setattr(pk, "has_rendered_stem", lambda image: False)
    monkeypatch.setattr(pk, "progress_cache_mask", lambda image: "mask")
    monkeypatch.setattr(pk, "progress_cache_changed", lambda *a: False)
    monkeypatch.setattr(pk, "_poll_sleep", lambda totals: clock.advance(0.1))

    # 等待阶段跑 10.1 秒的等效轮数，然后停。
    grabs = {"n": 0}

    def counting_grab(region):
        grabs["n"] += 1
        if grabs["n"] > 120:
            raise _Stop()
        return _blank()

    monkeypatch.setattr(pk, "grab_screen", counting_grab)

    with pytest.raises(_Stop):
        pk.wait_for_question(
            (0, 0, 430, 788), _StubVision(), timeout=30.0, verbose=False,
            ocr=SlowOcr(), capture_context=pk.PKCaptureContext(),
        )

    wait_elapsed = clock.now
    ocr_time = 1.5 * ocr_calls["n"]

    duty = ocr_time / wait_elapsed
    assert duty <= 0.4, (
        f"等待阶段 {wait_elapsed:.1f}s 里题号行 OCR 占了 {duty:.0%}"
        f"（{ocr_calls['n']} 次）—— 还是会把第 1 题的倒计时吃掉"
    )

    base = 1000.0

    class Vision:
        usage = None
        last_question_usage = None

        def __init__(self):
            self.consulted = False

        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            self.consulted = True
            return VisionQuestion(
                stem=read.stem,
                options=list(read.options),
                answer_index=0,
                screen="question",
                local_read=True,
            )

    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    # 答题阶段会再抓一次屏做 fresh 校验 —— 这一抓不能再抛 _Stop。
    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())

    context = pk.PKCaptureContext()
    context.captured_at = base - wait_elapsed
    context.confirmed_at = base

    read = OcrRead(
        stem="驳___走蛇",
        options=["龙", "凤", "虎"],
        confidences=[0.99, 0.99, 0.99],
        button_count=3,
        has_progress=True,
        progress_text="第1题·共10题",
    )

    vision = Vision()
    clock.now = base
    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        scale=1.0,
        vision=vision,
        image=_blank(),
        question=None,
        captured_at=base - wait_elapsed,   # 采集在 wait_elapsed 秒前
        ocr_read=read,
        capture_context=context,
        verbose=False,
    )

    assert vision.consulted, "本地读数被丢掉，一次模型都没问"
    assert result["guessed"] is False, "第 1 题还是蒙的"
    assert result["answer"] == 0, f"答案没落下来：{result}"


class _Stop(Exception):
    pass
