"""PK 残缺预读交接、图片回退及模型失败恢复回归。"""

from __future__ import annotations

import pytest

from weread_quiz.pk import runner as pk
from weread_quiz.errors import AnswerUnavailable
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


def _partial_read() -> OcrRead:
    """**残缺**的读数：选项读出 2 个，按钮有 3 个。"""
    return OcrRead(
        stem="——文鸟迹",
        options=["龟", "鳌"],          # 只读到 2 个，实际 3 个
        confidences=[0.99, 0.99],
        button_count=3,
        has_progress=True,
        progress_text="第1题·共10题",
    )


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---- 用例 --------------------------------------------------------------


def test_partial_preread_skips_text_and_reads_image(monkeypatch):
    """残缺选项只作界面证据，不能用于文本求答案。"""
    now = 1000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    context = pk.PKCaptureContext()
    context.captured_at = now
    context.confirmed_at = now
    context.ocr_read = _partial_read()

    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())

    class Vision:
        usage = None
        last_question_usage = None

        def __init__(self):
            self.text_calls = 0
            self.image_calls = 0

        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            self.text_calls += 1
            # 只读出 2 个选项 —— 下标在错误的坐标系里。
            return VisionQuestion(
                stem=read.stem,
                options=list(read.options),
                answer_index=0,
                screen="question",
                local_read=True,
            )

        def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
            self.image_calls += 1
            # 图片路径看得见全部三个选项。
            return VisionQuestion(
                stem="——文鸟迹",
                options=["龟", "鳌", "鳖"],
                answer_index=2,
                screen="question",
            )

    vision = Vision()
    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        vision=vision,
        image=_blank(),
        question=None,
        captured_at=now,
        ocr_read=context.ocr_read,
        capture_context=context,
        verbose=False,
    )

    # 残缺选项不能可靠映射按钮，直接走图片路径，不浪费文本请求。
    assert vision.text_calls == 0, "不应拿缺失选项的预读请求文本答案"
    assert vision.image_calls == 1, "残缺预读应直接回退图片路径"
    assert result["guessed"] is False, "不该蒙 —— 图片路径给出了可用答案"
    assert result["answer"] == 2, "应该用图片路径给出的下标（鳖）"


def test_image_path_failure_does_not_abort_the_run(monkeypatch):
    """图片路径失败不该让整局中断 —— 该蒙就蒙。"""
    now = 1000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())

    clicked = []
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **k: clicked.append(box.center_y))

    class Vision:
        usage = None
        last_question_usage = None

        def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
            raise AnswerUnavailable("invalid_response")

    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        vision=Vision(),
        image=_blank(),
        question=None,
        captured_at=now,
        verbose=False,
    )

    assert result["guessed"] is True, "图片路径失败后既没蒙也没跳过 —— 抛出去了"
    assert clicked, "按钮都没点 —— 整局白白结束"


def test_both_paths_failing_still_clicks_something(monkeypatch):
    """两条路都失败时，也要点一个按钮 —— 绝不能中断整局。"""
    now = 1000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "pick_button", lambda boxes, i: boxes[i])
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())

    clicked = []
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **k: clicked.append(box.center_y))

    class Vision:
        usage = None
        last_question_usage = None

        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            return None

        def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
            raise AnswerUnavailable("invalid_response")

    context = pk.PKCaptureContext()
    context.captured_at = now
    context.confirmed_at = now

    result = pk.answer_one_question(
        region=(0, 0, 430, 788),
        vision=Vision(),
        image=_blank(),
        question=None,
        captured_at=now,
        ocr_read=_partial_read(),
        capture_context=context,
        verbose=False,
    )

    assert result["guessed"] is True, "两条路都失败时应该蒙题"
    assert clicked, "按钮都没点 —— 整局白白结束"


def test_wait_phase_keeps_only_progress_from_rejected_read(monkeypatch):
    """等待阶段独立保留题号证据，不交接被门控拒绝的选项。"""
    clock = FakeClock()
    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk, "looks_like_question_template", lambda image: True)
    monkeypatch.setattr(pk, "has_rendered_stem", lambda image: True)
    monkeypatch.setattr(pk, "progress_cache_mask", lambda image: "mask")
    monkeypatch.setattr(pk, "progress_cache_changed", lambda *a: False)
    monkeypatch.setattr(pk, "_poll_sleep", lambda totals: clock.advance(0.1))

    class RejectingOcr:
        """本地读题门控判否（选项数不符），但**带**题号行证据。"""

        def read_question_local(self, image):
            raise pk.OcrRejected(
                "整图选项数不符（OCR 2 vs 按钮 3）；拼接路未运行",
                read=_partial_read(),
            )

    class Vision:
        usage = None
        last_question_usage = None
        local_ocr_available = True

        def read_question_local(self, image):
            return RejectingOcr().read_question_local(image)

        def read_question_visual(self, image, prompt=None, *, timeout=None, deadline=None):
            raise AssertionError("不该走图片路径")

    grabs = {"n": 0}

    def counting_grab(region):
        grabs["n"] += 1
        return _blank()

    monkeypatch.setattr(pk, "grab_screen", counting_grab)

    context = pk.PKCaptureContext()
    # 等待阶段会在第一轮就确认答题页（模板放行 + 题号行读到）并返回。
    logs = []

    class Recorder:
        def log(self, message):
            logs.append(message)

    image, question = pk.wait_for_question(
        (0, 0, 430, 788), Vision(), timeout=30.0, verbose=False,
        ocr=None, capture_context=context, recorder=Recorder(),
    )

    assert image is not None
    assert question is None
    assert context.ocr_read is None, "门控判否读数不能冒充完整预读"
    assert context.progress_text == "第1题·共10题", "题号证据应独立保留"
    assert not any("本地读题门控通过" in message for message in logs)


@pytest.mark.parametrize("local_read,source", [(True, "本地 OCR"), (False, "图片模型")])
def test_option_count_mismatch_rejects_candidate_and_names_source(monkeypatch, local_read, source):
    now = 1000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)
    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    clicked = []
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **k: clicked.append(box))

    class Vision:
        usage = None
        last_question_usage = None

        def read_question(self, image, **kwargs):
            return VisionQuestion("题目", ["甲", "丙"], answer_index=1, local_read=local_read)

    result = pk.answer_one_question(
        (0, 0, 430, 788),Vision(), image=_blank(), captured_at=now, verbose=False,
    )
    assert result["guessed"] is True, "选项数量不符的候选下标不能继续使用"
    assert result["answer"] == 0
    assert source in result["error"]
    assert len(clicked) == 1


def test_option_count_mismatch_respects_disabled_guessing(monkeypatch):
    monkeypatch.setattr(pk.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: pytest.fail("不应点击无效候选答案"))

    class Vision:
        usage = None
        last_question_usage = None

        def read_question(self, image, **kwargs):
            return VisionQuestion("题目", ["甲", "丙"], answer_index=1)

    with pytest.raises(AnswerUnavailable):
        pk.answer_one_question(
            (0, 0, 430, 788),Vision(), image=_blank(), captured_at=1000.0,
            guess_on_failure=False, verbose=False,
        )


def test_empty_slot_preread_never_requests_text_answer(monkeypatch):
    monkeypatch.setattr(pk.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(3))
    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    read = _partial_read()
    read.options = ["甲", "", "丙"]
    read.confidences = [0.99, 0.0, 0.99]

    class Vision:
        usage = None
        last_question_usage = None

        def answer_from_ocr_read(self, *a, **k):
            pytest.fail("空槽预读不得发送文本答案请求")

        def read_question(self, image, **kwargs):
            return VisionQuestion("题目", ["甲", "乙", "丙"], answer_index=2)

    result = pk.answer_one_question(
        (0, 0, 430, 788),Vision(), image=_blank(), captured_at=1000.0,
        ocr_read=read, verbose=False,
    )
    assert result["answer"] == 2
    assert not result["guessed"]


class _Stop(Exception):
    pass
