"""慢文本请求不能饿死图片回退；蒙答记录只能复用已确认同题的 OCR。"""
import pytest
from PIL import Image

from weread_quiz.errors import AnswerUnavailable
from weread_quiz.models import Rect
from weread_quiz.ocr_core import OcrDecision, OcrRead
from weread_quiz.pk import runner as pk
from weread_quiz.pk.hybrid import HybridQuestionReader
from weread_quiz.vision import VisionQuestion


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Ocr:
    def __init__(self, clock, seconds=12):
        self.clock = clock
        self.item = OcrRead("题目", ["甲", "乙"], [0.99, 0.99], 2,
                            has_progress=True, progress_text="第1题·共10题")
        self.item.countdown_seconds = seconds
        self.accepted = True

    def read(self, image):
        self.clock.now += 0.85
        return self.item

    def decide_merged(self, whole, strip):
        return OcrDecision(self.accepted, "" if self.accepted else "置信度不足", whole)


class Model:
    usage = None
    last_question_usage = None

    def __init__(self, clock, *, supports_image=True, overrun=0.0,
                 text_answer=None, image_fails=False, text_cost=None):
        self.clock = clock
        self.supports_image = supports_image
        self.overrun = overrun
        self.text_answer = text_answer
        self.image_fails = image_fails
        self.text_cost = text_cost
        self.text_calls = []
        self.image_calls = []

    def answer_from_text(self, prompt, count, *, timeout=None, deadline=None):
        self.text_calls.append((timeout, deadline))
        cost = timeout + self.overrun if self.text_answer is None else 0.1
        self.clock.now += cost if self.text_cost is None else self.text_cost
        return self.text_answer

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        assert self.supports_image, "没有图片能力时不该请求图片"
        self.image_calls.append((deadline, deadline - self.clock.now))
        self.clock.now += 1.0
        if self.image_fails:
            raise AnswerUnavailable("image_request_failed")
        return VisionQuestion(stem="题目", options=["甲", "乙"], answer_index=1)


@pytest.fixture
def scene(monkeypatch):
    clock = Clock()
    image = Image.new("RGB", (430, 788))
    state = {"same": True, "clicks": []}
    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk, "grab_screen", lambda region: image)
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: [
        Rect(60, 380, 310, 44), Rect(60, 450, 310, 44)])
    monkeypatch.setattr(pk, "is_same_question", lambda *a: state["same"])
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **kw: state["clicks"].append(box.center_y))
    monkeypatch.setattr(pk, "read_question_number", lambda *a, **kw: (2, 10, "第2题·共10题"))
    return clock, image, state


def answer(scene, entry, model, *, seconds=12):
    clock, image, _ = scene
    ocr = Ocr(clock, seconds)
    reader = HybridQuestionReader(model, ocr, verbose=False, text_timeout=6.5)
    options = {}
    if entry == "preread":
        options["ocr_read"] = ocr.read(image)
    result = pk.answer_one_question(None, reader, image=image, captured_at=1000.0,
        capture_context=pk.PKCaptureContext(captured_at=1000.0), verbose=False, **options)
    return result, reader


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
@pytest.mark.parametrize("overrun", [0.0, 0.02, 0.2])
def test_slow_text_leaves_image_budget_even_when_timeout_exits_late(scene, entry, overrun):
    clock, _, state = scene
    model = Model(clock, overrun=overrun)
    result, _ = answer(scene, entry, model)
    assert model.text_calls == [(pytest.approx(5.85), pytest.approx(1006.7))]
    assert len(model.image_calls) == 1
    assert model.image_calls[0][0] == 1010.5, "图片回退不能延长整题截止时间"
    assert model.image_calls[0][1] >= 3.5
    assert result["answer"] == 1 and not result["guessed"]
    assert len(state["clicks"]) == 1
    assert clock.now < 1010.5


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_timeout_exit_beyond_margin_does_not_force_image_request(scene, entry):
    clock, _, state = scene
    model = Model(clock, overrun=0.4)
    result, _ = answer(scene, entry, model)
    assert model.image_calls == [], "剩余不足 3.5 秒时不能硬发图片请求"
    assert result["guessed"] and result["options"] == ["甲", "乙"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_image_only_budget_skips_text_request(scene, entry):
    clock, _, state = scene
    model = Model(clock)
    result, _ = answer(scene, entry, model, seconds=6)
    assert model.text_calls == []
    assert model.image_calls == [(1004.5, pytest.approx(3.65))]
    assert result["answer"] == 1 and not result["guessed"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_no_image_capability_keeps_configured_text_budget(scene, entry):
    clock, _, _ = scene
    model = Model(clock, supports_image=False, text_answer=1)
    result, _ = answer(scene, entry, model)
    assert model.text_calls == [(6.5, 1010.5)]
    assert model.image_calls == []
    assert not result["guessed"]


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_no_image_capability_never_attempts_image_after_text_failure(scene, entry):
    clock, _, state = scene
    model = Model(clock, supports_image=False, text_cost=1.0)
    result, _ = answer(scene, entry, model)
    assert model.text_calls == [(6.5, 1010.5)]
    assert model.image_calls == []
    assert result["guessed"] and result["options"] == ["甲", "乙"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_legal_text_answer_does_not_trigger_image_review(scene, entry):
    clock, _, _ = scene
    model = Model(clock, text_answer=1)
    result, _ = answer(scene, entry, model)
    assert len(model.text_calls) == 1
    assert model.image_calls == []
    assert result["answer"] == 1 and not result["guessed"]


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_failed_models_keep_same_question_ocr_for_guess_record(scene, entry):
    clock, _, state = scene
    model = Model(clock, overrun=0.2, image_fails=True)
    result, _ = answer(scene, entry, model)
    assert result["guessed"] and not result["skipped"]
    assert result["stem"] == "题目"
    assert result["options"] == ["甲", "乙"]
    assert result["answer"] == 0
    assert len(model.image_calls) == 1
    assert len(state["clicks"]) == 1
    text = pk.format_question_line(result, ui_number=1, question_no=1)
    assert "题目" in text and "蒙第一个：甲" in text


def test_changed_question_does_not_reuse_ocr_in_timeout_record(scene):
    clock, _, state = scene
    state["same"] = False
    result, _ = answer(scene, "hybrid", Model(clock, image_fails=True))
    # 不扩大本次修改范围：沿用现有超时蒙答策略，但不能把旧题内容标在新题上。
    assert result["guessed"]
    assert result["stem"] == "" and result["options"] == []
    assert len(state["clicks"]) == 1


def test_reader_clears_previous_accepted_ocr_when_next_frame_is_rejected(scene):
    clock, image, _ = scene
    model = Model(clock, text_answer=1)
    ocr = Ocr(clock)
    reader = HybridQuestionReader(model, ocr, verbose=False)
    reader.read_question(image, deadline=1010.5)
    assert reader.last_question_local_read is ocr.item
    ocr.accepted = False
    reader.read_question(image, deadline=1010.5)
    assert reader.last_question_local_read is None
