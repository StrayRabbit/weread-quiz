"""倒计时预算及超时题号复用的调用链回归。"""
from types import SimpleNamespace

import pytest
from PIL import Image

from weread_quiz.pk import runner as pk
from weread_quiz.errors import RecognitionTimeout
from weread_quiz.models import Rect
from weread_quiz.ocr_core import OcrRead
from weread_quiz.pk.hybrid import HybridQuestionReader
from weread_quiz.pk.ocr import RapidOcrReader
from weread_quiz.vision import VisionQuestion


def line(text, x=180, y=220, confidence=0.99):
    return ([(x - 20, y - 8), (x + 20, y - 8), (x + 20, y + 8), (x - 20, y + 8)], text, confidence)


def mapped(*timer_lines):
    rows = [line("第1题·共10题", y=278), line("题目", y=320),
            line("甲", y=402), line("乙", y=474), *timer_lines]
    scene = Image.new("RGB", (430, 788), (20, 35, 70))
    return RapidOcrReader()._map_lines(rows, [(378, 426), (450, 498)],
                                       stem_from_bands=True, image=scene)


@pytest.mark.parametrize("text,seconds", [("00:12", 12), ("00:09", 9), ("00：05", 5), ("00:00", 0)])
def test_countdown_is_retained_from_existing_full_frame_ocr(text, seconds):
    assert getattr(mapped(line(text)), "countdown_seconds", None) == seconds


@pytest.mark.parametrize("rows", [
    [line("00:99")], [line("00:13")], [line("01:12")],
    [line("00:12", confidence=0.7)], [line("00:12", y=320)],
    [line("00:12", x=80)], [line("00:12"), line("00:09")],
])
def test_unreliable_countdown_is_not_used(rows):
    assert getattr(mapped(*rows), "countdown_seconds", None) is None


def read(seconds=12):
    item = OcrRead("题目", ["甲", "乙"], [0.99, 0.99], 2,
                   has_progress=True, progress_text="第1题·共10题")
    item.countdown_seconds = seconds
    return item


@pytest.fixture
def scene(monkeypatch):
    state = {"now": 1000.0, "clicks": 0, "number_reads": 0, "same": True}
    image = Image.new("RGB", (430, 788))
    monkeypatch.setattr(pk.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(pk, "grab_screen", lambda region: image)
    monkeypatch.setattr(pk, "find_option_buttons", lambda frame: [Rect(60, 380, 310, 44), Rect(60, 450, 310, 44)])
    monkeypatch.setattr(pk, "is_same_question", lambda *a: state["same"])
    def click(*a, **kw):
        state["clicks"] += 1
    def number(*a, **kw):
        state["number_reads"] += 1
        return 2, 10, "第2题·共10题"
    monkeypatch.setattr(pk, "click_box", click)
    monkeypatch.setattr(pk, "read_question_number", number)
    return state, image


@pytest.mark.parametrize("seconds,captured,expected_deadline", [(12, 999.0, 1010.2), (5, 999.0, 1003.2), (None, 1000.0, 1006.0)])
def test_preread_budget_accounts_for_capture_age(scene, seconds, captured, expected_deadline):
    state, image = scene
    seen = []
    def answer(item, **kw):
        seen.append(kw["deadline"])
        return VisionQuestion(stem=item.stem, options=item.options, answer_index=1, screen="question", local_read=True)
    result = pk.answer_one_question(None, SimpleNamespace(answer_from_ocr_read=answer),
        image=image, captured_at=captured, ocr_read=read(seconds), verbose=False)
    assert seen == [expected_deadline]
    assert result["guessed"] is False


def test_countdown_with_no_remaining_budget_does_not_call_model(scene):
    state, image = scene
    def answer(*a, **kw):
        pytest.fail("没有时间时不能调用模型")
    result = pk.answer_one_question(None, SimpleNamespace(answer_from_ocr_read=answer),
        image=image, captured_at=998.0, ocr_read=read(2), verbose=False, guess_on_failure=False)
    assert result["skipped"] is True
    assert state["clicks"] == 0


def test_no_image_model_uses_countdown_budget_for_seven_second_answer(scene):
    state, image = scene
    seen = []
    class Ocr:
        def read(self, image):
            state["now"] += 0.6
            return read(12)
        def decide_merged(self, whole, strip):
            return SimpleNamespace(accepted=True, read=whole, reason="")
    class Model:
        supports_image = False

        def answer_from_text(self, prompt, count, **kw):
            seen.append(kw["deadline"])
            state["now"] += 7.0
            return 1
    hybrid = HybridQuestionReader(Model(), Ocr(), verbose=False, text_timeout=12.0)
    result = pk.answer_one_question(None, hybrid, image=image, captured_at=999.0, verbose=False)
    assert seen == [1010.2]
    assert result["guessed"] is False
    assert state["clicks"] == 1


def test_timer_deadline_still_applies_to_local_model_answer(scene):
    state, image = scene
    def answer(item, **kw):
        state["now"] = 1011.3
        return VisionQuestion(stem=item.stem, options=item.options, answer_index=1, screen="question", local_read=True)
    result = pk.answer_one_question(None, SimpleNamespace(answer_from_ocr_read=answer),
        image=image, captured_at=1000.0, ocr_read=read(12), verbose=False, guess_on_failure=False)
    assert result["skipped"] is True
    assert state["clicks"] == 0


@pytest.mark.parametrize("same,expected_number,expected_reads", [(True, 1, 0), (False, 2, 1)])
def test_timeout_reuses_progress_only_when_fresh_frame_is_same(scene, same, expected_number, expected_reads):
    state, image = scene
    state["same"] = same
    context = pk.PKCaptureContext(captured_at=1000.0, progress_text="第1题·共10题")
    def answer(*a, **kw):
        raise RecognitionTimeout("deadline_exceeded")
    result = pk.answer_one_question(None, SimpleNamespace(answer_from_ocr_read=answer),
        image=image, captured_at=1000.0, ocr_read=read(), capture_context=context, verbose=False)
    assert result["guessed"] is True
    assert result["ui_number"] == expected_number
    assert state["number_reads"] == expected_reads


def test_timeout_without_progress_still_reads_clicked_frame(scene):
    state, image = scene
    def model(*a, **kw):
        raise RecognitionTimeout("deadline_exceeded")
    result = pk.answer_one_question(None, SimpleNamespace(read_question=model),
        image=image, captured_at=1000.0, verbose=False)
    assert result["ui_number"] == 2
    assert state["number_reads"] == 1
