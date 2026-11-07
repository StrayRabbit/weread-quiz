"""PK 模型误分类不能覆盖当前帧的本地题号证据。"""

from __future__ import annotations

import pytest
from PIL import Image

from weread_quiz.pk import runner as pk
from weread_quiz.models import Rect
from weread_quiz.vision import VisionQuestion, make_non_question


@pytest.fixture
def scene(monkeypatch):
    image = Image.new("RGB", (430, 788), (20, 35, 70))
    boxes = [Rect(left=60, top=400 + 80 * i, width=310, height=44)
             for i in range(3)]
    clicks = []
    monkeypatch.setattr(pk.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(pk, "grab_screen", lambda region: image)
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: boxes)
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **k: clicks.append(box))
    monkeypatch.setattr(pk, "read_question_number", lambda *a, **k: (6, 10, "第6题·共10题"))
    return image, clicks


class Vision:
    usage = None
    last_question_usage = None

    def __init__(self, screen="other", progress="第6题·共10题"):
        self.screen = screen
        self.progress = progress
        self.calls = 0
        self.last_question_progress_text = ""

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        self.calls += 1
        self.last_question_progress_text = self.progress
        return make_non_question(self.screen)


class Recorder:
    def __init__(self):
        self.logs = []
        self.shots = []

    def log(self, message):
        self.logs.append(message)

    def shot(self, label, *, image):
        self.shots.append(label)


@pytest.mark.parametrize("screen", ["other", "matching"])
def test_local_progress_recovers_from_model_misclassification(scene, screen):
    image, clicks = scene
    context = pk.PKCaptureContext(captured_at=100.0)
    vision = Vision(screen)
    recorder = Recorder()
    result = pk.answer_one_question(
        (0, 0, 430, 788),vision, image=image,
        capture_context=context, recorder=recorder, verbose=False,
    )
    assert result["guessed"] is True
    assert result["skipped"] is False
    assert len(clicks) == 1
    assert vision.calls == 1, "分类冲突不得再请求一次模型"
    assert screen in result["error"]
    assert any("题号" in message and screen in message for message in recorder.logs)
    assert recorder.shots, "误分类恢复后仍须保留题目截图"


def test_misclassification_recovery_still_checks_same_question(scene, monkeypatch):
    image, clicks = scene
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: False)
    result = pk.answer_one_question(
        (0, 0, 430, 788),Vision(), image=image,
        capture_context=pk.PKCaptureContext(captured_at=100.0), verbose=False,
    )
    assert result["skipped"] is True
    assert result["reason"] == "screen_changed"
    assert not clicks


def test_misclassification_does_not_override_disabled_guessing(scene):
    image, clicks = scene
    with pytest.raises(pk.AnswerUnavailable):
        pk.answer_one_question(
            (0, 0, 430, 788),Vision(), image=image,
            capture_context=pk.PKCaptureContext(captured_at=100.0),
            guess_on_failure=False, verbose=False,
        )
    assert not clicks


def test_non_question_without_progress_does_not_click(scene):
    image, clicks = scene
    recorder = Recorder()
    with pytest.raises(pk.NotAQuestionScreen):
        pk.answer_one_question(
            (0, 0, 430, 788),Vision(progress=""), image=image,
            capture_context=pk.PKCaptureContext(captured_at=100.0),
            recorder=recorder, verbose=False,
        )
    assert not clicks
    assert any("other" in message for message in recorder.logs), "退出时应记录模型分类"


def test_missing_cached_progress_can_use_local_progress_reader(scene):
    image, clicks = scene

    class Ocr:
        def read_progress_text(self, candidate):
            assert candidate is image
            return True, "第6题·共10题"

    result = pk.answer_one_question(
        (0, 0, 430, 788),Vision(progress=""), image=image,
        captured_at=100.0, ocr=Ocr(), verbose=False,
    )
    assert result["guessed"] is True
    assert len(clicks) == 1


def test_run_continues_after_sixth_question_misclassification(scene, monkeypatch):
    image, clicks = scene
    vision = Vision()

    def wait(region, vision, *, capture_context, **kwargs):
        capture_context.captured_at = 100.0
        return image, None

    def next_question(region, current, *, capture_context, **kwargs):
        capture_context.reset()
        capture_context.captured_at = 100.0
        return image, None

    def read_question(candidate, prompt=None, *, timeout=None, deadline=None):
        vision.calls += 1
        if vision.calls == 1:
            vision.last_question_progress_text = "第6题·共10题"
            return make_non_question("other")
        vision.last_question_progress_text = "第7题·共10题"
        return VisionQuestion("下一题", ["甲", "乙", "丙"], answer_index=1)

    finals = []
    monkeypatch.setattr(pk, "wait_for_question", wait)
    monkeypatch.setattr(pk, "wait_for_next_question", next_question)
    monkeypatch.setattr(pk, "_read_final", lambda *a, **k: finals.append(True))
    vision.read_question = read_question
    report = pk.run_pk((0, 0, 430, 788), vision, max_questions=2, verbose=False)
    assert report.answered == 2, "第6题误分类不能提前结束整局"
    assert report.questions[0]["guessed"] is True
    assert report.questions[1]["guessed"] is False
    assert report.ui_numbers == [6, 7]
    assert len(clicks) == 2
    assert vision.calls == 2
    assert len(finals) == 1
