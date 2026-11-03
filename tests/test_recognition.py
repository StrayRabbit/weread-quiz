"""两种识别 adapter 共用的行为契约；不抓屏、不点击、不联网。"""
from types import SimpleNamespace

import pytest
from PIL import Image

from weread_quiz.daily_hybrid import DailyHybridReader
from weread_quiz.daily_ocr import DailyOcrRead, DailyOcrReader
from weread_quiz.errors import AnswerUnavailable, OcrRejected, RecognitionTimeout
from weread_quiz.ocr_core import OcrDecision, OcrRead
from weread_quiz.pk_hybrid import HybridQuestionReader
from weread_quiz.pk_ocr import RapidOcrReader
from weread_quiz.vision import VisionQuestion


class Model:
    def __init__(self):
        self.calls = []
        self.text_result = 1

    def answer_from_text(self, prompt, count, **kwargs):
        self.calls.append(("text", prompt, count, kwargs))
        return self.text_result

    def read_question(self, image, **kwargs):
        self.calls.append(("image", image.size, kwargs))
        return VisionQuestion("图片题目", ["甲", "乙", "丙"], 2)


@pytest.fixture(params=["pk", "daily"])
def adapter(request, monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    image = Image.new("RGB", (430, 788))
    fields = dict(stem="题目", options=["甲", "乙", "丙"],
                  confidences=[0.99] * 3, button_count=3,
                  has_progress=True, progress_text="第2题·共12题")
    if request.param == "daily":
        read = DailyOcrRead(**fields, progress_no=2)
        ocr = DailyOcrReader()
        reader_type = DailyHybridReader
    else:
        read = OcrRead(**fields)
        ocr = RapidOcrReader()
        reader_type = HybridQuestionReader
    counts = {"whole": 0, "strip": 0}

    def whole(frame):
        assert frame is image
        counts["whole"] += 1
        return read

    def strip(frame, previous=None):
        counts["strip"] += 1
        return None

    monkeypatch.setattr(ocr, "read", whole)
    monkeypatch.setattr(ocr, "read_strip", strip)
    model = Model()
    reader = reader_type(model, ocr, verbose=False)
    return SimpleNamespace(reader=reader, model=model, read=read, image=image,
                           counts=counts, mode=request.param)


def test_preread_is_consumed_without_second_ocr(adapter):
    local = adapter.reader.read_question_local(adapter.image)
    question = adapter.reader.answer_from_ocr_read(local, timeout=3.0, deadline=103.0)
    assert adapter.counts == {"whole": 1, "strip": 0}
    assert len(adapter.model.calls) == 1
    assert adapter.model.calls[0][0] == "text"
    assert "0. 甲\n1. 乙\n2. 丙" in adapter.model.calls[0][1]
    assert adapter.model.calls[0][3] == {"timeout": 3.0, "deadline": 103.0}
    assert question.answer_index == 1 and question.local_read
    if adapter.mode == "daily":
        assert question.progress_no == 2


def test_local_rejection_preserves_progress_without_model_call(adapter):
    adapter.read.confidences = [0.1] * 3
    with pytest.raises(OcrRejected) as rejected:
        adapter.reader.read_question_local(adapter.image)
    assert rejected.value.read is adapter.read
    assert rejected.value.read.has_progress
    assert adapter.model.calls == []
    assert adapter.counts == {"whole": 1, "strip": 1}


@pytest.mark.parametrize("options", [["甲", "", "丙"], ["甲", "丙"], []])
def test_partial_preread_never_reaches_text_model(adapter, options):
    adapter.read.options = options
    assert adapter.reader.answer_from_ocr_read(adapter.read) is None
    assert adapter.model.calls == []


@pytest.mark.parametrize("index", [None, -1, 3])
def test_invalid_model_index_is_not_mapped_to_button(adapter, index):
    adapter.model.text_result = index
    assert adapter.reader.answer_from_ocr_read(adapter.read) is None
    assert len(adapter.model.calls) == 1


def test_text_failure_falls_back_once_without_repeating_ocr(adapter):
    adapter.model.text_result = None
    question = adapter.reader.read_question(adapter.image, deadline=109.0)
    assert question.answer_index == 2 and not question.local_read
    assert adapter.counts == {"whole": 1, "strip": 0}
    assert [call[0] for call in adapter.model.calls] == ["text", "image"]
    assert adapter.model.calls[-1][-1]["deadline"] == 109.0
    assert adapter.model.calls[-1][1] == ((370, 500) if adapter.mode == "daily" else (430, 788))


def test_insufficient_remaining_budget_does_not_send_image(adapter):
    adapter.model.text_result = None
    with pytest.raises(RecognitionTimeout):
        adapter.reader.read_question(adapter.image, deadline=103.0)
    assert [call[0] for call in adapter.model.calls] == ["text"]


def test_visual_entry_never_runs_local_recognition(adapter):
    adapter.reader.read_question_visual(adapter.image, deadline=109.0)
    assert adapter.counts == {"whole": 0, "strip": 0}
    assert [call[0] for call in adapter.model.calls] == ["image"]


def test_adapter_keeps_its_text_failure_policy(adapter):
    failure = AnswerUnavailable("request failed")

    def answer(prompt, count, **kwargs):
        adapter.model.calls.append(("text", prompt, count, kwargs))
        raise failure

    adapter.model.answer_from_text = answer
    if adapter.mode == "pk":
        with pytest.raises(AnswerUnavailable) as rejected:
            adapter.reader.read_question(adapter.image, deadline=109.0)
        assert rejected.value is failure
        assert [call[0] for call in adapter.model.calls] == ["text"]
    else:
        question = adapter.reader.read_question(adapter.image, deadline=109.0)
        assert question.answer_index == 2
        assert [call[0] for call in adapter.model.calls] == ["text", "image"]
    assert adapter.counts == {"whole": 1, "strip": 0}


def test_shared_local_read_observes_whole_before_deadline_check(monkeypatch):
    from weread_quiz.recognition import read_local

    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    read = OcrRead("题目", ["甲"], [0.99], 1, has_progress=True)
    seen = []
    ocr = SimpleNamespace(read=lambda image: read,
                          decide_merged=lambda whole, strip: OcrDecision(True, "", whole))

    def observe(item):
        seen.append(item)
        return 105.0

    result = read_local(ocr, object(), deadline=101.0, observe=observe)
    assert result.read is read and result.deadline == 105.0
    assert not result.strip_ran and result.strip_options == 0
    assert seen == [read]


def test_shared_local_read_retains_rejected_strip_evidence():
    from weread_quiz.recognition import read_local

    whole = OcrRead("题目", [], [], 3)
    strip = OcrRead("题目", ["甲"], [0.2], 3, has_progress=True)
    ocr = SimpleNamespace(
        read=lambda image: whole, read_strip=lambda image, previous: strip,
        decide_merged=lambda first, second: OcrDecision(False, "判否", first, needs_strip=second is None),
    )
    result = read_local(ocr, object())
    assert not result.decision.accepted
    assert result.read is strip
    assert result.strip_ran and result.strip_options == 1


def test_shared_cascade_does_not_catch_text_failure():
    from weread_quiz.recognition import text_then_image

    def answer():
        raise AnswerUnavailable("request failed")

    with pytest.raises(AnswerUnavailable, match="request failed"):
        text_then_image(answer, lambda: pytest.fail("异常处置由 adapter 决定"),
                        minimum_image_budget=3.5)


def test_shared_cascade_rejects_expired_image_budget(monkeypatch):
    from weread_quiz.recognition import text_then_image

    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    with pytest.raises(RecognitionTimeout):
        text_then_image(lambda: None, lambda: pytest.fail("原预算已耗尽"),
                        deadline=100.0, minimum_image_budget=3.5)
