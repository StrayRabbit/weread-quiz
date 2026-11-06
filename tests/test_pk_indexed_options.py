"""PK 选项必须按按钮槽位排列，漏字不能压缩下标。"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from weread_quiz.errors import OcrRejected
from weread_quiz.ocr_core import OcrRead
from weread_quiz.pk_hybrid import HybridQuestionReader
from weread_quiz.pk_ocr import RapidOcrReader


BANDS = [(380, 424), (460, 504), (540, 584)]


def row(text, y, x=190, score=0.99):
    return ([[x, y - 8], [x + 30, y - 8], [x + 30, y + 8], [x, y + 8]], text, score)


def partial():
    return OcrRead("题目", ["甲", "", "丙"], [0.99, 0.0, 0.99], 3,
                   has_progress=True, progress_text="第6题·共10题")


def frame():
    image = Image.new("RGB", (430, 788), (20, 35, 70))
    draw = ImageDraw.Draw(image)
    for index, (top, bottom) in enumerate(BANDS):
        draw.rectangle((200, top + 14, 210 + index * 8, bottom - 14), fill="white")
    return image


class Recognizer:
    def __init__(self, texts=("乙",), scores=None):
        self.texts = texts
        self.scores = scores if scores is not None else (0.99,) * len(texts)
        self.calls = []

    def recognize_txt(self, crops):
        self.calls.append(crops)
        return SimpleNamespace(txts=self.texts, scores=self.scores)


@pytest.fixture
def reader(monkeypatch):
    reader = RapidOcrReader()
    monkeypatch.setattr(reader, "_bands", lambda image: BANDS)
    return reader


def test_whole_frame_keeps_missing_middle_slot(reader, monkeypatch):
    rows = [row("第6题·共10题", 278), row("题目", 320), row("甲", 402), row("丙", 562)]
    # _engine_run_scaled 返回的坐标已经折回原图 —— 这里只关心映射逻辑。
    monkeypatch.setattr(reader, "_engine_run_scaled", lambda image: rows)
    read = reader.read(frame())
    assert read.options == ["甲", "", "丙"]
    assert read.confidences == [0.99, 0.0, 0.99]
    assert read.button_count == 3
    assert read.has_progress


def test_multiple_lines_in_one_button_are_one_option(reader, monkeypatch):
    rows = [row("题目", 320), row("下行", 408), row("丙", 562),
            row("上行", 390), row("乙", 482)]
    monkeypatch.setattr(reader, "_engine_run_scaled", lambda image: rows)
    read = reader.read(frame())
    assert read.options == ["上行下行", "乙", "丙"]
    assert len(read.confidences) == 3


def test_recovery_recognizes_only_missing_button_and_preserves_indices(reader):
    engine = Recognizer()
    reader._engine = engine
    read = reader.read_strip(frame(), partial())
    assert read.options == ["甲", "乙", "丙"]
    assert read.confidences == [0.99, 0.99, 0.99]
    assert read.has_progress and read.progress_text == "第6题·共10题"
    assert len(engine.calls) == 1
    assert len(engine.calls[0]) == 1, "完整可信的按钮不应重复识别"
    assert engine.calls[0][0].shape[1] < 60, "裁图应去掉大面积按钮背景"
    assert reader.decide_merged(partial(), read).accepted


@pytest.mark.parametrize("texts,scores", [(("",), (0.99,)), ((), ()), (("乙", "多余"), (0.99, 0.99)), (("乙",), ())])
def test_failed_or_unpaired_recognition_does_not_shift_options(reader, texts, scores):
    reader._engine = Recognizer(texts, scores)
    read = reader.read_strip(frame(), partial())
    assert read.options == ["甲", "", "丙"]
    assert not reader.decide_merged(partial(), read).accepted


def test_empty_button_has_no_hallucinated_recognition(reader):
    image = frame()
    ImageDraw.Draw(image).rectangle((68, 466, 362, 498), fill=(20, 35, 70))
    engine = Recognizer()
    reader._engine = engine
    read = reader.read_strip(image, partial())
    assert read.options == ["甲", "", "丙"]
    assert engine.calls == []


def test_empty_slot_is_rejected_even_at_zero_threshold():
    reader = RapidOcrReader(min_confidence=0.0)
    read = partial()
    decision = reader.decide_merged(read, None)
    assert not decision.accepted
    assert decision.needs_strip
    assert "1" in decision.reason and "下标" in decision.reason


def test_compacted_old_read_is_not_reused_with_wrong_indices(reader):
    engine = Recognizer(("甲", "乙", "丙"))
    reader._engine = engine
    read = partial()
    read.options = ["甲", "丙"]
    read.confidences = [0.99, 0.99]
    recovered = reader.read_strip(frame(), read)
    assert recovered.options == ["甲", "乙", "丙"]
    assert len(engine.calls[0]) == 3
    assert reader.decide_merged(read, None).needs_strip


@pytest.mark.parametrize("previous_text", ["月", "日"])
def test_low_confidence_evidence_is_not_erased_by_higher_recovery_score(reader, previous_text):
    read = partial()
    read.options[1] = previous_text
    read.confidences[1] = 0.60
    reader._engine = Recognizer(("日",), (0.95,))
    recovered = reader.read_strip(frame(), read)
    assert not reader.decide_merged(read, recovered).accepted


def test_multiline_button_recovery_keeps_both_lines_in_same_slot(reader):
    image = frame()
    draw = ImageDraw.Draw(image)
    draw.rectangle((68, 466, 362, 498), fill=(20, 35, 70))
    draw.rectangle((200, 468, 220, 477), fill="white")
    draw.rectangle((200, 487, 220, 496), fill="white")
    reader._engine = Recognizer(("上行", "下行"))
    recovered = reader.read_strip(image, partial())
    assert recovered.options == ["甲", "上行下行", "丙"]
    assert len(reader._engine.calls[0]) == 2


def test_complete_read_does_not_request_button_recovery(reader, monkeypatch):
    read = partial()
    read.options[1] = "乙"
    read.confidences[1] = 0.99
    monkeypatch.setattr(reader, "read", lambda image: read)
    monkeypatch.setattr(reader, "read_strip", lambda *args: pytest.fail("完整读数不应补读"))
    result = HybridQuestionReader(object(), reader, verbose=False).read_question_local(frame())
    assert result is read


def test_hybrid_answers_in_original_button_coordinate_system(reader, monkeypatch):
    reader._engine = Recognizer()
    monkeypatch.setattr(reader, "read", lambda image: partial())

    class Model:
        def answer_from_text(self, body, count, **kwargs):
            assert "0. 甲\n1. 乙\n2. 丙" in body
            assert count == 3
            return 2

    hybrid = HybridQuestionReader(Model(), reader, verbose=False)
    question = hybrid.read_question(frame())
    assert question.options == ["甲", "乙", "丙"]
    assert question.answer_index == 2
    assert question.local_read


def test_unrecovered_slot_never_reaches_text_model(reader, monkeypatch):
    reader._engine = Recognizer(("",))
    monkeypatch.setattr(reader, "read", lambda image: partial())
    hybrid = HybridQuestionReader(object(), reader, verbose=False)
    with pytest.raises(OcrRejected) as rejected:
        hybrid.read_question_local(frame())
    assert rejected.value.read.has_progress
    assert rejected.value.read.options == ["甲", "", "丙"]


def test_direct_preread_answer_rejects_empty_slot_before_request(reader):
    class Model:
        def answer_from_text(self, *args, **kwargs):
            pytest.fail("空槽不能绕过门控直接提交文本模型")

    hybrid = HybridQuestionReader(Model(), reader, verbose=False)
    assert hybrid.answer_from_ocr_read(partial()) is None


def test_recovery_without_whole_read_keeps_card_evidence(reader, monkeypatch):
    monkeypatch.setattr(reader, "_read_card_text_detail", lambda image: (True, "题目", "第6题·共10题"))
    reader._engine = Recognizer(("甲", "乙", "丙"))
    recovered = reader.read_strip(frame())
    assert recovered.options == ["甲", "乙", "丙"]
    assert recovered.has_progress
    assert reader.decide_merged(None, recovered).accepted


def test_recovery_cannot_supply_missing_progress_evidence(reader):
    read = partial()
    read.has_progress = False
    read.progress_text = ""
    reader._engine = Recognizer()
    recovered = reader.read_strip(frame(), read)
    assert recovered.options == ["甲", "乙", "丙"]
    assert not reader.decide_merged(read, recovered).accepted
