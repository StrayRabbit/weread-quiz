"""整行 OCR 吞掉填空位时，仅用本地分段识别恢复题干。"""
from __future__ import annotations

import pytest

from weread_quiz.pk.ocr import RapidOcrReader, TEXT_PROMPT
from weread_quiz.recognition import answer_from_read
from test_blank_line import scene


def _reader(monkeypatch, *, text="早魅虐", parts=None, lines=((212, 238, 322),)):
    image = scene(lines)
    rows = [
        ([[171, 270], [259, 270], [259, 287], [171, 287]], "第1题·共10题", 0.99),
        ([[164, 300], [265, 300], [265, 328], [164, 328]], text, 0.81),
        ([[203, 391], [226, 391], [226, 415], [203, 415]], "以", 0.99),
        ([[203, 463], [226, 463], [226, 487], [203, 487]], "为", 0.99),
    ]
    reader = RapidOcrReader()
    monkeypatch.setattr(reader, "_bands", lambda _: [(378, 426), (450, 498)])
    monkeypatch.setattr(reader, "_engine_run_scaled", lambda _: rows)
    outputs = iter(parts if parts is not None else [[("旱魃", 0.90)], [("虐", 0.99)]])
    crops = []

    def recognize(crop):
        crops.append(crop)
        return next(outputs)

    monkeypatch.setattr(reader, "_engine_run_text", recognize)
    return reader, image, crops


def test_merged_stem_is_split_and_blank_restored_without_vision(monkeypatch):
    reader, image, crops = _reader(monkeypatch)
    read = reader.read(image)
    assert read.stem == "旱魃____虐"
    assert read.options == ["以", "为"]
    assert read.has_progress
    assert reader.decide(read).accepted
    assert len(crops) == 2, "只对填空线两侧运行识别，不再运行文本检测"

    class TextOnlyModel:
        supports_image = False

        def choose_index(self, stem, options, **kwargs):
            assert stem == "旱魃____虐"
            assert options == ["以", "为"]
            return 1

    question = answer_from_read(TextOnlyModel(), read, TEXT_PROMPT)
    assert question.answer_index == 1


@pytest.mark.parametrize("text, lines", [
    ("普通题干", ()),
    ("旱魃_虐", ((212, 238, 322),)),
    ("旱魃____虐", ((212, 238, 322),)),
])
def test_normal_and_already_restored_stems_do_not_request_extra_ocr(monkeypatch, text, lines):
    reader, image, crops = _reader(monkeypatch, text=text, lines=lines, parts=[])
    read = reader.read(image)
    expected = "旱魃____虐" if text == "旱魃_虐" else text
    assert read.stem == expected
    assert crops == []


@pytest.mark.parametrize("parts", [
    [None],
    [[("旱魃", 0.40)]],
    [[("旱魃", 0.90)], None],
    [[("旱魃", 0.90)], [("虐", 0.40)]],
])
def test_failed_or_uncertain_split_does_not_partially_rewrite_stem(monkeypatch, parts):
    reader, image, _crops = _reader(monkeypatch, parts=parts)
    read = reader.read(image)
    assert read.stem == "早魅虐", "补读不完整时不能拼出半个或错位的题干"
    assert read.options == ["以", "为"]


def test_two_embedded_blanks_are_recovered_in_reading_order(monkeypatch):
    reader, image, crops = _reader(
        monkeypatch, text="甲乙丙", lines=((188, 205, 322), (225, 242, 322)),
        parts=[[("甲", 0.99)], [("乙", 0.99)], [("丙", 0.99)]])
    read = reader.read(image)
    assert read.stem == "甲____乙____丙"
    assert len(crops) == 3
