"""RapidOCR 3.x 的输出适配：全程使用本地替身，不加载模型或联网。"""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
from PIL import Image

from weread_quiz.ocr_core import RapidOcrEngine
from weread_quiz.pk.ocr import RapidOcrReader
from weread_quiz.daily.ocr import DailyOcrReader
from weread_quiz.pk import number as pk_number


def _output(texts=("第 3 题·共 10 题", "汉朝")):
    boxes = np.array([[[10, 10], [80, 10], [80, 30], [10, 30]]] * len(texts))
    return SimpleNamespace(boxes=boxes, txts=texts, scores=tuple(0.99 for _ in texts))


class StubEngine:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return self.output


def test_engine_converts_v3_output_to_existing_rows():
    output = _output()
    stub = StubEngine(output)
    reader = RapidOcrEngine()
    reader._engine = stub

    rows = reader._engine_run(Image.new("RGBA", (100, 50)))

    assert rows is not None
    assert [row[1] for row in rows] == list(output.txts)
    assert [row[2] for row in rows] == list(output.scores)
    np.testing.assert_array_equal(rows[0][0], output.boxes[0])
    image, kwargs = stub.calls[0]
    assert image.shape == (50, 100, 3)
    assert kwargs == {"use_cls": False, "use_det": True}


def test_detection_mode_is_restored_after_recognition_only():
    """RapidOCR 3.x 会保留 use_det=False，下一次整图调用必须显式打开。"""
    class StatefulEngine:
        def __init__(self):
            self.use_det = True

        def __call__(self, image, **kwargs):
            self.use_det = kwargs.get("use_det", self.use_det)
            output = _output(("题目",))
            if not self.use_det:
                output.boxes = None
            return output

    reader = RapidOcrEngine()
    reader._engine = StatefulEngine()
    assert reader._engine_run_text(Image.new("RGB", (80, 30))) == [("题目", 0.99)]
    rows = reader._engine_run(Image.new("RGB", (430, 788)))
    assert rows is not None, "分段补读不能使后续整图 OCR 永久丢失文字框"
    assert rows[0][1] == "题目"


@pytest.mark.parametrize("output", [
    SimpleNamespace(boxes=None, txts=None, scores=None),
    SimpleNamespace(boxes=np.empty((0, 4, 2)), txts=(), scores=()),
])
def test_engine_empty_output_is_unreadable(output):
    reader = RapidOcrEngine()
    reader._engine = StubEngine(output)
    assert reader._engine_run(Image.new("RGB", (100, 50))) is None


def test_engine_inference_error_is_unreadable():
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    reader = RapidOcrEngine()
    reader._engine = fail
    assert reader._engine_run(Image.new("RGB", (100, 50))) is None


@pytest.mark.parametrize("reader_type", [RapidOcrEngine, RapidOcrReader, DailyOcrReader])
def test_readers_initialize_v3_and_warm_without_classification(monkeypatch, reader_type):
    stub = StubEngine(_output())
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=lambda: stub))

    def legacy_engine():
        raise AssertionError("must not initialize rapidocr_onnxruntime")

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime",
                        SimpleNamespace(RapidOCR=legacy_engine))
    reader = reader_type()
    assert reader.warmup(Image.new("RGB", (100, 50))), reader.load_error
    assert reader._engine is stub
    assert reader.warm_error == ""
    assert stub.calls[0][1] == {"use_cls": False}
    assert reader.warmup()
    assert len(stub.calls) == 1


@pytest.mark.parametrize("source", ["direct", "reader", "default"])
def test_question_number_reads_v3_output(monkeypatch, source):
    stub = StubEngine(_output(("第 3 题·共 10 题",)))
    if source == "default":
        monkeypatch.setattr(pk_number, "_shared_engine", None)
        monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=lambda: stub))
        monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime",
                            SimpleNamespace(RapidOCR=lambda: StubEngine(_output(()))))
        engine = None
    elif source == "reader":
        engine = RapidOcrReader()
        engine._engine = stub
    else:
        engine = stub

    result = pk_number.read_question_number(Image.new("RGB", (430, 788)), ocr=engine)

    assert result == (3, 10, "第 3 题·共 10 题")
    assert stub.calls[0][1] == {"use_cls": False}


def test_question_number_empty_v3_output_is_unknown():
    stub = StubEngine(SimpleNamespace(txts=None))
    assert pk_number.read_question_number(Image.new("RGB", (430, 788)), ocr=stub) == (None, None, "")
