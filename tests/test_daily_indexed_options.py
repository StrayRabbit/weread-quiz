"""daily 单字漏检回归：按钮已定位时，补读不能再次依赖文本检测器。"""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from weread_quiz.daily import ocr as daily_ocr
from weread_quiz.daily.ocr import DailyOcrRead, DailyOcrReader


BANDS = [(307, 353), (377, 423), (447, 493)]


def partial():
    return DailyOcrRead(
        stem="____所周知", options=[], confidences=[], button_count=3,
        has_progress=True, progress_no=1,
    )


class RecognitionEngine:
    def __init__(self, texts, scores):
        self.texts = texts
        self.scores = scores
        self.crops = []

    def recognize_txt(self, crops):
        self.crops.extend(crops)
        return SimpleNamespace(txts=self.texts, scores=self.scores)


def setup_reader(monkeypatch, texts=("毂", "栩", "众"), scores=(0.99, 0.99, 0.99)):
    image = Image.new("RGB", (430, 788), (30, 45, 55))
    draw = ImageDraw.Draw(image)
    for top, bottom in BANDS:
        draw.rectangle((207, top + 15, 223, bottom - 15), fill="white")
    reader = DailyOcrReader()
    reader._engine = RecognitionEngine(texts, scores)
    monkeypatch.setattr(daily_ocr, "find_option_bands", lambda image: BANDS)
    monkeypatch.setattr(daily_ocr, "has_highlighted_button", lambda image: False)
    # 复现文本检测器只返回两项；补读应绕过它，不能把漏检后的列表当下标。
    monkeypatch.setattr(reader, "_engine_run", lambda image: [
        (None, "毂", 0.99), (None, "众", 0.99),
    ])
    return reader, image


def test_button_recognition_recovers_all_single_character_options(monkeypatch):
    reader, image = setup_reader(monkeypatch)
    read = reader.read_strip(image, partial())
    assert read.options == ["毂", "栩", "众"]
    assert read.progress_no == 1 and read.has_progress
    assert read.stem == "____所周知"
    assert reader.decide_merged(partial(), read).accepted
    assert len(reader._engine.crops) == 3
    # 单字应紧裁，不能把整条宽按钮直接交给单行识别器压扁。
    assert all(crop.shape[1] < 100 for crop in reader._engine.crops)


@pytest.mark.parametrize("texts,scores", [
    (("毂", "众"), (0.99, 0.99)),
    (("毂", "", "众"), (0.99, 0.99, 0.99)),
    (("毂", "栩", "众"), (0.99, 0.3, 0.99)),
    (("毂", "栩", "众"), (0.99, float("nan"), 0.99)),
    (("毂", "栩", "众"), (0.99, float("inf"), 0.99)),
])
def test_unreliable_button_recognition_is_not_accepted(monkeypatch, texts, scores):
    reader, image = setup_reader(monkeypatch, texts, scores)
    recovered = reader.read_strip(image, partial())
    assert not reader.decide_merged(partial(), recovered).accepted


def test_button_crops_preserve_top_to_bottom_order(monkeypatch):
    reader, image = setup_reader(monkeypatch)
    draw = ImageDraw.Draw(image)
    for index, (top, bottom) in enumerate(BANDS):
        draw.rectangle((207, top + 15, 223, bottom - 15),
                       fill=(200 + index * 20,) * 3)
    reader.read_strip(image, partial())
    crops = reader._engine.crops
    assert len(crops) == 3
    assert [int(np.asarray(crop).max()) for crop in crops] == [200, 220, 240]


def test_recovery_does_not_erase_original_low_confidence(monkeypatch):
    reader, image = setup_reader(monkeypatch)
    whole = partial()
    whole.options = ["毂", "栩", "众"]
    whole.confidences = [0.5, 0.99, 0.99]
    recovered = reader.read_strip(image, whole)
    assert recovered.confidences[0] == 0.5
    assert not reader.decide_merged(whole, recovered).accepted


def test_missing_button_text_does_not_shift_option_indices(monkeypatch):
    reader, image = setup_reader(monkeypatch)
    top, bottom = BANDS[1]
    ImageDraw.Draw(image).rectangle((0, top, image.width, bottom), fill=(30, 45, 55))
    assert reader.read_strip(image, partial()) is None
    assert not reader._engine.crops


def test_button_inference_failure_falls_back(monkeypatch):
    reader, image = setup_reader(monkeypatch)

    def fail(crops):
        raise RuntimeError("识别失败")

    monkeypatch.setattr(reader._engine, "recognize_txt", fail)
    assert reader.read_strip(image, partial()) is None


def test_revealed_frame_does_not_run_button_recognition(monkeypatch):
    reader, image = setup_reader(monkeypatch)
    monkeypatch.setattr(daily_ocr, "has_highlighted_button", lambda image: True)
    assert reader.read_strip(image, partial()) is None
    assert not reader._engine.crops
