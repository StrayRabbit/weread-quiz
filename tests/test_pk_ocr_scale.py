"""PK 整图 OCR 放大识别：坐标折回原图，填空横线统一成占位符。"""
from __future__ import annotations

from PIL import Image

from weread_quiz.pk.ocr import RapidOcrReader, _normalize_blank


def test_scaled_run_folds_boxes_back_to_original_image(monkeypatch):
    reader = RapidOcrReader()
    seen = {}

    def engine_run(image):
        seen["size"] = image.size
        # 放大后的坐标：原图的 (100, 200) 在 2 倍图上是 (200, 400)。
        return [([[200, 400], [260, 400], [260, 460], [200, 460]], "题", 0.99)]

    monkeypatch.setattr(reader, "_engine_run", engine_run)
    rows = reader._engine_run_scaled(Image.new("RGB", (430, 788)))
    assert seen["size"] == (860, 1576), "整图要放大后再送检测模型"
    assert rows[0][0][0] == (100.0, 200.0), "文字框坐标要折回原图"
    assert rows[0][1] == "题"


def test_normalize_blank_replaces_underscore_with_placeholder():
    assert _normalize_blank("区区_心") == "区区____心"
    assert _normalize_blank("区区__心") == "区区____心"
    assert _normalize_blank("雨足高田白") == "雨足高田白"


def test_stem_uses_blank_placeholder_instead_of_underscore(monkeypatch):
    """横线被并进文字框时，题干里是「____」占位符，不是真下划线。"""
    reader = RapidOcrReader()
    monkeypatch.setattr(reader, "_bands", lambda image: [(380, 424)])
    rows = [([[100, 300], [160, 300], [160, 330], [100, 330]], "区区_心", 0.99)]
    monkeypatch.setattr(reader, "_engine_run_scaled", lambda image: rows)
    read = reader.read(Image.new("RGB", (430, 788)))
    assert read.stem == "区区____心"
