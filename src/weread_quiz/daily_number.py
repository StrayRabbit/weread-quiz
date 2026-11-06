"""读取每日一答题号行；优先使用卡片 OCR 预读，独立裁图仅作兜底。"""

from __future__ import annotations

import re

from PIL import Image

_NUMBER_BAND = (196, 216)
_NUMBER_LEFT = 0
_NUMBER_RIGHT = 260

# OCR 前的放大倍数。题号字很小，放大后才稳。
_OCR_SCALE = 3

# 「第 N 题」里的 N。
_RE_CURRENT = re.compile(r"第\s*(\d+)\s*题")


def read_question_number(image: Image.Image, ocr=None) -> int | None:
    """读出题号行里的当前题号。读不出返回 None。

    **先走识别模式**：题号行是位置已知的窄文字带，不需要文本检测 ——
    检测器在这一条上要 1 秒（实测 1050ms），识别模式只要 20ms。
    读题号要用在「答案到了以后按题号复核」这条路上，慢不起。

    识别模式读不出整条时退回检测模式（那条对噪声更稳）。
    """
    band = image.crop(
        (_NUMBER_LEFT, _NUMBER_BAND[0], _NUMBER_RIGHT, _NUMBER_BAND[1])
    )
    if band.size[1] < 8:
        return None
    band = band.resize(
        (band.size[0] * _OCR_SCALE, band.size[1] * _OCR_SCALE), Image.LANCZOS
    )

    engine = ocr
    if engine is None:
        from .ocr_core import RapidOcrEngine

        engine = RapidOcrEngine()
    text = _recognize_band(engine, band)
    if not text:
        try:
            result = engine._engine_run(band)
        except Exception:  # noqa: BLE001 - 读不出当"不知道"
            return None
        text = "".join(line[1] for line in (result or []) if line[1].strip())
    if not text:
        return None
    match = _RE_CURRENT.search(text)
    return int(match.group(1)) if match else None


def _recognize_band(engine, band) -> str:
    """识别模式读一条已知位置的文字带；引擎不支持就当读不出。"""
    fast = getattr(engine, "_engine_run_text", None)
    if not callable(fast):
        return ""
    try:
        lines = fast(band)
    except Exception:  # noqa: BLE001 - 读不出当"不知道"
        return ""
    return "".join(text for text, _ in (lines or []))
