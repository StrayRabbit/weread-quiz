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
    """读出题号行里的当前题号。读不出返回 None。"""
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
    try:
        result = engine._engine_run(band)
    except Exception:  # noqa: BLE001 - 读不出当"不知道"
        return None

    text = "".join(line[1] for line in (result or []) if line[1].strip())
    if not text:
        return None
    match = _RE_CURRENT.search(text)
    return int(match.group(1)) if match else None
