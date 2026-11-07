"""读取并解析 PK 题号行，返回当前题号、总题数及原文；无法确定时不猜测。"""

from __future__ import annotations

import re

from PIL import Image

_NUMBER_BAND = (250, 306)

_OCR_SCALE = 3

# 「第 N 题」里的 N。
_RE_CURRENT = re.compile(r"第\s*(\d+)\s*题")

# 「共 M 题」里的 M —— 一局总题数。
_RE_TOTAL = re.compile(r"共\s*(\d+)\s*题")

# 「最后一题」—— 11 题制的最后一题（分数翻倍）。它没有数字。
_LAST_QUESTION = "最后一题"


# 进程内共享的 RapidOCR 引擎（见 _default_engine）。
_shared_engine = None


def _default_engine():
    """进程内共享的 RapidOCR 引擎。"""
    global _shared_engine
    if _shared_engine is None:
        try:
            from ..ocr_core import build_rapid_ocr  # noqa: PLC0415 - 延迟导入
            _shared_engine = build_rapid_ocr()
        except Exception:  # noqa: BLE001 - 缺依赖/读不出就当没有
            return None
    return _shared_engine


def read_question_number(image: Image.Image, ocr=None) -> tuple[int | None, int | None, str]:
    """读出题号行。返回 (当前题号, 总题数, 原始文本)。"""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None, None, ""

    band = image.crop((0, _NUMBER_BAND[0], image.size[0], _NUMBER_BAND[1]))
    if band.size[1] < 8:
        return None, None, ""
    band = band.resize((band.size[0] * _OCR_SCALE, band.size[1] * _OCR_SCALE),
                       Image.LANCZOS)

    engine = ocr
    if engine is None:
        engine = _default_engine()
        if engine is None:
            return None, None, ""
    elif hasattr(engine, "_engine") and engine._engine is not None:
        engine = engine._engine

    try:
        output = engine(np.asarray(band.convert("RGB")), use_cls=False)
        text = "".join(line for line in (output.txts or ()) if line.strip())
    except Exception:  # noqa: BLE001 - 读不出当没有
        return None, None, ""

    if not text:
        return None, None, ""

    return parse_progress_text(text)


def parse_progress_text(text: str) -> tuple[int | None, int | None, str]:
    """从题号行的**原文**里解析出 (当前题号, 总题数, 原文)。"""
    if not text:
        return None, None, text

    current_match = _RE_CURRENT.search(text)
    total_match = _RE_TOTAL.search(text)
    current = int(current_match.group(1)) if current_match else None
    total = int(total_match.group(1)) if total_match else None
    if current is None and _LAST_QUESTION in text:
        # 「最后一题」—— 当前题号未知，但总题数可能读得到
        # （11 题制时那一行通常不写"共 N 题"，所以 total 也常为 None）。
        return None, total, text
    return current, total, text
