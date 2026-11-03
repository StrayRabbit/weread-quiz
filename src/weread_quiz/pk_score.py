"""PK 比分读取：从答题卡片顶部的「我 VS 对手」行读出双方分数。

    比分是**唯一**能判定胜负的证据 —— 结算页只有称号、星级和奖励积分，
    积分档位与胜负无关（实测同一星级既有 +1 也有 +5）。
"""

from __future__ import annotations

import re

# 比分行所在的纵向区间：题目卡片上方的「头像 分数 VS 分数 头像」那一行。
_SCORE_BAND = (95, 185)

_DIGITS = re.compile(r"^\d+$")


def _band_rows(image, ocr) -> list[tuple[str, float, float]]:
    """读整图 OCR，只留比分那一带。返回 (文字, 左, 右)。

    **用整图而不是裁剪比分条** —— 实测裁剪版更慢（1296ms vs 576ms，
    小图会被放大重采样），而且会漏读对方的两位数分数（149 张里错 4 张）。
    """
    reader = getattr(ocr, "_engine_run", None)
    if reader is None:
        return []
    try:
        result = reader(image) or []
    except Exception:  # noqa: BLE001 - 读不出就当没有比分
        return []
    rows = []
    for box, text, _confidence in result:
        text = re.sub(r"\s+", "", str(text))
        if not text:
            continue
        x0 = float(min(p[0] for p in box))
        x1 = float(max(p[0] for p in box))
        y0 = float(min(p[1] for p in box))
        y1 = float(max(p[1] for p in box))
        if not (_SCORE_BAND[0] <= (y0 + y1) / 2 <= _SCORE_BAND[1]):
            continue
        rows.append((text, x0, x1))
    return rows


def _part_center(x0: float, x1: float, text: str, start: int, length: int) -> float:
    """子串在整行里的横向中心（按字符等宽估算）。"""
    if not text:
        return (x0 + x1) / 2
    ratio = (start + length / 2) / len(text)
    return x0 + (x1 - x0) * ratio


def read_score(image, ocr) -> tuple[int | None, int | None]:
    """读出「我 VS 对手」的分数，返回 (我方, 对手)。

    OCR 偶尔把分数和 "VS" 合成一行（"120 VS"），所以按字符位置切分，
    不能假设「一个数字一行」。

    左边是我方、右边是对手 —— 微信读书 PK 卡片固定这个布局。
    """
    rows = _band_rows(image, ocr)
    if not rows:
        return None, None

    # 先定位 VS 的横向位置：它左边的数字是我方，右边的是对手。
    vs_x: float | None = None
    for text, x0, x1 in rows:
        upper = text.upper()
        index = upper.find("VS")
        if index >= 0:
            vs_x = _part_center(x0, x1, text, index, 2)
            break
    divider = vs_x if vs_x is not None else image.width / 2

    mine: int | None = None
    theirs: int | None = None
    for text, x0, x1 in rows:
        upper = text.upper()
        index = upper.find("VS")
        if index >= 0:
            parts = ((text[:index], 0), (text[index + 2:], index + 2))
        else:
            parts = ((text, 0),)
        for part, offset in parts:
            if not _DIGITS.match(part):
                continue
            center = _part_center(x0, x1, text, offset, len(part))
            if center < divider:
                mine = int(part)
            else:
                theirs = int(part)
    return mine, theirs
