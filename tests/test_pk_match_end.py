"""「对局结束」的两条判据：比分行还在不在、结算页字样读不读得到。"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from weread_quiz.result import is_settlement_page
from weread_quiz.screen import has_score_row

# 青色（按钮/边框的颜色）
CYAN = (13, 193, 255)


def _frame(rows=(0,), x0=43, x1=388) -> Image.Image:
    """在指定行画一条整行青色横线（模拟比分面板的边框）。"""
    image = Image.new("RGB", (430, 788), (10, 20, 40))
    draw = ImageDraw.Draw(image)
    for y in rows:
        draw.rectangle((x0, y, x1, y + 2), fill=CYAN)
    return image


def test_score_row_detected_when_a_full_width_cyan_line_exists():
    assert has_score_row(_frame(rows=(131, 175)))


def test_score_row_absent_without_the_panel():
    assert not has_score_row(_frame(rows=()))
    # 结算页那一带最宽的一行只有称号六边形的边（176 像素）—— 不算数。
    assert not has_score_row(_frame(rows=(131,), x0=130, x1=290))


def test_score_row_ignores_short_fragments():
    """零碎的青色像素不算比分行。"""
    assert not has_score_row(_frame(rows=(131,), x0=43, x1=130))


class FakeOcr:
    """假的 OCR 读取器：按调用次数返回预设的行。"""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def _engine_run(self, image):
        self.calls += 1
        return self.rows


def _row(text, score=0.99):
    return ([[0, 0], [10, 0], [10, 10], [0, 10]], text, score)


@pytest.mark.parametrize("text", ["获得积分", "再来一局", "炫耀一下", "查看排行榜"])
def test_settlement_page_detected_from_its_own_words(text):
    assert is_settlement_page(_frame(), FakeOcr([_row(text)]))


def test_question_page_is_not_a_settlement_page():
    ocr = FakeOcr([_row("第3题·共10题"), _row("雨足高田白"), _row("披蓑半夜耕")])
    assert not is_settlement_page(_frame(), ocr)


def test_unreadable_frame_is_not_a_settlement_page():
    """读不出字（引擎不可用/空白帧）时当"不是结算页" —— 宁可多等。"""
    assert not is_settlement_page(_frame(), FakeOcr([]))
    assert not is_settlement_page(_frame(), None)
    assert not is_settlement_page(_frame(), object())
