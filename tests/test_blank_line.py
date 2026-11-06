"""填空横线：横线补回题干、阅读顺序修正与误报排除。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from weread_quiz.blank import PLACEHOLDER, TextSpan, compose_stem, find_blank_lines
from weread_quiz.daily_ocr import DailyOcrReader
from weread_quiz.pk_ocr import RapidOcrReader


def scene(lines=(), size=(430, 788)):
    """深色背景 + 若干条白色横线（x0, x1, y）。"""
    image = Image.new("RGB", size, (20, 35, 70))
    array = np.asarray(image).copy()
    for x0, x1, y in lines:
        array[y, x0:x1] = (255, 255, 255)
    return Image.fromarray(array)


def span(x0, x1, y0, y1, text):
    return TextSpan(x0=x0, x1=x1, y0=y0, y1=y1, text=text)


# ---- 横线检测 ----------------------------------------------------------


def test_blank_left_of_text_is_found():
    image = scene([(100, 140, 300)])
    spans = [span(150, 210, 290, 320, "题")]
    blanks = find_blank_lines(image, (248, 380), spans)
    assert len(blanks) == 1
    assert (blanks[0].x0, blanks[0].x1) == (100, 139)


def test_blank_just_touching_text_is_still_a_blank():
    """填空线常常紧贴文字（差一两像素）—— 那不算压着字。"""
    image = scene([(100, 140, 300)])
    spans = [span(141, 210, 290, 320, "题")]
    assert len(find_blank_lines(image, (248, 380), spans)) == 1


def test_stroke_inside_text_box_is_not_a_blank():
    """「一」这类横笔在自己的文字框里 —— 不能当成填空线。"""
    image = scene([(150, 190, 300)])
    spans = [span(150, 210, 290, 320, "一")]
    assert find_blank_lines(image, (248, 380), spans) == []


def test_blank_overlapping_text_box_margin_is_still_a_blank():
    """OCR 的文字框比字形宽几个像素 —— 紧跟文字的填空线蹭上这点余量，
    仍要认成填空线，不能当成字的横笔（实测「襟江带 __」就是这样丢的）。"""
    image = scene([(233, 258, 322)])
    spans = [span(167, 236, 300, 326, "襟江带")]
    blanks = find_blank_lines(image, (248, 380), spans)
    assert len(blanks) == 1
    assert (blanks[0].x0, blanks[0].x1) == (233, 257)


def test_line_with_ink_above_is_not_a_blank():
    """上下还有笔画 —— 它是更大图形的一部分。"""
    image = scene([(100, 140, 300), (100, 140, 292)])
    spans = [span(150, 210, 290, 320, "题")]
    assert find_blank_lines(image, (248, 380), spans) == []


def test_thick_white_bar_is_not_a_blank():
    """边框、图标比横线厚得多。"""
    image = scene([(100, 140, y) for y in range(300, 306)])
    spans = [span(150, 210, 290, 320, "题")]
    assert find_blank_lines(image, (248, 380), spans) == []


# ---- 题干合成 ----------------------------------------------------------


def test_blank_is_placed_before_the_text():
    image = scene([(100, 140, 300)])
    spans = [span(150, 210, 290, 320, "题")]
    assert compose_stem(spans, find_blank_lines(image, (248, 380), spans)) == "____题"


def test_blank_is_placed_between_two_texts():
    image = scene([(200, 240, 300)])
    spans = [span(100, 190, 290, 320, "甲"), span(250, 340, 290, 320, "乙")]
    assert compose_stem(spans, find_blank_lines(image, (248, 380), spans)) == "甲____乙"


def test_placeholder_is_four_underscores():
    assert PLACEHOLDER == "____"


def test_stem_follows_the_reading_order_on_screen():
    """同一行按横向位置拼 —— 之前按文字排序会把词序弄反。"""
    spans = [span(269, 354, 227, 253, "跳珠乱入"),
             span(75, 219, 229, 252, "黑云翻墨未遮山"),
             span(189, 227, 260, 288, "船。")]
    assert compose_stem(spans, []) == "黑云翻墨未遮山跳珠乱入船。"


def test_blank_without_text_on_its_line_is_ignored():
    """附近没有文字的那条线，多半是装饰而不是填空。"""
    image = scene([(100, 140, 300)])
    spans = [span(150, 210, 400, 430, "题")]
    assert compose_stem(spans, find_blank_lines(image, (248, 380), spans)) == "题"


# ---- 两条链路的接入 ----------------------------------------------------


def test_pk_reader_restores_blank_in_stem():
    image = scene([(171, 196, 322)])
    rows = [([[171, 270], [259, 270], [259, 287], [171, 287]], "第1题·共10题", 0.99),
            ([[197, 299], [262, 299], [262, 328], [197, 328]], "女红男", 0.99),
            ([[203, 391], [226, 391], [226, 415], [203, 415]], "蓝", 0.99)]
    read = RapidOcrReader()._map_lines(rows, [(378, 426)],
                                       stem_from_bands=True, image=image)
    assert read.stem == "____女红男"
    assert read.has_progress


def test_daily_reader_restores_blank_in_stem():
    image = scene([(170, 195, 249)])
    lines = [(206.0, 173.0, 256.0, 198.0, 213.0, "第1题·共12题", 0.99),
             (240.0, 196.0, 263.0, 227.0, 254.0, "窍不通", 0.99)]
    stem, has_progress, progress_no = DailyOcrReader()._split_stem(image, lines)
    assert stem == "____窍不通"
    assert has_progress
    assert progress_no == 1


def test_daily_reader_keeps_stroke_out_of_stem():
    """「第一次」的「一」是横笔，不是填空线。"""
    image = scene([(147, 163, 239)])
    lines = [(240.0, 73.0, 357.0, 229.0, 252.0, "小说“第一次亲密接触”里男女主", 0.99)]
    stem, _, _ = DailyOcrReader()._split_stem(image, lines)
    assert stem == "小说“第一次亲密接触”里男女主"
