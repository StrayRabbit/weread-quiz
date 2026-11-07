"""daily / PK 外轮廓定位：选项数量、内部孔洞和几何噪声。"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from weread_quiz.pk import buttons
from weread_quiz.daily import buttons as daily_buttons
from weread_quiz.daily.ocr import find_option_bands
from weread_quiz.errors import WqError
from weread_quiz.pk.ocr import RapidOcrReader


@pytest.mark.parametrize("mode", ["daily", "pk"])
@pytest.mark.parametrize("count", [2, 3, 4])
def test_contours_follow_actual_option_count_and_ocr_bands(mode, count):
    image = Image.new("RGB", (430, 788), (10, 15, 20))
    draw = ImageDraw.Draw(image)
    first = 300 if mode == "daily" else 380
    tops = [first + 60 * i for i in range(count)]
    for top in tops:
        if mode == "daily":
            draw.rectangle((74, top, 356, top + 44), fill=(42, 59, 74))
        else:
            # 仅边框符合颜色条件；内部深色不属于青色掩码。
            draw.rectangle((74, top, 356, top + 44), outline=(0, 200, 215), width=3)
        draw.rectangle((110, top + 15, 320, top + 28), fill="white")
    detector = daily_buttons if mode == "daily" else buttons

    boxes = detector.find_option_buttons(image)
    bands = find_option_bands(image) if mode == "daily" else RapidOcrReader()._bands(image)

    assert len(boxes) == count
    assert bands == [(top, top + 44) for top in tops]
    assert [box.center_y for box in boxes] == pytest.approx([top + 22 for top in tops], abs=1)


@pytest.mark.parametrize("mode", ["daily", "pk"])
def test_disconnected_fragments_are_not_combined_into_buttons(mode):
    image = Image.new("RGB", (430, 788), (10, 15, 20))
    draw = ImageDraw.Draw(image)
    color = (42, 59, 74) if mode == "daily" else (0, 200, 215)
    first = 300 if mode == "daily" else 380
    for top in (first, first + 70):
        draw.rectangle((65, top, 174, top + 44), fill=color)
        draw.rectangle((255, top, 364, top + 44), fill=color)
    detector = daily_buttons if mode == "daily" else buttons

    with pytest.raises(WqError):
        detector.find_option_buttons(image)
    bands = find_option_bands(image) if mode == "daily" else buttons.option_bands(image)
    assert bands == []


def test_pk_card_frame_does_not_hide_inner_button_contours():
    image = Image.new("RGB", (430, 788), (10, 15, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((44, 350, 386, 650), outline=(0, 200, 215), width=2)
    for top in (380, 460):
        draw.rectangle((74, top, 356, top + 44), outline=(0, 200, 215), width=3)

    assert buttons.option_bands(image) == [(380, 424), (460, 504)]
    assert len(buttons.find_option_buttons(image)) == 2


def test_pk_rejects_tall_non_button_color_blocks():
    image = Image.new("RGB", (430, 788), (10, 15, 20))
    draw = ImageDraw.Draw(image)
    for top in (350, 520):
        draw.rectangle((74, top, 356, top + 119), fill=(0, 200, 215))

    assert buttons.option_bands(image) == []
    with pytest.raises(WqError):
        buttons.find_option_buttons(image)


@pytest.mark.parametrize("mode", ["daily", "pk"])
def test_thin_lines_and_empty_images_are_not_buttons(mode):
    detector = daily_buttons if mode == "daily" else buttons
    image = Image.new("RGB", (430, 788), (10, 15, 20))
    with pytest.raises(WqError):
        detector.find_option_buttons(image)
    draw = ImageDraw.Draw(image)
    color = (42, 59, 74) if mode == "daily" else (0, 200, 215)
    for top in (380, 450):
        draw.rectangle((74, top, 356, top + 2), fill=color)
    with pytest.raises(WqError):
        detector.find_option_buttons(image)


@pytest.mark.parametrize("filename,tops", [
    ("pk_two_options.png", [377, 449]),
    ("pk_shifted_options.png", [407, 479]),
])
@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
def test_real_pk_contours_keep_button_order_at_different_scales(filename, tops, scale):
    with Image.open(Path(__file__).parent / "fixtures" / filename) as source:
        image = source.convert("RGB")
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)

    boxes = buttons.find_option_buttons(image)
    bands = buttons.option_bands(image)

    assert len(boxes) == 2
    assert bands is not None
    for box, (top, bottom), expected in zip(boxes, bands, tops):
        assert top == pytest.approx(expected, abs=2)
        assert bottom == pytest.approx(expected + 48, abs=2)
        assert box.center_y == pytest.approx(expected + 24, abs=2)


@pytest.mark.parametrize("mode", ["daily", "pk"])
def test_small_or_out_of_range_images_return_no_bands(mode):
    if mode == "daily":
        assert daily_buttons.find_blue_gray_bands(np.zeros((10, 10, 3), dtype=np.uint8), 295, 540) == []
    else:
        assert buttons.find_cyan_bands(Image.new("RGB", (10, 10)), 340, 700) == []
        assert buttons.find_cyan_bands(Image.new("RGB", (430, 788)), 500, 400) == []
