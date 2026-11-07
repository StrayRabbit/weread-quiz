"""每日一答按钮定位：文字遮挡不应改变选项数量和点击位置。"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from weread_quiz.daily_buttons import find_option_buttons, pick_button


@pytest.fixture
def poem_image():
    path = Path(__file__).parent / "fixtures" / "daily_poem_three_options.png"
    with Image.open(path) as source:
        return source.convert("RGB")


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
def test_real_poem_page_keeps_third_option(poem_image, scale):
    image = poem_image
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)

    boxes = find_option_buttons(image)

    assert len(boxes) == 3
    assert [box.center_y for box in boxes] == pytest.approx([330, 400, 470], abs=3)
    assert 446 <= pick_button(boxes, 2).center_y <= 493


def test_real_poem_answer_clicks_third_option_without_guessing(poem_image, monkeypatch):
    from weread_quiz import daily
    from weread_quiz.models import Rect
    from weread_quiz.vision import VisionQuestion

    cfg = daily.DailyConfig(Rect(0, 0, 430, 788), None, verbose=False)
    question = VisionQuestion(
        stem="“日出江花红胜火”的下一句是？",
        options=["能不忆江南", "风景旧曾谙", "春来江水绿如蓝"],
        answer_index=2, screen="question",
    )
    clicks = []
    monkeypatch.setattr(daily, "grab_screen", lambda region: poem_image.copy())
    monkeypatch.setattr(daily, "click_box", lambda box, *a, **k: clicks.append(box.center_y))

    result = daily._answer_one_question_inner(cfg, poem_image, question)

    assert not result.get("skipped")
    assert result["answer"] == 2
    assert result["guessed"] is False
    assert result["error"] == ""
    assert clicks == pytest.approx([470], abs=3)


@pytest.mark.parametrize("text_width", [0, 60, 130, 200])
def test_option_text_does_not_split_button_band(text_width):
    image = Image.new("RGB", (430, 788), (10, 15, 20))
    draw = ImageDraw.Draw(image)
    for top in (306, 376, 446):
        draw.rectangle((74, top, 356, top + 47), fill=(42, 59, 74))
    if text_width:
        # 白字/抗锯齿不符合填充色条件；模拟末项文字连续遮挡多行。
        left = (430 - text_width) // 2
        draw.rectangle((left, 463, left + text_width - 1, 476), fill="white")

    boxes = find_option_buttons(image)

    assert len(boxes) == 3
    assert [box.center_y for box in boxes] == pytest.approx([329, 399, 469], abs=2)
    assert 446 <= pick_button(boxes, 2).center_y <= 493


def test_separate_buttons_are_not_joined_across_background():
    image = Image.new("RGB", (430, 788), (10, 15, 20))
    draw = ImageDraw.Draw(image)
    for top in (300, 360, 420, 480):
        draw.rectangle((74, top, 356, top + 39), fill=(42, 59, 74))

    boxes = find_option_buttons(image)

    assert len(boxes) == 4
    assert [box.center_y for box in boxes] == pytest.approx([319, 379, 439, 499], abs=2)
