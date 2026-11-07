"""题干正文边界与超时后耗时归因回归。"""

from PIL import Image

from weread_quiz.pk import runner as pk
from weread_quiz.models import Rect
from weread_quiz.pk.ocr import RapidOcrReader


def row(text, y):
    return ([[190, y - 8], [240, y - 8], [240, y + 8], [190, y + 8]], text, 0.99)


def scene():
    """没有横线的深色画面 —— 题干就是 OCR 读到的原文。"""
    return Image.new("RGB", (430, 788), (20, 35, 70))


def test_scoreboard_is_not_question_stem():
    rows = [row("100 VS", 150), row("第8题·共10题", 278),
            row("油画之所以色彩稳定？", 320), row("胶水", 432), row("植物油", 504)]
    read = RapidOcrReader()._map_lines(rows, [(410, 454), (482, 526)],
                                       stem_from_bands=True, image=scene())
    assert read.stem == "油画之所以色彩稳定？"
    assert read.has_progress
    assert read.options == ["胶水", "植物油"]


def test_numbers_and_vs_in_actual_question_body_are_preserved():
    rows = [row("130 VS", 150), row("最后一题·分数翻倍", 278),
            row("100 VS 80 表示什么？", 320), row("甲", 432), row("乙", 504)]
    read = RapidOcrReader()._map_lines(rows, [(410, 454), (482, 526)],
                                       stem_from_bands=True, image=scene())
    assert read.stem == "100 VS 80 表示什么？"


def test_timeout_guess_number_ocr_is_its_own_timing_stage(monkeypatch):
    image = Image.new("RGB", (430, 788))
    events = []
    monkeypatch.setattr(pk, "grab_screen", lambda region: image)
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: [Rect(60, 400, 310, 44)])
    monkeypatch.setattr(pk, "click_box", lambda *args, **kwargs: events.append("click"))

    def number(*args, **kwargs):
        events.append("number_ocr")
        return None, None, "最后一题·分数翻倍"

    monkeypatch.setattr(pk, "read_question_number", number)

    class Watch:
        def mark(self, name):
            events.append(name)

    result = pk._guess_after_timeout(
        (0, 0, 430, 788), image, None,
        lambda reason, error="": {"skipped": True}, True, Watch(), "recognition_timeout",
    )
    assert result["guessed"]
    assert events.index("click") < events.index("number_ocr")
    assert events[-1] == "题号OCR", "题号读取耗时不能归入截图落盘"
    assert result["ui_number"] is None
