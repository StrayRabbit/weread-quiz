"""换题确认必须使用当前干净题面的证据，不能沿用提示条或过期读数。"""
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from weread_quiz.daily import runner as daily
from weread_quiz.daily.props import has_toast
from weread_quiz.daily.screen import is_new_question, is_question_screen
from weread_quiz.models import Rect
from test_daily_pipeline import clock  # noqa: F401


def card(stem_y=240):
    image = Image.new("RGB", (430, 788), (10, 20, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 180, 400, 290), fill=(7, 40, 79))
    draw.rectangle((170, 202, 240, 207), fill="white")
    draw.rectangle((100, stem_y, 320, stem_y + 10), fill="white")
    for top in (310, 380, 450):
        draw.rectangle((40, top, 390, top + 44), fill=(42, 59, 74))
        draw.rectangle((200, top + 15, 230, top + 25), fill="white")
    assert is_question_screen(image)
    return image


def confirm(monkeypatch, before, frames, numbers, counts):
    frames = iter(frames)
    numbers = iter(numbers)
    counts = iter(counts)
    reads = []

    def read_number(image):
        value = next(numbers, reads[-1] if reads else None)
        reads.append(value)
        return value

    config = daily.DailyConfig(
        Rect(0, 0, 430, 788), SimpleNamespace(read_progress_number=read_number),
        verbose=False,
    )
    monkeypatch.setattr(daily, "grab_screen", lambda region: next(frames, before))
    monkeypatch.setattr(daily, "_read_prop_count", lambda *a: next(counts, 0))
    outcome = daily._confirm_prop_effect(
        config, "swap", expected_number=1, expected_count=1, before_image=before,
    )
    return outcome, reads


def test_toast_change_does_not_confirm_unchanged_question(monkeypatch):
    before = card()
    toast = before.copy()
    ImageDraw.Draw(toast).rectangle((155, 207, 274, 260), fill=(1, 8, 16))
    assert has_toast(toast) and is_new_question(before, toast)
    outcome, _ = confirm(monkeypatch, before, [toast, before, before], [1], [0])
    assert outcome[0] is False, "提示条消失后仍是旧题，不能确认换题成功"


def test_transient_stem_change_is_not_reused_after_return_to_old_question(monkeypatch):
    before, changed = card(), card(270)
    assert not has_toast(changed) and is_new_question(before, changed)
    outcome, _ = confirm(monkeypatch, before, [changed, before, before], [1], [1, 0])
    assert outcome[0] is False, "当前题干没变，不能沿用历史帧差"


@pytest.mark.parametrize("numbers, expected", [([1, 2, 2], False), ([1, 2, 1], True)])
def test_swap_waits_after_first_mismatched_number(monkeypatch, numbers, expected):
    before, changed = card(), card(270)
    outcome, reads = confirm(
        monkeypatch, before, [changed] * 3, numbers, [0],
    )
    assert outcome[0] is expected
    assert len(reads) == 3, "首次读到题号不符时不能抢先确认成功"


def test_swap_still_confirms_when_number_is_unreadable(monkeypatch):
    before, changed = card(), card(270)
    outcome, _ = confirm(monkeypatch, before, [changed] * 2, [None], [0])
    assert outcome[0] is True, "题号读不出时仍允许当前题干变化加库存扣减兜底"
    assert outcome[2] is changed
