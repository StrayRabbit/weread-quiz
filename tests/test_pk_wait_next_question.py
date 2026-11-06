"""答完一题后等下一题：答错、换题转场都不能当成对局结束，只有结算页才结束。"""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from weread_quiz import pk


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(pk.time, "monotonic", value)
    monkeypatch.setattr(pk.time, "sleep", value.sleep)
    return value


# 各状态下的界面特征：
#   按钮 / 答题卡片 / 比分行
#   question、next   有 / 有 / 有   —— 正常答题页
#   answered        无 / 有 / 有   —— 答错：按钮染红，青色边框检测不到
#   nocard          无 / 无 / 有   —— 换题转场：卡片淡出，比分行不动
#   blank           无 / 无 / 无   —— 整页渐隐/空白
#   settlement      无 / 无 / 无   —— 结算页（另用页面上才有的字确认）
_STATES = {
    "question": (True, True, True),
    "next": (True, True, True),
    "answered": (False, True, True),
    "nocard": (False, False, True),
    "blank": (False, False, False),
    "settlement": (False, False, False),
}


def _frame(state: str) -> Image.Image:
    """造一张带状态标签的假帧。"""
    image = Image.new("RGB", (430, 788), (10, 20, 40))
    ImageDraw.Draw(image).text((10, 10), state, fill="white")
    image.info["state"] = state
    return image


def _install(monkeypatch, states):
    """按状态序列喂帧，并按状态给出各像素判据的答案。"""
    frames = [_frame(state) for state in states]
    index = {"i": 0}

    def grab(_region):
        frame = frames[min(index["i"], len(frames) - 1)]
        index["i"] += 1
        return frame

    def state_of(image):
        return image.info.get("state", "")

    def flags(image):
        return _STATES[state_of(image)]

    monkeypatch.setattr(pk, "grab_screen", grab)
    monkeypatch.setattr(pk, "is_question_screen", lambda image: flags(image)[0])
    monkeypatch.setattr(pk, "looks_like_question_template",
                        lambda image: flags(image)[1])
    monkeypatch.setattr(pk, "has_score_row", lambda image: flags(image)[2])
    monkeypatch.setattr(pk, "has_answer_reveal", lambda _image: False)
    monkeypatch.setattr(pk, "is_new_question",
                        lambda _before, after: state_of(after) == "next")
    # 结算页的确认靠读页面上才有的字；这里只让「settlement」状态读得到。
    monkeypatch.setattr(pk, "_is_settlement_page",
                        lambda image, _ocr: state_of(image) == "settlement")
    return frames


def test_wrong_answer_keeps_waiting_for_next_question(monkeypatch):
    """答错时按钮变红、检测不到按钮 —— 不能当成对局结束，要接着等下一题。"""
    _install(monkeypatch, ["answered", "answered", "next"])
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is not None
    assert current.info["state"] == "next"


def test_transition_without_card_is_not_the_end(monkeypatch):
    """换题转场时答题卡片淡出 —— 卡片没了不等于对局结束（比分行还在）。"""
    _install(monkeypatch, ["nocard", "answered", "next"])
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is not None
    assert current.info["state"] == "next"


def test_blank_page_is_not_the_end(monkeypatch):
    """整页渐隐/空白时读不出结算字样 —— 继续等，不能收工。"""
    _install(monkeypatch, ["blank", "answered", "next"])
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is not None
    assert current.info["state"] == "next"


def test_settlement_page_ends_the_match(monkeypatch):
    """结算页：卡片和比分行都没了，页面上又读得出结算字样 —— 这才结束。"""
    _install(monkeypatch, ["settlement"])
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is None


def test_wait_covers_the_whole_question_window(monkeypatch, clock):
    """下一题迟迟不来时，要等满整个题目窗口（13 秒），不是 4 秒。"""
    _install(monkeypatch, ["answered"])
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is None, "等不到下一题只能收工"
    assert clock.now - 100.0 >= pk._ANSWERED_WAIT_MAX - 0.1
