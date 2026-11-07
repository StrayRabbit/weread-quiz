"""答完一题后等下一题：答错、换题转场都不能当成对局结束，只有结算页才结束。"""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from weread_quiz.pk import runner as pk


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
    # 「再来一局」「炫耀一下」同样会被几何检测当成两个按钮。
    "settlement_buttons": (True, False, False),
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
    # 结算页的确认靠读页面上才有的字，只让两种结算状态读得到。
    monkeypatch.setattr(pk, "_is_settlement_page",
                        lambda image, _ocr: state_of(image) in {"settlement", "settlement_buttons"})
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


@pytest.mark.parametrize("states", [
    ["settlement_buttons"],
    ["blank", "settlement_buttons"],
])
def test_settlement_buttons_are_not_a_new_question(monkeypatch, clock, states):
    """真实结算页有两个青色按钮、题干区也变了，仍须确认结算而非交给模型。"""
    _install(monkeypatch, states)
    monkeypatch.setattr(pk, "is_new_question", lambda _before, _after: True)
    checked = []

    def settlement(image, _ocr):
        checked.append(image.info["state"])
        return image.info["state"] == "settlement_buttons"

    monkeypatch.setattr(pk, "_is_settlement_page", settlement)
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is None, "结算按钮不能触发第 11 题"
    assert "settlement_buttons" in checked, "必须运行本地结算确认"
    assert clock.now - 100.0 < 1.0, "已确认结算后不能等满题目窗口"


@pytest.mark.parametrize("card", [True, None])
def test_real_next_question_keeps_template_unavailable_fallback(monkeypatch, card):
    _install(monkeypatch, ["next"])
    monkeypatch.setattr(pk, "looks_like_question_template", lambda _image: card)
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is not None
    assert current.info["state"] == "next"


def test_unconfirmed_settlement_buttons_are_not_returned_on_timeout(monkeypatch, clock):
    """OCR 暂时读不出结算字样时可以继续等，但不能到期后又交出伪下一题。"""
    _install(monkeypatch, ["settlement_buttons"])
    monkeypatch.setattr(pk, "is_new_question", lambda _before, _after: True)
    monkeypatch.setattr(pk, "_is_settlement_page", lambda *_args: False)
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is None
    assert clock.now - 100.0 >= pk._ANSWERED_WAIT_MAX - 0.1


def test_unchanged_question_is_not_returned_on_timeout(monkeypatch, clock):
    _install(monkeypatch, ["question"])
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is None, "到期后仍是原题，不得再尝试答一次"


def test_wait_covers_the_whole_question_window(monkeypatch, clock):
    """下一题迟迟不来时，要等满整个题目窗口（13 秒），不是 4 秒。"""
    _install(monkeypatch, ["answered"])
    current, _ = pk.wait_for_next_question(None, _frame("question"), delay=0)
    assert current is None, "等不到下一题只能收工"
    assert clock.now - 100.0 >= pk._ANSWERED_WAIT_MAX - 0.1
