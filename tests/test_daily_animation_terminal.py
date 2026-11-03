"""Daily 选项动画、单条按钮带终态及真实 OCR 坐标类型回归。"""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from weread_quiz import daily, daily_screen
from weread_quiz.daily_ocr import DailyOcrRead, DailyOcrReader, _is_point_sequence
from weread_quiz.models import Rect
from test_daily_pipeline import Clock


def question_frame(option_width=8):
    image = Image.new("RGB", (430, 788), (10, 20, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((170, 202, 240, 207), fill="white")
    draw.rectangle((140, 240, 280, 250), fill="white")
    for top in (310, 380, 450):
        draw.rectangle((40, top, 390, top + 44), fill=(42, 59, 74))
        draw.rectangle((200, top + 15, 200 + option_width, top + 25), fill="white")
    return image


def configuration(**kwargs):
    return daily.DailyConfig(Rect(0, 0, 430, 788),
                             kwargs.pop("vision", SimpleNamespace()),
                             verbose=False, **kwargs)


@pytest.fixture
def fake_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(daily.time, "monotonic", clock)
    monkeypatch.setattr(daily.time, "sleep", clock.sleep)
    return clock


def test_stability_rejects_changing_option_text():
    before, after = question_frame(), question_frame(option_width=14)
    assert daily_screen.stem_change_count(before, after) == 0
    assert not daily_screen.is_question_stable(before, after)
    assert not daily_screen.is_same_question(before, after)


def test_stability_accepts_identical_option_text():
    image = question_frame()
    assert daily_screen.is_question_stable(image, image.copy())
    assert daily_screen.is_same_question(image, image.copy())


def test_first_question_waits_for_option_text_to_stop_changing(monkeypatch, fake_clock):
    frames = [question_frame(width) for width in (8, 14, 20, 20)]
    captures = []
    reads = []

    def grab(region):
        image = frames[len(captures)]
        captures.append(image)
        return image

    def local(image):
        reads.append(image)
        return DailyOcrRead(stem="题目", options=["甲", "乙", "丙"],
                            confidences=[0.99] * 3, button_count=3,
                            has_progress=True, progress_no=1)

    monkeypatch.setattr(daily, "grab_screen", grab)
    vision = SimpleNamespace(read_question_local=local, local_ocr_available=True)
    image, question = daily.wait_for_question(configuration(vision=vision), timeout=2)
    assert image is frames[-1] and question is None
    assert len(captures) == 4
    assert reads == [frames[-1]]


def terminal_frame():
    image = Image.new("RGB", (430, 788), (10, 20, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 240, 300, 250), fill="white")
    draw.rectangle((84, 300, 345, 344), fill=(42, 59, 74))
    return image


def test_single_button_terminal_is_confirmed_once(monkeypatch, fake_clock):
    from weread_quiz.daily_buttons import _detect_any_buttons, find_blue_gray_bands

    image = terminal_frame()
    assert _detect_any_buttons(image)
    assert len(find_blue_gray_bands(np.asarray(image), 295, 540)) == 1
    assert not daily.has_progress_line(image)
    assert not daily.has_answer_reveal(image)
    calls = []

    def read_terminal(frame):
        calls.append(frame)
        return SimpleNamespace(kind="revive", evidence="可以通过以下方式复活")

    monkeypatch.setattr(daily, "grab_screen", lambda region: image.copy())
    cfg = configuration(ocr=SimpleNamespace(read_terminal=read_terminal))
    outcome = daily.wait_for_next_question(cfg, question_frame())
    assert outcome.status == "revive"
    assert "可以通过以下方式复活" in outcome.reason
    assert len(calls) == 1
    assert fake_clock.now - 100.0 < 3.0


def test_single_button_without_terminal_text_is_not_called_wrong(monkeypatch, fake_clock):
    image = terminal_frame()
    calls = []

    def read_terminal(frame):
        calls.append(frame)
        return None

    monkeypatch.setattr(daily, "grab_screen", lambda region: image.copy())
    cfg = configuration(ocr=SimpleNamespace(read_terminal=read_terminal))
    outcome = daily.wait_for_next_question(cfg, question_frame())
    assert outcome.status == "interrupted"
    assert len(calls) == 1


def test_single_button_terminal_verdict_rejected_if_frame_changes(monkeypatch):
    image = terminal_frame()
    changed = image.copy()
    ImageDraw.Draw(changed).rectangle((120, 250, 150, 260), fill="white")
    terminal = SimpleNamespace(kind="revive", evidence="可以通过以下方式复活")
    monkeypatch.setattr(daily, "grab_screen", lambda region: changed)
    cfg = configuration(ocr=SimpleNamespace(read_terminal=lambda frame: terminal))
    outcome, fresh = daily._confirm_terminal(cfg, image)
    assert outcome is None and fresh is changed


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_terminal_accepts_numpy_real_coordinates(dtype):
    box = np.array([[120, 235], [310, 235], [310, 255], [120, 255]], dtype=dtype)
    reader = DailyOcrReader()
    reader._engine = lambda *args, **kwargs: SimpleNamespace(
        boxes=np.array([box]), txts=["可以通过以下方式复活"], scores=[0.99],
    )
    terminal = reader.read_terminal(terminal_frame())
    assert terminal is not None
    assert terminal.kind == "revive"
    assert terminal.evidence == "可以通过以下方式复活"


@pytest.mark.parametrize("value", [True, np.bool_(True), "120", 120 + 0j,
                                    float("nan"), float("inf"), np.float32("nan"),
                                    np.float32("inf")])
def test_terminal_coordinates_reject_invalid_scalars(value):
    box = [[value, 235], [310, 235], [310, 255], [120, 255]]
    assert not _is_point_sequence(box)


@pytest.mark.parametrize("box", [None, [], [[1, 2]] * 3, [[1]] * 4, "1234"])
def test_terminal_coordinates_reject_malformed_boxes(box):
    assert not _is_point_sequence(box)


@pytest.mark.parametrize("text,score,top", [
    ("可以通过以下方式复活", 0.8, 235),
    ("可以通过以下方式复活", 0.99, 320),
    ("复活", 0.99, 235),
    ("可以通过以下方式复活", float("nan"), 235),
    ("可以通过以下方式复活", float("inf"), 235),
])
def test_numpy_terminal_read_keeps_confidence_phrase_and_position_guards(text, score, top):
    box = np.array([[120, top], [310, top], [310, top + 20], [120, top + 20]],
                   dtype=np.float32)
    reader = DailyOcrReader()
    reader._engine = lambda *args, **kwargs: SimpleNamespace(
        boxes=np.array([box]), txts=[text], scores=[score],
    )
    assert reader.read_terminal(terminal_frame()) is None
