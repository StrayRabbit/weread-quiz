"""决策模型（读不了图）时，等待阶段不能一声不吭。

决策模型只接文本路径 —— 本地 OCR 读不出题时没有能读图的模型兜底。
等答题页本身是对的（可能还在匹配中），但**干等不出声**会让用户
以为程序死了。
"""

from __future__ import annotations

from weread_quiz.pk import runner as pk


class Recorder:
    def __init__(self):
        self.lines = []

    def log(self, message):
        self.lines.append(message)

    def shot(self, *args, **kwargs):
        return "shot.png"


class DecisionOnlyVision:
    """决策模型替身：能判下标，读不了图。"""

    usage = None
    last_question_usage = None
    supports_image = False

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        raise AssertionError("本用例不该调模型")


class VisionWithImage(DecisionOnlyVision):
    supports_image = True


def _frame():
    from PIL import Image

    return Image.new("RGB", (430, 788), (20, 35, 70))


def _reject_everything(monkeypatch):
    """本地判据一律不放行 —— 模拟"还没进答题页"。"""
    monkeypatch.setattr(pk, "_POLL_INTERVAL", 0)
    monkeypatch.setattr(pk, "grab_screen", lambda region: _frame())
    monkeypatch.setattr(pk, "looks_like_question_template", lambda image: False)
    monkeypatch.setattr(pk, "has_rendered_stem", lambda image: False)
    monkeypatch.setattr(pk, "_read_progress_line", lambda image, ocr: (False, ""))


def test_wait_says_out_when_the_model_cannot_read_images(monkeypatch):
    """读不了图的模型：等待期间要出声，并说明没有兜底。"""
    _reject_everything(monkeypatch)
    recorder = Recorder()
    pk.wait_for_question(object(), DecisionOnlyVision(), timeout=0.5,
                         verbose=False, recorder=recorder)
    notes = [line for line in recorder.lines if "等待答题页" in line]
    assert notes, recorder.lines
    assert any("读不了图" in line for line in notes), notes


def test_wait_note_omits_the_caveat_when_a_vision_model_exists(monkeypatch):
    """有能读图的模型时，不该说"没有兜底"。"""
    _reject_everything(monkeypatch)
    recorder = Recorder()
    pk.wait_for_question(object(), VisionWithImage(), timeout=0.5,
                         verbose=False, recorder=recorder)
    notes = [line for line in recorder.lines if "等待答题页" in line]
    assert notes, recorder.lines
    assert not any("读不了图" in line for line in notes), notes
