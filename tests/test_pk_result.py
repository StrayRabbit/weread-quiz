"""结算优先复用本地 OCR；胜负只认比赛比分，奖励积分档位不参与判胜负。"""

from types import SimpleNamespace

import pytest
from PIL import Image

from weread_quiz import pk
from weread_quiz.result import read_result, verdict_from_score


def line(text, x=150, y=440, width=70, score=0.99):
    return ([[x, y], [x + width, y], [x + width, y + 20], [x, y + 20]], text, score)


class Ocr:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def _engine_run(self, image):
        self.calls += 1
        return self.rows


class Model:
    def __init__(self, detail=None):
        self.detail = detail or {"result": "", "title": "", "score": ""}
        self.calls = 0

    def read_result_detail(self, image):
        self.calls += 1
        return self.detail


def frame():
    return Image.new("RGB", (430, 788))


def test_local_reward_skips_image_request_and_preserves_title():
    ocr = Ocr([line("进击举人·4等", y=100), line("获得积分"), line("+5", x=240)])
    model = Model()
    logs = []
    result = read_result(frame(), vision=model, ocr=ocr, recorder=SimpleNamespace(log=logs.append))
    assert result == {"result": "", "title": "进击举人·4等", "score": "+5"}
    assert ocr.calls == 1
    assert model.calls == 0
    assert any("本地 OCR" in text and "+5" in text for text in logs)


@pytest.mark.parametrize("tier", ["1", "3", "4", "5"])
def test_reward_tier_never_implies_a_verdict(tier):
    """积分档位与输赢无关 —— +1 的那一局是 125:60 的大胜。

    拿积分反推胜负会把赢了的一局报成落败。
    """
    result = read_result(frame(), ocr=Ocr([line(f"获得积分 +{tier}")]))
    assert result["score"] == f"+{tier}"
    assert result["result"] == ""


@pytest.mark.parametrize("rows", [
    [line("+5")],
    [line("获得积分"), line("+5", y=150)],
    [line("获得积分"), line("+5", x=20)],
    [line("获得积分", score=0.5), line("+5")],
    [line("获得积分"), line("+5", x=240, score=0.5)],
    [line("获得积分 +10")],
    [line("获得积分"), line("+5", x=240), line("+1", x=280)],
    [line("130 VS 55"), line("★★★★")],
    [],
])
def test_unreliable_local_evidence_falls_back_to_model(rows):
    model = Model({"result": "", "title": "模型称号", "score": "+3"})
    result = read_result(frame(), vision=model, ocr=Ocr(rows))
    assert result == {"result": "", "title": "模型称号", "score": "+3"}
    assert model.calls == 1


def test_local_ocr_exception_does_not_block_model_fallback():
    class Broken:
        def _engine_run(self, image):
            raise RuntimeError("unavailable")

    model = Model({"result": "VICTORY", "score": "+5"})
    assert read_result(frame(), vision=model, ocr=Broken())["result"] == "胜利"
    assert model.calls == 1


def test_conflicting_explicit_outcome_and_score_does_not_infer_victory():
    """页面写着失败、比分却是领先的 —— 两个都不硬报。"""
    result = read_result(frame(), ocr=Ocr([line("DEFEAT", y=200)]), score=(125, 60))
    assert result["result"] == ""
    assert result["title"] == ""


def test_without_local_reader_preserves_existing_model_contract():
    model = Model({"result": "", "title": "称号", "score": "+5"})
    assert read_result(frame(), vision=model) == {"result": "", "title": "称号", "score": "+5"}


@pytest.mark.parametrize("score, verdict", [
    ((125, 60), "胜利"),
    ((60, 125), "落败"),
    ((60, 60), "平局"),
    ((None, 60), ""),
    (None, ""),
])
def test_verdict_comes_from_the_score(score, verdict):
    assert verdict_from_score(score) == verdict


def test_score_decides_the_verdict_even_with_a_reward():
    """有奖励积分时也一样 —— 胜负看比分，不看积分。"""
    ocr = Ocr([line("获得积分"), line("+1", x=240)])
    result = read_result(frame(), ocr=ocr, score=(125, 60))
    assert result["result"] == "胜利"
    assert result["score"] == "+1"


def test_read_final_reuses_existing_hybrid_ocr(monkeypatch):
    image = frame()
    ocr = Ocr([line("获得积分 +5")])
    model = Model()
    model._ocr = ocr
    monkeypatch.setattr(pk.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pk, "grab_screen", lambda region: image)
    report = pk.PKReport()
    pk._read_final((0, 0, 430, 788), report, model, None, False)
    assert report.final["score"] == "+5"
    assert ocr.calls == 1
    assert model.calls == 0
    # 没有比分就**不判胜负** —— 不能拿 +5 反推胜利。
    assert report.outcome == "unknown"
    assert report.final["result"] == ""


def test_read_final_uses_the_score_from_the_last_question_card(monkeypatch):
    """结算页上没有比分 —— 胜负来自结算前最后一张答题卡片。"""
    image = frame()
    ocr = Ocr([line("获得积分 +1")])
    model = Model()
    model._ocr = ocr
    monkeypatch.setattr(pk.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pk, "grab_screen", lambda region: image)
    context = pk.PKCaptureContext()
    context.final_frame = object()
    monkeypatch.setattr(pk, "_read_final_score", lambda ctx, o: (125, 60))
    report = pk.PKReport()
    pk._read_final((0, 0, 430, 788), report, model, None, False,
                   capture_context=context)
    assert report.score == (125, 60)
    assert report.final["result"] == "胜利"
    assert report.outcome == "victory"
