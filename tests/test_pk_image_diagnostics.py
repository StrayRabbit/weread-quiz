"""图片模型只记录结构诊断，不记录回复原文或敏感请求内容。"""

import pytest
from PIL import Image

from weread_quiz.errors import AnswerUnavailable
from weread_quiz.vision import VisionService


def service(monkeypatch, body, sink):
    vision = VisionService("http://127.0.0.1", "not-a-real-key", "test", timing_sink=sink)
    monkeypatch.setattr(vision, "recognize", lambda *args, **kwargs: body)
    return vision


def test_non_question_image_classification_is_logged_without_response_body(monkeypatch):
    logs = []
    vision = service(monkeypatch, '{"screen":"other","note":"PRIVATE_RESPONSE"}', logs.append)
    question = vision.read_question(Image.new("RGB", (430, 788)))
    assert question.screen == "other"
    assert any("kind=question" in text and "screen=other" in text for text in logs)
    assert any("options=0" in text and "response_chars=" in text for text in logs)
    assert all("PRIVATE_RESPONSE" not in text and "not-a-real-key" not in text for text in logs)


def test_invalid_image_response_has_safe_diagnostics(monkeypatch):
    logs = []
    vision = service(monkeypatch, "PRIVATE_RESPONSE", logs.append)
    with pytest.raises(AnswerUnavailable):
        vision.read_question(Image.new("RGB", (430, 788)))
    assert any("kind=question" in text and "status=invalid_response" in text for text in logs)
    assert all("PRIVATE_RESPONSE" not in text for text in logs)


def test_result_diagnostics_explain_non_string_score(monkeypatch):
    logs = []
    vision = service(monkeypatch, '{"result":"","title":"PRIVATE_TITLE","score":5}', logs.append)
    result = vision.read_result_detail(Image.new("RGB", (430, 788)))
    assert result["score"] == ""
    assert any("kind=result" in text and "score_type=int" in text for text in logs)
    assert all("PRIVATE_TITLE" not in text for text in logs)


@pytest.mark.parametrize("body,status", [("PRIVATE_RESPONSE", "missing_json"), ("{broken}", "invalid_json")])
def test_invalid_result_json_is_diagnosed(monkeypatch, body, status):
    logs = []
    vision = service(monkeypatch, body, logs.append)
    assert vision.read_result_detail(Image.new("RGB", (430, 788))) == {}
    assert any(f"status={status}" in text for text in logs)
    assert all("PRIVATE_RESPONSE" not in text for text in logs)


def test_diagnostic_sink_failure_does_not_break_image_read(monkeypatch):
    def broken(message):
        raise RuntimeError("log unavailable")

    vision = service(monkeypatch, '{"screen":"other"}', broken)
    assert vision.read_question(Image.new("RGB", (430, 788))).screen == "other"
