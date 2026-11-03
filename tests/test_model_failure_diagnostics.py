"""请求预算、图片解析结构诊断与 daily 蒙答日志回归。"""
import time

import pytest
from PIL import Image

from weread_quiz.errors import AnswerUnavailable, RecognitionTimeout
from weread_quiz.vision import VisionService


@pytest.mark.parametrize("requested", [0.1, 1.0, 3.0, 6.0])
def test_http_read_timeout_uses_full_requested_budget(requested):
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test")
    options = vision._timeout_option(requested)
    assert options["timeout"].read == pytest.approx(requested)
    assert options["timeout"].connect == pytest.approx(requested)


def test_http_timeout_is_clamped_by_remaining_deadline(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test")
    options = vision._timeout_option(3.0, deadline=101.5)
    assert options["timeout"].read == pytest.approx(1.5)


def test_http_timeout_keeps_service_ceiling():
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test", timeout=2.0)
    options = vision._timeout_option(3.0, deadline=time.monotonic() + 10.0)
    assert options["timeout"].read == pytest.approx(2.0)


def test_expired_http_deadline_is_rejected_before_request(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test")
    with pytest.raises(RecognitionTimeout):
        vision._timeout_option(3.0, deadline=100.0)


def test_unspecified_http_timeout_keeps_client_default():
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test")
    assert vision._timeout_option(None) == {}


@pytest.mark.parametrize("body,reason", [
    ("", "empty_response"),
    ("PRIVATE_RESPONSE", "missing_json"),
    ("{PRIVATE_RESPONSE}", "invalid_json"),
    ('[{"stem":"PRIVATE_RESPONSE"}]', "invalid_shape"),
    ('{"options":["PRIVATE_RESPONSE"],"answer":0}', "missing_stem"),
    ('{"stem":"","options":[],"answer":-1}', "empty_stem"),
    ('{"stem":"PRIVATE_RESPONSE","answer":0}', "missing_options"),
    ('{"stem":"PRIVATE_RESPONSE","options":[],"answer":0}', "empty_options"),
    ('{"stem":"PRIVATE_RESPONSE","options":"not a list","answer":0}', "invalid_options"),
    ('{"stem":"PRIVATE_RESPONSE","options":[""],"answer":0}', "empty_option"),
])
def test_image_parse_failure_has_safe_specific_reason(monkeypatch, body, reason):
    from weread_quiz.vision import parse_vision_response

    logs = []
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test", timing_sink=logs.append)
    monkeypatch.setattr(vision, "recognize", lambda *args, **kwargs: body)
    with pytest.raises(AnswerUnavailable):
        parse_vision_response(body)
    with pytest.raises(AnswerUnavailable):
        vision.read_question(Image.new("RGB", (430, 788)), deadline=time.monotonic() + 6.0)
    assert any(f"parse_reason={reason}" in line for line in logs)
    assert all("PRIVATE_RESPONSE" not in line and "PRIVATE_KEY" not in line for line in logs)
    assert all("content_empty=True" not in line for line in logs)


def test_image_parse_failure_reports_whole_read_elapsed_time(monkeypatch):
    logs = []
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test", timing_sink=logs.append)

    def recognize(*args, **kwargs):
        now[0] += 2.5
        return '{"stem":"","options":[],"answer":-1}'

    monkeypatch.setattr(vision, "recognize", recognize)
    with pytest.raises(AnswerUnavailable, match="invalid_response"):
        vision.read_question(Image.new("RGB", (430, 788)), deadline=106.0)
    assert any("elapsed_ms=2500" in line and "remaining_ms=3500" in line for line in logs)
    assert not any("elapsed=0.000s" in line for line in logs)


def test_parse_reason_never_uses_arbitrary_exception_text(monkeypatch):
    import weread_quiz.vision as module

    logs = []
    vision = VisionService("http://example.invalid", "PRIVATE_KEY", "test", timing_sink=logs.append)
    monkeypatch.setattr(vision, "recognize", lambda *args, **kwargs: "PRIVATE_RESPONSE")

    def broken_parser(body):
        raise ValueError("PRIVATE_EXCEPTION")

    monkeypatch.setattr(module, "parse_vision_response", broken_parser)
    with pytest.raises(AnswerUnavailable):
        vision.read_question(Image.new("RGB", (430, 788)), deadline=time.monotonic() + 6.0)
    assert any("parse_reason=unexpected_parse_error" in line for line in logs)
    assert all("PRIVATE_EXCEPTION" not in line and "PRIVATE_RESPONSE" not in line for line in logs)
