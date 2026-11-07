"""PK 真实预热仅在匹配前执行，不能消耗正在进行的题目时间。"""

from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from weread_quiz import advance
from weread_quiz import cli
from weread_quiz.pk import runner as pk
from weread_quiz.errors import RecognitionTimeout
from weread_quiz.route import ROUTE_PK, get_route
from weread_quiz.vision import VisionService


class Recorder:
    def __init__(self):
        self.lines = []

    def log(self, message):
        self.lines.append(message)


def _frame():
    """一张不是空白帧的画面（空白帧会被当成「窗口还没画出来」跳过）。"""
    image = Image.new("RGB", (430, 788), "white")
    image.paste((0, 0, 0), (10, 10, 60, 40))
    return image


@pytest.mark.parametrize("mode", ["ok", "invalid", "timeout"])
def test_real_warmup_is_one_bounded_request_and_failure_is_nonfatal(mode):
    requests = []

    def handle(request):
        requests.append(request)
        assert 0 < request.extensions["timeout"]["read"] <= 8.0
        if mode == "timeout":
            raise httpx.ReadTimeout("test", request=request)
        content = '{"answer_index":0}' if mode == "ok" else "not an answer"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    recorder = Recorder()
    with VisionService("https://example.invalid/v1", "test", "test", retries=3) as vision:
        vision._client = httpx.Client(transport=httpx.MockTransport(handle))
        cli._warmup_pk_model(vision, recorder)
    assert len(requests) == 1
    assert any("模型预热" in line and "ms" in line for line in recorder.lines)
    assert any(("成功" if mode == "ok" else "失败") in line for line in recorder.lines)


@pytest.mark.parametrize("already_question", [False, True])
def test_warmup_before_match_rechecks_button_and_skips_active_question(monkeypatch, already_question):
    events = []
    monkeypatch.setattr(advance, "grab_screen", lambda region: _frame())
    monkeypatch.setattr(cli, "_make_pk_question_judge", lambda **kw: lambda image: already_question)
    monkeypatch.setattr(advance, "_read_all_lines", lambda *a: [
        ("自动匹配", 1.0, ((10, 10), (60, 10), (60, 30), (10, 30))),
    ])
    monkeypatch.setattr(cli, "_warmup_pk_model", lambda *a: events.append("warmup"))
    monkeypatch.setattr(advance, "_recheck", lambda *a: events.append("recheck") or "ok")
    monkeypatch.setattr(advance, "click_box", lambda *a: events.append("click"))
    assert cli._advance_to_ready(ROUTE_PK, None, object(), Recorder())
    assert events == ([] if already_question else ["warmup", "recheck", "click"])


def test_match_warmup_runs_once_even_when_button_recheck_fails(monkeypatch):
    calls = []

    def fake_advance(**kwargs):
        callback = kwargs["before_click"]
        callback("问答PK")
        callback("自动匹配")
        callback("自动匹配")
        return True

    monkeypatch.setattr(cli, "advance_to_question", fake_advance)
    monkeypatch.setattr(cli, "_warmup_pk_model", lambda *a: calls.append(1))
    assert cli._advance_to_ready(ROUTE_PK, None, object(), Recorder())
    assert calls == [1]


def test_daily_does_not_install_model_warmup(monkeypatch):
    def fake_advance(**kwargs):
        assert "before_click" not in kwargs
        return True

    monkeypatch.setattr(cli, "advance_to_question", fake_advance)
    assert cli._advance_to_ready(get_route("daily-days"), None, object(), Recorder())


def test_preread_accepts_answer_after_four_seconds_within_question_budget(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(pk.time, "monotonic", lambda: clock[0])
    answer = object()

    def read(*args, **kwargs):
        clock[0] += 4.0
        return answer

    vision = SimpleNamespace(answer_from_ocr_read=read)
    assert pk._answer_from_pre_read(vision, object(), deadline=1006.0) is answer


def test_stale_preread_retains_existing_short_request_budget(monkeypatch):
    monkeypatch.setattr(pk.time, "monotonic", lambda: 1000.0)
    seen = {}

    def read(*args, **kwargs):
        seen.update(kwargs)
        return None

    assert pk._answer_from_pre_read(
        SimpleNamespace(answer_from_ocr_read=read), object(), deadline=999.0,
    ) is None
    assert seen == {"timeout": 3.0, "deadline": 1003.0}


def test_preread_response_after_deadline_is_rejected(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(pk.time, "monotonic", lambda: clock[0])

    def read(*args, **kwargs):
        clock[0] = 1006.1
        return object()

    with pytest.raises(RecognitionTimeout):
        pk._answer_from_pre_read(SimpleNamespace(answer_from_ocr_read=read), object(), deadline=1006.0)
