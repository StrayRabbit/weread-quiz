"""PK 文本用足预算；超时直接蒙答，提前无效响应或 OCR 判否才走图片。"""
import pytest
from PIL import Image

from weread_quiz.errors import AnswerUnavailable, RecognitionTimeout
from weread_quiz.models import Rect
from weread_quiz.ocr_core import OcrDecision, OcrRead
from weread_quiz.pk import runner as pk
from weread_quiz.pk.hybrid import HybridQuestionReader
from weread_quiz.vision import VisionQuestion


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Ocr:
    def __init__(self, clock, seconds=12):
        self.clock = clock
        self.item = OcrRead("题目", ["甲", "乙"], [0.99, 0.99], 2,
                            has_progress=True, progress_text="第1题·共10题")
        self.item.countdown_seconds = seconds
        self.accepted = True

    def read(self, image):
        self.clock.now += 0.85
        return self.item

    def decide_merged(self, whole, strip):
        return OcrDecision(self.accepted, "" if self.accepted else "置信度不足", whole)


class Model:
    usage = None
    last_question_usage = None

    def __init__(self, clock, *, supports_image=True, overrun=0.0,
                 text_answer=None, image_fails=False, text_cost=None):
        self.clock = clock
        self.supports_image = supports_image
        self.overrun = overrun
        self.text_answer = text_answer
        self.image_fails = image_fails
        self.text_cost = text_cost
        self.text_calls = []
        self.image_calls = []

    def answer_from_text(self, prompt, count, *, timeout=None, deadline=None):
        self.text_calls.append((timeout, deadline))
        cost = timeout + self.overrun if self.text_answer is None else 0.1
        self.clock.now += cost if self.text_cost is None else self.text_cost
        return self.text_answer

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        assert self.supports_image, "没有图片能力时不该请求图片"
        self.image_calls.append((deadline, deadline - self.clock.now))
        self.clock.now += 1.0
        if self.image_fails:
            raise AnswerUnavailable("image_request_failed")
        return VisionQuestion(stem="题目", options=["甲", "乙"], answer_index=1)


@pytest.fixture
def scene(monkeypatch):
    clock = Clock()
    image = Image.new("RGB", (430, 788))
    state = {"same": True, "clicks": []}
    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk, "grab_screen", lambda region: image)
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: [
        Rect(60, 380, 310, 44), Rect(60, 450, 310, 44)])
    monkeypatch.setattr(pk, "is_same_question", lambda *a: state["same"])
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **kw: state["clicks"].append(box.center_y))
    monkeypatch.setattr(pk, "read_question_number", lambda *a, **kw: (2, 10, "第2题·共10题"))
    return clock, image, state


def answer(scene, entry, model, *, seconds=12, text_timeout=12.0):
    clock, image, _ = scene
    ocr = Ocr(clock, seconds)
    reader = HybridQuestionReader(model, ocr, verbose=False, text_timeout=text_timeout)
    options = {}
    if entry == "preread":
        options["ocr_read"] = ocr.read(image)
    result = pk.answer_one_question(None, reader, image=image, captured_at=1000.0,
        capture_context=pk.PKCaptureContext(captured_at=1000.0), verbose=False, **options)
    return result, reader


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
@pytest.mark.parametrize("overrun", [0.0, 0.02, 0.2])
def test_text_timeout_uses_full_budget_then_guesses_without_image(scene, entry, overrun):
    clock, _, state = scene
    model = Model(clock, overrun=overrun)
    result, _ = answer(scene, entry, model)
    assert model.text_calls == [(pytest.approx(10.35), pytest.approx(1011.2))]
    assert model.image_calls == [], "文本超时后不再串行请求图片"
    assert result["answer"] == 0 and result["guessed"]
    assert len(state["clicks"]) == 1
    assert clock.now + 0.28 < 1012.0, "超时退出后仍须给实际蒙答点击留时间"


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_eight_second_text_answer_is_not_cut_off_for_image_reserve(scene, entry):
    _, _, state = scene
    model = Model(scene[0], text_answer=1, text_cost=8.0)
    result, _ = answer(scene, entry, model)
    assert model.text_calls == [(pytest.approx(10.35), pytest.approx(1011.2))]
    assert model.image_calls == []
    assert result["answer"] == 1 and not result["guessed"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_fast_invalid_text_result_can_still_fall_back_to_image(scene, entry):
    clock, _, state = scene
    model = Model(clock, text_cost=0.5)
    result, _ = answer(scene, entry, model)
    assert len(model.text_calls) == 1
    assert model.image_calls == [(pytest.approx(1011.2), pytest.approx(9.85))]
    assert result["answer"] == 1 and not result["guessed"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_fast_invalid_response_without_image_budget_guesses(scene, entry):
    clock, _, state = scene
    model = Model(clock, text_cost=0.5)
    result, _ = answer(scene, entry, model, seconds=4)
    assert len(model.text_calls) == 1
    assert model.image_calls == []
    assert result["guessed"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_short_question_budget_is_given_to_text_instead_of_skipping_it(scene, entry):
    clock, _, state = scene
    model = Model(clock, text_answer=1)
    result, _ = answer(scene, entry, model, seconds=6)
    assert model.text_calls == [(pytest.approx(4.35), pytest.approx(1005.2))]
    assert model.image_calls == []
    assert result["answer"] == 1 and not result["guessed"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_configured_text_timeout_is_still_an_upper_limit(scene, entry):
    clock, _, _ = scene
    model = Model(clock, supports_image=False, text_answer=1)
    result, _ = answer(scene, entry, model, text_timeout=6.5)
    assert model.text_calls == [(6.5, pytest.approx(1007.35))]
    assert model.image_calls == []
    assert not result["guessed"]


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_configured_timeout_does_not_trigger_image_even_with_time_left(scene, entry):
    clock, _, state = scene
    model = Model(clock)
    result, _ = answer(scene, entry, model, text_timeout=2.0)
    assert model.text_calls == [(2.0, pytest.approx(1002.85))]
    assert model.image_calls == [], "配置上限到时也是超时，不能改走图片"
    assert result["guessed"] and len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_no_image_capability_never_attempts_image_after_text_failure(scene, entry):
    clock, _, state = scene
    model = Model(clock, supports_image=False, text_cost=1.0)
    result, _ = answer(scene, entry, model)
    assert model.text_calls == [(pytest.approx(10.35), pytest.approx(1011.2))]
    assert model.image_calls == []
    assert result["guessed"] and result["options"] == ["甲", "乙"]
    assert len(state["clicks"]) == 1


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_legal_text_answer_does_not_trigger_image_review(scene, entry):
    clock, _, _ = scene
    model = Model(clock, text_answer=1)
    result, _ = answer(scene, entry, model)
    assert len(model.text_calls) == 1
    assert model.image_calls == []
    assert result["answer"] == 1 and not result["guessed"]


@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_failed_models_keep_same_question_ocr_for_guess_record(scene, entry):
    clock, _, state = scene
    model = Model(clock, text_cost=0.5, image_fails=True)
    result, _ = answer(scene, entry, model)
    assert result["guessed"] and not result["skipped"]
    assert result["stem"] == "题目"
    assert result["options"] == ["甲", "乙"]
    assert result["answer"] == 0
    assert len(model.image_calls) == 1
    assert len(state["clicks"]) == 1
    text = pk.format_question_line(result, ui_number=1, question_no=1)
    assert "题目" in text and "蒙第一个：甲" in text


def test_changed_question_does_not_reuse_ocr_in_timeout_record(scene):
    clock, _, state = scene
    state["same"] = False
    result, _ = answer(scene, "hybrid", Model(clock, image_fails=True))
    assert result["guessed"]
    assert result["stem"] == "" and result["options"] == []
    assert len(state["clicks"]) == 1


def test_rejected_ocr_goes_directly_to_image_with_original_deadline(scene):
    clock, image, state = scene
    ocr = Ocr(clock)
    ocr.accepted = False
    model = Model(clock)
    reader = HybridQuestionReader(model, ocr, verbose=False)
    result = pk.answer_one_question(None, reader, image=image, captured_at=1000.0,
        capture_context=pk.PKCaptureContext(captured_at=1000.0), verbose=False)
    assert model.text_calls == []
    assert model.image_calls == [(pytest.approx(1011.2), pytest.approx(10.35))]
    assert result["answer"] == 1 and not result["guessed"]
    assert len(state["clicks"]) == 1


def test_reader_clears_previous_accepted_ocr_when_next_frame_is_rejected(scene):
    clock, image, _ = scene
    model = Model(clock, text_answer=1)
    ocr = Ocr(clock)
    reader = HybridQuestionReader(model, ocr, verbose=False)
    reader.read_question(image, deadline=1010.5)
    assert reader.last_question_local_read is ocr.item
    ocr.accepted = False
    reader.read_question(image, deadline=1010.5)
    assert reader.last_question_local_read is None


def test_timeout_keeps_guess_disabled_when_requested(scene):
    clock, image, state = scene
    reader = HybridQuestionReader(Model(clock), Ocr(clock), verbose=False)
    result = pk.answer_one_question(None, reader, image=image, captured_at=1000.0,
                                   guess_on_failure=False, verbose=False)
    assert result["skipped"] and not result["guessed"]
    assert state["clicks"] == []


def test_recognition_timeout_exception_never_falls_back_to_image(scene, monkeypatch):
    clock, image, _ = scene
    model = Model(clock)

    def timed_out(*args, **kwargs):
        raise RecognitionTimeout("deadline_exceeded")

    monkeypatch.setattr(model, "answer_from_text", timed_out)
    reader = HybridQuestionReader(model, Ocr(clock), verbose=False)
    with pytest.raises(RecognitionTimeout):
        reader.read_question(image, deadline=1011.2)
    assert model.image_calls == []


@pytest.mark.allow_network
@pytest.mark.parametrize("entry", ["hybrid", "preread"])
def test_real_http_text_timeout_guesses_without_an_image_request(monkeypatch, entry):
    import http.server
    import json
    import threading
    import time

    from weread_quiz.vision import VisionService

    release = threading.Event()
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            is_image = any(item.get("type") == "image_url"
                           for item in payload["messages"][0]["content"])
            requests.append(is_image)
            if not is_image:
                release.wait(2.0)
                return
            body = json.dumps({"choices": [{"message": {"content":
                '{"screen":"question","stem":"题目","options":["甲","乙"],"answer":1}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    item = OcrRead("题目", ["甲", "乙"], [0.99, 0.99], 2,
                   has_progress=True, progress_text="第1题·共10题")

    class LocalOcr:
        def read(self, image):
            return item

        def decide_merged(self, whole, strip):
            return OcrDecision(True, "", whole)

    image = Image.new("RGB", (430, 788))
    clicks = []
    monkeypatch.setattr(pk, "grab_screen", lambda _: image)
    monkeypatch.setattr(pk, "find_option_buttons",
                        lambda _: [Rect(60, 380, 310, 44), Rect(60, 450, 310, 44)])
    monkeypatch.setattr(pk, "is_same_question", lambda *_: True)
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **kw: clicks.append(box))
    monkeypatch.setattr(pk, "read_question_number",
                        lambda *a, **kw: pytest.fail("同题蒙答应复用已有题号，不额外 OCR"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    service = VisionService(f"http://127.0.0.1:{server.server_port}", "local-test", "test")
    try:
        service.warmup_client()
        reader = HybridQuestionReader(service, LocalOcr(), verbose=False, text_timeout=0.15)
        started = time.monotonic()
        result = pk.answer_one_question(None, reader, image=image, captured_at=started,
            capture_context=pk.PKCaptureContext(captured_at=started),
            ocr_read=item if entry == "preread" else None, verbose=False)
        assert result["guessed"] and result["answer"] == 0
        assert len(clicks) == 1
        assert requests == [False], "真实文本 HTTP 超时后不得再发图片请求"
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
        service.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=1.0)
