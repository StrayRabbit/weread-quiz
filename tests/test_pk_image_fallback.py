"""PK 文本路径限时后，图片路径兜底的调用链回归。"""

from __future__ import annotations

import pytest

from weread_quiz import pk
from weread_quiz.errors import AnswerUnavailable
from weread_quiz.models import Rect
from weread_quiz.ocr_core import OcrRead
from weread_quiz.vision import VisionQuestion


class FakeBox(Rect):
    @property
    def center_y(self) -> int:
        return self.top + self.height // 2


def _boxes(count: int) -> list[FakeBox]:
    return [FakeBox(left=60, top=400 + 80 * i, width=310, height=44)
            for i in range(count)]


def _blank():
    from PIL import Image
    return Image.new("RGB", (430, 788), (20, 35, 70))


def _read(countdown: int = 12) -> OcrRead:
    """**完整**的预读：选项数与按钮数一致，带倒计时。"""
    read = OcrRead(
        stem="题目",
        options=["甲", "乙"],
        confidences=[0.99, 0.99],
        button_count=2,
        has_progress=True,
        progress_text="第1题·共10题",
    )
    read.countdown_seconds = countdown
    return read


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Vision:
    """文本路径没给出答案；图片路径按需给答案。"""

    usage = None
    last_question_usage = None

    def __init__(self, clock: FakeClock, *, image_answer=1, text_calls=None,
                 text_cost: float = 4.0):
        self._clock = clock
        self._image_answer = image_answer
        self._text_cost = text_cost
        self.text_calls = text_calls if text_calls is not None else []
        self.image_calls = []

    def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
        self.text_calls.append(timeout)
        self._clock.advance(self._text_cost)   # 文本路径慢，但不超上限
        return None

    def read_question_visual(self, image, prompt=None, *, timeout=None, deadline=None):
        self.image_calls.append(deadline)
        self._clock.advance(1.0)
        if self._image_answer is None:
            raise AnswerUnavailable("invalid_response")
        return VisionQuestion(stem="题目", options=["甲", "乙"],
                              answer_index=self._image_answer)


@pytest.fixture
def scene(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk, "grab_screen", lambda region: _blank())
    monkeypatch.setattr(pk, "find_option_buttons", lambda image: _boxes(2))
    monkeypatch.setattr(pk, "is_same_question", lambda a, b: True)
    clicked = []
    monkeypatch.setattr(pk, "click_box", lambda box, *a, **k: clicked.append(box))
    return clock, clicked


def test_slow_text_path_falls_back_to_image_path(scene):
    """文本路径没答上来 → 走图片路径，而不是直接蒙答。"""
    clock, clicked = scene
    vision = Vision(clock)

    result = pk.answer_one_question(
        (0, 0, 430, 788), 1.0, vision, image=_blank(), captured_at=clock.now,
        ocr_read=_read(), verbose=False,
    )

    assert result["guessed"] is False, "图片路径给了答案就不该蒙答"
    assert result["answer"] == 1
    assert vision.image_calls, "图片路径根本没被调用"
    assert len(clicked) == 1


def test_text_path_is_not_retried_before_the_image_path(scene):
    """文本路径只该被调一次 —— 再来一遍只会把剩余预算也吃掉。"""
    clock, _ = scene
    vision = Vision(clock)

    pk.answer_one_question(
        (0, 0, 430, 788), 1.0, vision, image=_blank(), captured_at=clock.now,
        ocr_read=_read(), verbose=False,
    )

    assert len(vision.text_calls) == 1, "文本路径被重复调用了"
    assert len(vision.image_calls) == 1


def test_no_image_path_when_budget_is_almost_gone(scene):
    """剩余预算不够跑图片路径 → 别试，直接蒙答。"""
    clock, clicked = scene
    vision = Vision(clock, text_cost=0.5)
    # 截图是 8 秒前拍的（等待阶段磨蹭过），倒计时 12 秒 → 预算到
    # 1010.5 秒，文本路径跑完后只剩 2 秒 —— 跑不动图片路径。
    clock.now = 1008.0

    result = pk.answer_one_question(
        (0, 0, 430, 788), 1.0, vision, image=_blank(), captured_at=1000.0,
        ocr_read=_read(), verbose=False,
    )

    assert result["guessed"] is True
    assert vision.image_calls == [], "预算不够时不该再发图片请求"
    assert len(clicked) == 1


def test_text_path_timeout_is_capped(scene):
    """文本路径单次最多 _TEXT_PATH_TIMEOUT 秒 —— 剩下的留给图片路径。"""
    clock, _ = scene
    vision = Vision(clock)

    pk.answer_one_question(
        (0, 0, 430, 788), 1.0, vision, image=_blank(), captured_at=clock.now,
        ocr_read=_read(), verbose=False,
    )

    assert vision.text_calls == [pk._TEXT_PATH_TIMEOUT]


@pytest.mark.allow_network
def test_timed_out_text_request_falls_through_to_image_request():
    """真实 HTTP 层：文本请求超时返回 None（不当异常），图片请求接手。

    这是「文本路径慢 → 图片路径兜底」能成立的前提：超时的文本请求
    不能把异常抛到上层，也不能把 deadline 耗光。
    """
    import http.server
    import json
    import threading
    import time

    from weread_quiz.vision import VisionService

    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            content = payload["messages"][0]["content"]
            if not any(item.get("type") == "image_url" for item in content):
                release.wait(3.0)      # 文本请求故意不回
                return
            body = json.dumps({"choices": [{"message": {"content": (
                '{"screen":"question","stem":"题目",'
                '"options":["甲","乙"],"answer":1}')}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    vision = VisionService(f"http://127.0.0.1:{server.server_port}", "local-test", "test")
    try:
        vision.warmup_client()
        deadline = time.monotonic() + 8.0
        index = vision.answer_from_text("题目", 2, timeout=1.0, deadline=deadline)
        assert index is None, "超时的文本请求应当返回 None，而不是抛异常"
        assert time.monotonic() < deadline, "文本请求不该把 deadline 耗光"
        question = vision.read_question(_blank(), deadline=deadline)
        assert question.answer_index == 1
    finally:
        release.set()
        vision.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=1.0)
