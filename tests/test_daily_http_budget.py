"""真实本地 HTTP 验证每日一答文本超时不回退、提前无效结果仍可回退。"""
import http.server
import json
import threading
import time

import pytest

from weread_quiz.daily import hybrid as hybrid_module
from weread_quiz.daily import runner as daily
from weread_quiz.daily.hybrid import DailyHybridReader
from weread_quiz.errors import RecognitionTimeout
from weread_quiz.models import Rect
from weread_quiz.vision import VisionService
from test_daily_pipeline import Ocr, config, frame, read


@pytest.mark.allow_network
@pytest.mark.parametrize("entry", ["daily", "hybrid"])
@pytest.mark.parametrize("response", ["timeout", "invalid"])
def test_daily_http_text_outcome_controls_image_fallback(monkeypatch, entry, response):
    release = threading.Event()
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            is_image = any(item.get("type") == "image_url"
                           for item in payload["messages"][0]["content"])
            requests.append(is_image)
            if not is_image and response == "timeout":
                release.wait(2.0)
                return
            content = ('{"stem":"题目","options":["甲","乙","丙"],"answer":2}'
                       if is_image else '{"answer_index":99}')
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    image = frame()
    boxes = [Rect(60, 320 + index * 70, 310, 44) for index in range(3)]
    clicks = []
    monkeypatch.setattr(daily, "_TEXT_TIMEOUT", 0.15)
    monkeypatch.setattr(hybrid_module, "_TEXT_TIMEOUT", 0.15)
    monkeypatch.setattr(daily, "grab_screen", lambda _: image)
    monkeypatch.setattr(daily, "find_option_buttons", lambda _: boxes)
    monkeypatch.setattr(daily, "is_question_screen", lambda _: True)
    monkeypatch.setattr(daily, "is_same_question", lambda *_: True)
    monkeypatch.setattr(daily, "click_box", lambda box, *a, **kw: clicks.append(box))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    service = VisionService(f"http://127.0.0.1:{server.server_port}", "local-test", "test")
    try:
        service.warmup_client()
        reader = DailyHybridReader(service, Ocr(read()), verbose=False)
        started = time.monotonic()
        if entry == "daily":
            context = daily.DailyCaptureContext(captured_at=started, confirmed_at=started)
            result = daily.answer_one_question(config(reader), image, None, 1,
                                              capture_context=context)
            assert result["guessed"] is (response == "timeout")
            assert result["answer"] == (0 if response == "timeout" else 2)
            assert len(clicks) == 1
        elif response == "timeout":
            with pytest.raises(RecognitionTimeout):
                reader.read_question(image, deadline=started + 9.0)
        else:
            question = reader.read_question(image, deadline=started + 9.0)
            assert question.answer_index == 2
        assert requests == ([False] if response == "timeout" else [False, True])
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
        service.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=1.0)
