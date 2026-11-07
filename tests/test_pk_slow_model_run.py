"""PK 题号缓存范围、首题剩余预算及预热连接请求超时回归。"""

from __future__ import annotations

import time

import pytest

from weread_quiz.pk import runner as pk
from weread_quiz.pk import screen
from weread_quiz.vision import VisionService


# ---- 用例 1：缓存的"变了才重读"判据 -----------------------------------


def test_progress_cache_ignores_motion_outside_progress_line():
    """题干区的动画不该让题号行 OCR 的缓存失效。"""
    from PIL import Image
    import numpy as np

    # 造两张图：题干区有差异，题号行**完全一样**。
    a = Image.new("L", (430, 788), 20)
    b = Image.new("L", (430, 788), 20)
    a.paste(255, (100, 300, 200, 340))   # 题干区有白字
    b.paste(255, (100, 305, 200, 345))   # 同样的白字，**位置差 5px**

    mask_a = screen.progress_cache_mask(a)
    mask_b = screen.progress_cache_mask(b)

    # 题号行带（y248:312 的一部分）应该完全一致
    assert np.array_equal(mask_a[:10], mask_b[:10]), (
        "掩码的上半部分（题号行所在）不该因题干区的位移而变化"
    )


def test_progress_cache_changed_ignores_motion_below_progress_line():
    """题号行**下面**的动画不该让缓存失效。"""
    from PIL import Image
    import numpy as np

    a = Image.new("L", (430, 788), 20)
    b = Image.new("L", (430, 788), 20)
    # 题号行（y~260）一样
    a.paste(255, (120, 258, 300, 266))
    b.paste(255, (120, 258, 300, 266))
    # 题干区（y~320）不同
    a.paste(255, (100, 320, 320, 350))
    b.paste(255, (100, 326, 320, 352))

    assert not screen.progress_cache_changed(
        screen.progress_cache_mask(a), a.size, b
    ), "题干区的动画让题号行 OCR 的缓存失效了 —— 等待阶段被 OCR 泡着"


def test_progress_cache_still_detects_real_progress_line_change():
    """题号行**真的**变了（换题）时，缓存必须失效。"""
    from PIL import Image

    a = Image.new("L", (430, 788), 20)
    b = Image.new("L", (430, 788), 20)
    a.paste(255, (120, 258, 300, 266))   # 第1题·共10题
    b.paste(255, (120, 258, 200, 266))   # 同一行，内容不同

    assert screen.progress_cache_changed(
        screen.progress_cache_mask(a), a.size, b
    ), "题号行变了却没检测到 —— 会把上一题的结论套用到新画面上"


# ---- 用例 2：文本路径的预算夹紧方向 ------------------------------------


def test_text_deadline_uses_remaining_question_budget(monkeypatch):
    """首题预读用单题剩余预算，但单次最多 _TEXT_PATH_TIMEOUT 秒。

    上限不是「省时间」，是给**图片路径留预算**：文本路径吃光整题预算
    的话，超时后就只能蒙答（见 pk_budget._TEXT_PATH_TIMEOUT）。
    """
    now = 1000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    seen = {}

    class Vision:
        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            seen["deadline"] = deadline
            seen["timeout"] = timeout
            return None

    from weread_quiz.ocr_core import OcrRead
    read = OcrRead(stem="x", options=["a", "b"],
                   confidences=[0.9, 0.9], button_count=2)

    pk._answer_from_pre_read(Vision(), read, deadline=now + 6.0)

    assert seen["deadline"] == pytest.approx(now + 6.0)
    assert seen["timeout"] == pytest.approx(6.0 - 3.5 - 0.3)


def test_text_deadline_still_respects_an_earlier_deadline(monkeypatch):
    """deadline 更早时，要用它（别超总预算）。"""
    now = 1000.0
    monkeypatch.setattr(pk.time, "monotonic", lambda: now)

    seen = {}

    class Vision:
        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            seen["deadline"] = deadline
            seen["timeout"] = timeout
            return None

    from weread_quiz.ocr_core import OcrRead
    read = OcrRead(stem="x", options=["a", "b"],
                   confidences=[0.9, 0.9], button_count=2)

    pk._answer_from_pre_read(Vision(), read, deadline=now + 1.5)

    assert seen["deadline"] == pytest.approx(now + 1.5), (
        "deadline 更早时应该用它 —— 别超总预算"
    )
    assert seen["timeout"] == pytest.approx(1.5), (
        "剩余预算比上限还短时，用剩余预算"
    )


# ---- 用例 3：预热客户端的请求预算 --------------------------------------


@pytest.mark.allow_network
def test_prewarmed_client_accepts_response_inside_three_second_budget():
    """读超时不包含开局预热成本，2.2 秒响应不能被 1.8 秒的假预算掐掉。"""
    import http.server
    import threading

    release = threading.Event()

    class SlowHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            release.wait(2.2)
            body = b'{"choices":[{"message":{"content":"{\\"answer_index\\":1}"}}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    vision = VisionService(f"http://127.0.0.1:{server.server_port}", "local-test", "test")
    try:
        vision.warmup_client()
        started = time.monotonic()
        answer = vision.answer_from_text("test", 3, timeout=3.0, deadline=started + 3.0)
        assert answer == 1
    finally:
        release.set()
        vision.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=1.0)
