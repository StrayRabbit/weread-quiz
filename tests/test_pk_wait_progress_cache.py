"""PK 等待阶段题号 OCR 重读间隔、占用率和画面绑定缓存回归。"""

from __future__ import annotations

import time

import pytest

from weread_quiz.pk import runner as pk


class FakeClock:
    """可控的单调时钟。"""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _blank(width: int = 430, height: int = 788):
    from PIL import Image
    return Image.new("RGB", (width, height), (20, 35, 70))


class SlowProgressOcr:
    """题号行 OCR 替身：每次调用都要 1.5 秒（像真实 RapidOCR）。"""

    def __init__(self, clock: FakeClock, *, result=(None, ""), cost: float = 1.5):
        self.clock = clock
        self.result = result
        self.cost = cost
        self.calls = 0
        # 每次调用的**开始**时刻。
        self.starts: list[float] = []

    def read_progress_text(self, image):
        self.calls += 1
        self.starts.append(self.clock.now)
        self.clock.advance(self.cost)   # OCR 本身要 1.5 秒
        return self.result

    def read_progress(self, image):
        return self.result[0]


class _StubVision:
    """等待阶段用不到模型（本地门控就会判否），所以只是占位。"""

    usage = None
    last_question_usage = None

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        raise AssertionError("本用例不该调模型")


class _StopWait(Exception):
    """用来跳出等待阶段的循环。"""


def _run_wait(monkeypatch, clock: FakeClock, ocr: SlowProgressOcr, *,
              polls: int = 12, region_changed=False):
    """跑等待阶段的前 `polls` 轮，然后停下来。"""
    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk, "looks_like_question_template", lambda image: False)
    monkeypatch.setattr(pk, "has_rendered_stem", lambda image: False)
    monkeypatch.setattr(pk, "progress_cache_mask", lambda image: "mask")
    monkeypatch.setattr(pk, "progress_cache_changed", lambda *a: region_changed)
    monkeypatch.setattr(pk, "_poll_sleep", lambda totals: clock.advance(0.1))

    grabs = {"n": 0}

    def counting_grab(region):
        grabs["n"] += 1
        if grabs["n"] > polls:
            raise _StopWait()
        return _blank()

    monkeypatch.setattr(pk, "grab_screen", counting_grab)

    with pytest.raises(_StopWait):
        pk.wait_for_question(
            (0, 0, 430, 788), _StubVision(), timeout=30.0, verbose=False,
            ocr=ocr, capture_context=pk.PKCaptureContext(),
        )


# ---- 用例 --------------------------------------------------------------


def test_progress_retry_interval_not_shorter_than_one_ocr_run():
    """重读间隔必须 ≥ 单次题号行 OCR 的耗时。"""
    assert pk._PROGRESS_CACHE_SECONDS >= pk._PROGRESS_OCR_SECONDS, (
        f"重读间隔 {pk._PROGRESS_CACHE_SECONDS}s 比单次 OCR "
        f"{pk._PROGRESS_OCR_SECONDS}s 还短 —— 等待阶段会一直泡在 OCR 里"
    )


def test_progress_ocr_duty_cycle_stays_low(monkeypatch):
    """等待阶段花在题号行 OCR 上的时间占比要可控。"""
    clock = FakeClock()
    ocr = SlowProgressOcr(clock, result=(None, ""))
    _run_wait(monkeypatch, clock, ocr, polls=120)

    total = clock.now
    ocr_time = ocr.cost * ocr.calls
    duty = ocr_time / total if total else 1.0

    assert ocr.calls >= 3, f"样本太少（{ocr.calls} 次 OCR），测不出稳态占用率"
    assert duty <= 0.4, (
        f"题号行 OCR 占了等待阶段 {duty:.0%}（{ocr_time:.1f}s / {total:.1f}s，"
        f"{ocr.calls} 次）—— 太高了，第 1 题的倒计时会被吃掉"
    )


def test_progress_ocr_gap_accounts_for_its_own_duration(monkeypatch):
    """两次 OCR 之间必须留出足够间隔，容得下一次 OCR 跑完。"""
    clock = FakeClock()
    ocr = SlowProgressOcr(clock, result=(None, ""))
    _run_wait(monkeypatch, clock, ocr, polls=120)

    assert len(ocr.starts) >= 2, "至少要有两次 OCR 才能谈间隔"
    for earlier, later in zip(ocr.starts, ocr.starts[1:]):
        gap = later - earlier
        assert gap >= ocr.cost, (
            f"两次 OCR 只隔了 {gap:.2f}s，而一次 OCR 要 {ocr.cost}s —— "
            f"下一次还没跑完上一次就又开始了"
        )


def test_progress_ocr_reruns_when_region_changes(monkeypatch):
    """区域变了必须重读 —— 旧结论不能套用到新画面上。"""
    clock = FakeClock()
    ocr = SlowProgressOcr(clock, result=(None, ""))
    _run_wait(monkeypatch, clock, ocr, polls=6, region_changed=True)

    assert ocr.calls >= 2, "区域变了却没有重读题号行 —— 结论会套用到新画面上"
