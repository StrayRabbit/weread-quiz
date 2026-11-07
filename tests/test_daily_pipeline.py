"""每日一答的本地确认、预算及逐题处理回归。"""
from types import SimpleNamespace

import pytest
from PIL import Image

from weread_quiz import daily
from weread_quiz.daily_hybrid import DailyHybridReader
from weread_quiz.daily_ocr import DailyOcrRead, DailyOcrReader
from weread_quiz.daily_vision import FALLBACK_RESERVE_SECONDS, text_path_deadline
from weread_quiz.errors import AnswerUnavailable, OcrRejected, RecognitionTimeout
from weread_quiz.models import Rect
from weread_quiz.vision import VisionQuestion


def frame():
    return Image.new("RGB", (430, 788), (20, 35, 70))


def read(*, count=3, confidence=0.99):
    return DailyOcrRead(
        stem="题目", options=["甲", "乙", "丙"][:count],
        confidences=[confidence] * count, button_count=3,
        has_progress=True, progress_no=1,
    )


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(daily.time, "monotonic", value)
    monkeypatch.setattr(daily.time, "sleep", value.sleep)
    return value


class Ocr(DailyOcrReader):
    def __init__(self, result):
        super().__init__()
        self.result = result
        self.calls = self.strip_calls = 0

    def read(self, image):
        self.calls += 1
        return self.result

    def read_strip(self, image, whole=None):
        self.strip_calls += 1
        return None


class SlowOcr(Ocr):
    """本地 OCR 要花掉一点时间 —— 真实链路上它会吃掉近半秒。"""

    def __init__(self, result, clock, seconds=0.5):
        super().__init__(result)
        self._clock = clock
        self._seconds = seconds

    def read(self, image):
        result = super().read(image)
        self._clock.sleep(self._seconds)
        return result


class Vision:
    usage = last_question_usage = None

    def __init__(self):
        self.calls = []
        self.text_result = 1

    def answer_from_text(self, body, count, *, timeout=None, deadline=None):
        self.calls.append(("text", timeout, deadline))
        return self.text_result

    def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
        self.calls.append(("image", timeout, deadline))
        return VisionQuestion(stem="题目", options=["甲", "乙", "丙"],
                              answer_index=2, screen="question")


def test_local_read_keeps_progress_evidence_on_rejection():
    evidence = read(count=2)
    vision = Vision()
    reader = DailyHybridReader(vision, Ocr(evidence), verbose=False)
    with pytest.raises(OcrRejected) as rejected:
        reader.read_question_local(frame())
    assert rejected.value.read is evidence
    assert vision.calls == []


def test_preread_text_has_own_budget_and_no_second_ocr():
    vision = Vision()
    ocr = Ocr(read())
    reader = DailyHybridReader(vision, ocr, verbose=False)
    local = reader.read_question_local(frame())
    question = reader.answer_from_ocr_read(local, timeout=3.0, deadline=103.0)
    assert question.answer_index == 1
    assert question.local_read and question.progress_no == 1
    assert ocr.calls == 1
    assert vision.calls == [("text", 3.0, 103.0)]


def test_visual_entry_does_not_repeat_local_ocr():
    vision = Vision()
    ocr = Ocr(read(count=2))
    reader = DailyHybridReader(vision, ocr, verbose=False)
    reader.read_question_visual(frame(), timeout=2.0, deadline=102.0)
    assert ocr.calls == 0
    assert vision.calls == [("image", 2.0, 102.0)]


def test_hybrid_does_not_cascade_after_absolute_deadline(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("weread_quiz.daily_hybrid.time.monotonic", clock)
    vision = Vision()

    def slow_text(*args, **kwargs):
        vision.calls.append(("text", kwargs["timeout"], kwargs["deadline"]))
        clock.sleep(7)
        return None

    vision.answer_from_text = slow_text
    reader = DailyHybridReader(vision, Ocr(read()), verbose=False)
    with pytest.raises(RecognitionTimeout):
        reader.read_question(frame(), deadline=106.0)
    assert [c[0] for c in vision.calls] == ["text"]


def test_hybrid_cascade_leaves_fallback_reserve(monkeypatch):
    """混合级联里的文本路径同样要让出回退预算。"""
    clock = Clock()
    monkeypatch.setattr("weread_quiz.daily_hybrid.time.monotonic", clock)
    vision = Vision()

    def burning_text(*args, **kwargs):
        vision.calls.append(("text", kwargs["timeout"], kwargs["deadline"]))
        clock.sleep(kwargs["deadline"] - clock.now)
        return None

    vision.answer_from_text = burning_text
    reader = DailyHybridReader(vision, SlowOcr(read(), clock), verbose=False)
    reader.read_question(frame(), deadline=106.0)
    assert [c[0] for c in vision.calls] == ["text", "image"]
    text_deadline = vision.calls[0][2]
    image_deadline = vision.calls[1][2]
    assert image_deadline - text_deadline >= FALLBACK_RESERVE_SECONDS


@pytest.fixture
def desktop(monkeypatch):
    image = frame()
    boxes = [SimpleNamespace(center_y=320 + i * 70) for i in range(3)]
    clicks = []
    monkeypatch.setattr(daily, "grab_screen", lambda region: image)
    monkeypatch.setattr(daily, "find_option_buttons", lambda image: boxes)
    monkeypatch.setattr(daily, "pick_button", lambda boxes, index: boxes[index])
    monkeypatch.setattr(daily, "is_question_screen", lambda image: True)
    monkeypatch.setattr(daily, "is_question_stable", lambda a, b: True)
    monkeypatch.setattr(daily, "is_same_question", lambda a, b: True)
    monkeypatch.setattr(daily, "has_answer_reveal", lambda image: False)
    monkeypatch.setattr("weread_quiz.daily_screen.has_rendered_stem", lambda image: True)
    monkeypatch.setattr(daily, "click_box", lambda box, *a, **k: clicks.append(box.center_y))
    return image, clicks


def config(vision, **kwargs):
    return daily.DailyConfig(Rect(0, 0, 430, 788), vision, verbose=False, **kwargs)


def test_wait_confirms_locally_without_model_calls(desktop):
    vision = Vision()
    ocr = Ocr(read())
    reader = DailyHybridReader(vision, ocr, verbose=False)
    context = daily.DailyCaptureContext()
    image, question = daily.wait_for_question(config(reader), capture_context=context)
    assert image is desktop[0] and question is None
    assert vision.calls == []
    assert context.ocr_read is ocr.result and context.ocr_accepted
    assert context.captured_at is not None and context.confirmed_at is not None


def test_wait_partial_options_still_confirms_interface(desktop):
    vision = Vision()
    ocr = Ocr(read(count=2))
    reader = DailyHybridReader(vision, ocr, verbose=False)
    context = daily.DailyCaptureContext()
    image, question = daily.wait_for_question(config(reader), capture_context=context)
    assert image is desktop[0] and question is None
    assert context.ocr_read is ocr.result and not context.ocr_accepted
    assert vision.calls == []


def test_preread_is_used_once_during_answer(desktop):
    vision = Vision()
    ocr = Ocr(read())
    reader = DailyHybridReader(vision, ocr, verbose=False)
    cfg = config(reader)
    context = daily.DailyCaptureContext()
    image, question = daily.wait_for_question(cfg, capture_context=context)
    calls_before = ocr.calls
    result = daily.answer_one_question(cfg, image, question, 1, capture_context=context)
    assert ocr.calls == calls_before
    assert [c[0] for c in vision.calls] == ["text"]
    assert vision.calls[0][1] == 5.0
    assert result["answer"] == 1 and not result["guessed"]
    assert len(desktop[1]) == 1


def test_partial_preread_goes_directly_to_image(desktop):
    vision = Vision()
    ocr = Ocr(read(count=2))
    reader = DailyHybridReader(vision, ocr, verbose=False)
    cfg = config(reader)
    context = daily.DailyCaptureContext()
    image, _ = daily.wait_for_question(cfg, capture_context=context)
    before = ocr.calls
    result = daily.answer_one_question(cfg, image, None, 1, capture_context=context)
    assert ocr.calls == before
    assert [c[0] for c in vision.calls] == ["image"]
    assert result["answer"] == 2 and not result["guessed"]


def test_text_failure_uses_remaining_budget_for_image(desktop, clock):
    vision = Vision()

    def failing_text(*args, **kwargs):
        vision.calls.append(("text", kwargs["timeout"], kwargs["deadline"]))
        clock.sleep(2.0)
        return None

    vision.answer_from_text = failing_text
    reader = DailyHybridReader(vision, Ocr(read()), verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(), ocr_accepted=True)
    result = daily.answer_one_question(config(reader), desktop[0], None, 1,
                                       capture_context=context)
    assert [c[0] for c in vision.calls] == ["text", "image"]
    # 识别总预算 9 秒（锚点 100.0 → 109.0），文本路径上限 5 秒。
    assert vision.calls[0][2] == 105.0
    assert vision.calls[1][2] == 109.0
    assert result["answer"] == 2


def test_slow_text_request_still_gets_answered(desktop, clock):
    """复现 2026-11-02 第 5 题的失败：模型慢一次，不该把整题赔进去。

    实测那次：文本请求 3.09 秒 —— 旧上限 3.0 秒直接判死，回退只剩
    2.5 秒也超时，两条路同时空 → 蒙答第 1 个按钮（30.5%，错的）→
    今日机会用完。对齐 PK 后文本上限 5 秒，这题在文本路径就答得出来。
    """
    vision = Vision()

    def slow_text(*args, **kwargs):
        vision.calls.append(("text", kwargs["timeout"], kwargs["deadline"]))
        clock.sleep(3.09)
        deadline = kwargs.get("deadline")
        if deadline is not None and clock.now >= deadline:
            raise RecognitionTimeout("deadline_exceeded")
        return 1

    def slow_image(image, prompt=None, *, timeout=None, deadline=None):
        vision.calls.append(("image", timeout, deadline))
        clock.sleep(3.0)
        if deadline is not None and clock.now >= deadline:
            raise RecognitionTimeout("deadline_exceeded")
        return VisionQuestion(stem="题目", options=["甲", "乙", "丙"],
                              answer_index=2, screen="question")

    vision.answer_from_text = slow_text
    vision.read_question = slow_image
    reader = DailyHybridReader(vision, SlowOcr(read(), clock, seconds=0.45),
                               verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(), ocr_accepted=True)
    result = daily.answer_one_question(config(reader), desktop[0], None, 1,
                                       capture_context=context)
    assert [c[0] for c in vision.calls] == ["text"]
    assert result["answer"] == 1 and not result["guessed"]


def test_text_path_leaves_fallback_reserve(desktop, clock):
    """文本路径烧完预算后，图片回退必须还拿得到保留预算。

    2026-11-02 实测：文本路径超时吃掉 3 秒后回退只剩 2.5 秒，模型慢
    一点就两条路同时空 → 蒙答（第 5 题「中东地区石油出口量…百分比」）。
    """
    vision = Vision()

    def burning_text(*args, **kwargs):
        vision.calls.append(("text", kwargs["timeout"], kwargs["deadline"]))
        clock.sleep(kwargs["deadline"] - clock.now)
        return None

    vision.answer_from_text = burning_text
    reader = DailyHybridReader(vision, SlowOcr(read(), clock), verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(), ocr_accepted=True)
    daily.answer_one_question(config(reader), desktop[0], None, 1,
                              capture_context=context)
    assert [c[0] for c in vision.calls] == ["text", "image"]
    text_deadline = vision.calls[0][2]
    image_deadline = vision.calls[1][2]
    assert image_deadline - text_deadline >= FALLBACK_RESERVE_SECONDS


@pytest.mark.parametrize("overrun", [0.001, 0.11, 0.2])
@pytest.mark.parametrize("entry", ["daily", "hybrid"])
def test_text_timeout_overrun_still_allows_image_fallback(desktop, clock, entry, overrun):
    """文本超时晚退出一点，也应留够 3.5 秒启动图片回退。"""
    vision = Vision()
    image_started = []

    def timed_out_text(*args, **kwargs):
        vision.calls.append(("text", kwargs["timeout"], kwargs["deadline"]))
        clock.sleep(kwargs["deadline"] - clock.now + overrun)
        raise RecognitionTimeout("deadline_exceeded")

    def slow_image(image, prompt=None, *, timeout=None, deadline=None):
        vision.calls.append(("image", timeout, deadline))
        image_started.append(clock.now)
        clock.sleep(3.4)
        return VisionQuestion(stem="题目", options=["甲", "乙", "丙"],
                              answer_index=2, screen="question")

    vision.answer_from_text = timed_out_text
    vision.read_question = slow_image
    reader = DailyHybridReader(vision, SlowOcr(read(), clock, seconds=0.542),
                               verbose=False)
    if entry == "daily":
        context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0)
        result = daily.answer_one_question(config(reader), desktop[0], None, 1,
                                           capture_context=context)
        assert result["answer"] == 2 and not result["guessed"]
    else:
        question = reader.read_question(desktop[0], deadline=109.0)
        assert question.answer_index == 2
    assert [call[0] for call in vision.calls] == ["text", "image"]
    assert vision.calls[1][2] == 109.0
    assert 109.0 - image_started[0] >= FALLBACK_RESERVE_SECONDS
    assert clock.now < 109.0


def test_text_path_deadline_leaves_reserve_even_after_slow_ocr(desktop, clock):
    """OCR 花了时间也不能把回退饿死 —— 文本路径的截止时刻照样留出保留。"""
    vision = Vision()
    reader = DailyHybridReader(vision, SlowOcr(read(), clock, seconds=1.5),
                               verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(), ocr_accepted=True)
    daily.answer_one_question(config(reader), desktop[0], None, 1,
                              capture_context=context)
    # 总预算 109.0 − 图片门槛 3.5 − 退出余量 0.3 = 105.2。
    assert vision.calls[0][2] == pytest.approx(105.2)


def test_text_path_takes_all_when_budget_cannot_split(desktop, clock):
    """预算不够两条路分时，文本路径拿走全部 —— 它更快更便宜。"""
    vision = Vision()
    reader = DailyHybridReader(vision, SlowOcr(read(), clock, seconds=6.0),
                               verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(), ocr_accepted=True)
    daily.answer_one_question(config(reader), desktop[0], None, 1,
                              capture_context=context)
    assert [c[0] for c in vision.calls] == ["text"]
    assert vision.calls[0][2] == 109.0


def test_image_path_skipped_when_budget_below_floor(desktop, clock):
    """剩余预算不够图片路径就蒙答，不发一个注定超时的请求。

    与 PK 的 _IMAGE_PATH_MIN_BUDGET 同一判据。
    """
    vision = Vision()
    # 判否（选项数与按钮数不符）→ 不走文本路径；OCR 又花了 6 秒，
    # 剩下的 3 秒不够跑图片路径。
    reader = DailyHybridReader(vision, SlowOcr(read(count=2), clock, seconds=6.0),
                               verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(count=2), ocr_accepted=False)
    result = daily.answer_one_question(config(reader), desktop[0], None, 1,
                                       capture_context=context)
    assert vision.calls == []
    assert result["guessed"]


def test_text_path_deadline_keeps_reserve():
    """切分规则本身：文本路径只能用「剩余预算减去保留」。"""
    # 总预算 9 秒、保留 3.5 秒 → 文本路径截止 105.0。
    assert text_path_deadline(109.0, 100.0, 5.0) == 105.0
    # OCR 花了 0.5 秒 → 图片门槛之外还要留下 0.3 秒退出余量。
    assert text_path_deadline(109.0, 100.5, 5.0) == pytest.approx(105.2)
    # 仅剩 3.7 秒，装不下图片门槛 + 退出余量 → 全给文本路径。
    assert text_path_deadline(109.0, 105.3, 5.0) == 109.0
    # 剩余不够留回退 → 全给文本路径。
    assert text_path_deadline(109.0, 106.0, 5.0) == 109.0
    # 预算已耗光 → 文本路径仍有自己的一份短预算，图片回退不受益。
    assert text_path_deadline(109.0, 110.0, 5.0) == 115.0


@pytest.mark.parametrize("captured_at", [None, 90.0])
def test_unknown_or_old_image_never_clicks(desktop, captured_at):
    question = VisionQuestion("题目", ["甲", "乙", "丙"], 1)
    context = daily.DailyCaptureContext(captured_at=captured_at)
    result = daily.answer_one_question(config(Vision()), desktop[0], question, 1,
                                       capture_context=context)
    assert result["skipped"]
    assert desktop[1] == []


def test_local_answer_still_requires_same_question(desktop, monkeypatch):
    monkeypatch.setattr(daily, "is_same_question", lambda a, b: False)
    question = VisionQuestion("题目", ["甲", "乙", "丙"], 1, local_read=True)
    context = daily.DailyCaptureContext(captured_at=90.0)
    result = daily.answer_one_question(config(Vision()), desktop[0], question, 1,
                                       capture_context=context)
    assert result["skipped"] and desktop[1] == []


def test_validation_failsafe_propagates(desktop, monkeypatch):
    from pyautogui import FailSafeException

    def failsafe(region):
        raise FailSafeException()

    monkeypatch.setattr(daily, "grab_screen", failsafe)
    question = VisionQuestion("题目", ["甲", "乙", "丙"], 1)
    with pytest.raises(FailSafeException):
        daily.answer_one_question(config(Vision()), desktop[0], question, 1,
                                   capture_context=daily.DailyCaptureContext(captured_at=100.0))


def test_dry_run_does_not_click_or_wait_next(desktop, monkeypatch):
    monkeypatch.setattr(daily, "wait_for_next_question", lambda *a, **k: pytest.fail("dry-run 不等下一题"))
    report = daily.run_daily(config(DailyHybridReader(Vision(), Ocr(read()), verbose=False), dry_run=True))
    assert report.outcome == "dry_run"
    assert report.answered == 0
    assert desktop[1] == []


def test_last_question_line_counts_as_progress_evidence():
    """卡片最后一题写的是「最后一题」—— 也是题号证据，只是没有编号。

    认不出它，这一题就会被门控判否、被迫回退图片路径，题干里还会
    混进「最后一题」三个字（2026-11-02 实测）。
    """
    reader = DailyOcrReader()
    lines = [(205.0, 173.0, 256.0, 198.0, 213.0, "最后一题", 0.99),
             (240.0, 80.0, 340.0, 229.0, 252.0, "纳兰词里的那句话", 0.99)]
    stem, has_progress, progress_no = reader._split_stem(frame(), lines)
    assert has_progress is True
    assert progress_no is None, "读不出编号就别猜"
    assert stem == "纳兰词里的那句话", "「最后一题」不是题干的一部分"


def test_progress_line_with_number_is_unchanged():
    reader = DailyOcrReader()
    lines = [(205.0, 173.0, 256.0, 198.0, 213.0, "第7题·共12题", 0.99),
             (240.0, 80.0, 340.0, 229.0, 252.0, "题目", 0.99)]
    stem, has_progress, progress_no = reader._split_stem(frame(), lines)
    assert has_progress is True
    assert progress_no == 7
    assert stem == "题目"
