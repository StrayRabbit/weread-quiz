"""每日一答的多题统计、终态和资源安全回归。"""
from types import SimpleNamespace

import pytest
from PIL import Image

from weread_quiz import daily
from weread_quiz.daily_hybrid import DailyHybridReader
from weread_quiz.daily_vision import read_daily_answer
from weread_quiz.errors import AnswerUnavailable, OcrRejected
from test_daily_pipeline import Clock, Ocr, Vision, config, frame, read, clock, desktop


def test_two_question_run_tracks_each_question_usage(desktop, monkeypatch):
    from weread_quiz.usage import UsageTracker

    vision = Vision()
    vision.usage = UsageTracker()
    original = vision.answer_from_text

    def text(*args, **kwargs):
        vision.usage.record({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
        return original(*args, **kwargs)

    vision.answer_from_text = text
    ocr = Ocr(read())
    reader = DailyHybridReader(vision, ocr, verbose=False)
    captured = []
    monkeypatch.setattr(daily, "log_usage", lambda label, usage, recorder: captured.append((label, usage.requests)))
    monkeypatch.setattr(daily, "wait_for_next_question", lambda *a, **k: daily._WaitOutcome("next", desktop[0], captured_at=100.25))
    monkeypatch.setattr(daily, "_read_final", lambda cfg, **kwargs: "success")
    report = daily.run_daily(config(reader, max_questions=2))
    assert report.outcome == "success" and report.answered == 2
    assert report.skipped == 0
    assert ocr.calls == 2 and len(vision.calls) == 2
    assert captured == [("第 1 题", 1), ("第 2 题", 1)]


def test_model_failure_guesses_only_after_validation(desktop):
    class Unavailable(Vision):
        def read_question(self, *args, **kwargs):
            raise AnswerUnavailable("invalid_response")

    result = daily.answer_one_question(config(Unavailable()), desktop[0], None, 1,
                                       capture_context=daily.DailyCaptureContext(captured_at=100.0))
    assert result["guessed"] and result["answer"] == 0
    assert len(desktop[1]) == 1


def test_slow_image_response_never_clicks(desktop, clock):
    class Slow(Vision):
        def read_question(self, *args, **kwargs):
            clock.sleep(9)
            return super().read_question(*args, **kwargs)

    result = daily.answer_one_question(config(Slow()), desktop[0], None, 1,
                                       capture_context=daily.DailyCaptureContext(captured_at=100.0))
    assert result["skipped"] and not desktop[1]


def test_wait_fallback_is_one_shot_not_repeated(desktop, clock):
    evidence = read()
    evidence.has_progress = False
    vision = Vision()

    def failing_image(*args, **kwargs):
        vision.calls.append(("image", None, None))
        raise AnswerUnavailable("offline")

    vision.read_question = failing_image
    reader = DailyHybridReader(vision, Ocr(evidence), verbose=False)
    image, question = daily.wait_for_question(config(reader), timeout=10)
    assert image is question is None
    assert len(vision.calls) == 1
    assert clock.now >= 110


def test_empty_stem_never_calls_model(desktop, monkeypatch):
    monkeypatch.setattr("weread_quiz.daily_screen.has_rendered_stem", lambda image: False)
    vision = Vision()
    image, question = daily.wait_for_question(config(vision), timeout=1)
    assert image is question is None
    assert not vision.calls


def test_terminal_without_explicit_text_is_interrupted(desktop, monkeypatch):
    monkeypatch.setattr("weread_quiz.daily_buttons._detect_any_buttons", lambda image: False)
    monkeypatch.setattr(daily, "has_progress_line", lambda image: False)
    outcome = daily.wait_for_next_question(config(Vision()), desktop[0])
    assert outcome.status == "interrupted"


def test_terminal_requires_matching_fresh_frame(desktop, monkeypatch):
    monkeypatch.setattr("weread_quiz.daily_buttons._detect_any_buttons", lambda image: False)
    monkeypatch.setattr(daily, "has_progress_line", lambda image: False)
    terminal = SimpleNamespace(kind="revive", evidence="可以通过以下方式复活")
    cfg = config(Vision(), ocr=SimpleNamespace(read_terminal=lambda image: terminal))
    outcome = daily.wait_for_next_question(cfg, desktop[0])
    assert outcome.status == "revive"
    changed = Image.new("RGB", desktop[0].size, "black")
    monkeypatch.setattr(daily, "grab_screen", lambda region: changed)
    verdict, fresh = daily._confirm_terminal(cfg, desktop[0])
    assert verdict is None and fresh is changed


def test_last_skipped_question_does_not_read_final(desktop, monkeypatch):
    monkeypatch.setattr(daily, "is_same_question", lambda a, b: False)
    monkeypatch.setattr(daily, "_read_final", lambda *a: pytest.fail("末题跳过不得结算"))
    report = daily.run_daily(config(DailyHybridReader(Vision(), Ocr(read()), verbose=False), max_questions=1))
    assert report.outcome == "interrupted" and report.answered == 0


def test_daily_answer_forwards_budget_without_duplicate_request():
    vision = Vision()
    read_daily_answer(vision, frame(), timeout=2.0, deadline=102.0)
    assert vision.calls == [("image", 2.0, 102.0)]


def test_legacy_recognize_only_reader_is_supported():
    class Legacy:
        def recognize(self, image, prompt=None):
            return '{"stem":"题目","options":["甲","乙"],"answer":1}'

    assert read_daily_answer(Legacy(), frame()).answer_index == 1


def test_cli_closes_vision_if_advance_fails(monkeypatch, tmp_path, capsys):
    from weread_quiz import capture, cli, config as configuration, daily_hybrid, daily_vision, dpi
    from weread_quiz.window import finder

    cfg = SimpleNamespace(model_name="fake", model_thinking=True, validate_model=lambda: None)
    monkeypatch.setattr(configuration.AppConfig, "from_env", lambda: cfg)
    monkeypatch.setattr(dpi, "init_dpi_awareness", lambda: None)
    monkeypatch.setattr(dpi, "get_scale_for_window", lambda hwnd: 1.0)
    region = SimpleNamespace(left=0, top=0, width=430, height=788, as_dict=lambda: {})
    host = SimpleNamespace(title="微信读书", width=430, height=788, left=0, top=0,
                           hwnd=1, to_rect=lambda: region)
    monkeypatch.setattr(finder, "find_wechat_window", lambda: host)
    monkeypatch.setattr(capture, "RunRecorder", lambda **k: SimpleNamespace(run_dir=tmp_path, log=lambda *a: None))
    monkeypatch.setattr(capture, "grab_screen", lambda region: frame())
    closed = []
    vision = SimpleNamespace(close=lambda: closed.append(True), usage=None)
    monkeypatch.setattr(daily_vision, "build_daily_vision_service", lambda *a, **k: vision)
    ocr = SimpleNamespace(warm_error="")
    monkeypatch.setattr(daily_hybrid, "DailyHybridReader", lambda *a, **k: SimpleNamespace(_ocr=ocr, warmup=lambda sample: True))
    monkeypatch.setattr(cli, "_advance_to_ready", lambda *a, **k: False)
    assert cli._cmd_daily(False) == 2
    assert closed == [True]
    output = capsys.readouterr().out
    assert "识别预算=9s" in output and "文本上限=5s" in output


def test_unavailable_ocr_uses_image_without_retrying_local(desktop):
    vision = Vision()
    ocr = Ocr(read())
    reader = DailyHybridReader(vision, ocr, verbose=False)
    reader.local_ocr_available = False
    report = daily.run_daily(config(reader, dry_run=True))
    assert report.outcome == "dry_run"
    assert ocr.calls == 0
    assert [call[0] for call in vision.calls] == ["image"]


def test_dry_run_guess_is_not_reported_as_actual_answer(desktop):
    class Unavailable(Vision):
        def read_question(self, *args, **kwargs):
            raise AnswerUnavailable("offline")

    report = daily.run_daily(config(Unavailable(), dry_run=True))
    assert report.answered == report.solid == 0
    assert report.guessed == []


def test_question_masks_ignore_pixels_outside_card():
    from weread_quiz import daily_screen
    from PIL import ImageDraw

    before = frame()
    after = before.copy()
    draw = ImageDraw.Draw(after)
    draw.rectangle((0, 200, 9, 299), fill="white")
    draw.rectangle((420, 200, 429, 299), fill="white")
    assert daily_screen.stem_change_count(before, after) == 0
    assert not daily_screen.has_progress_line(after)
    assert not daily_screen.has_rendered_stem(after)
    draw.rectangle((100, 230, 200, 250), fill="white")
    assert daily_screen.is_new_question(before, after)
    assert daily_screen.has_rendered_stem(after)


def test_progress_cache_is_bound_to_source_image():
    reader = DailyHybridReader(Vision(), Ocr(read()), verbose=False)
    first, second = frame(), frame()
    reader.read_question_local(first)
    reader._ocr.result = None
    with pytest.raises(OcrRejected):
        reader.read_question_local(second)
    assert reader.read_progress_number(second) is None


@pytest.mark.parametrize("change", ["stem", "number"])
def test_click_identity_rejects_small_content_change(change):
    from weread_quiz import daily_screen
    from PIL import ImageDraw

    before = Image.new("RGB", (430, 788), (10, 20, 40))
    draw = ImageDraw.Draw(before)
    draw.rectangle((100, 202, 220, 207), fill="white")
    draw.rectangle((100, 240, 220, 250), fill="white")
    for top in (310, 380, 450):
        draw.rectangle((40, top, 390, top + 44), fill=(42, 59, 74))
    after = before.copy()
    changed = ImageDraw.Draw(after)
    y = 260 if change == "stem" else 202
    changed.rectangle((230, y, 239, y + 10), fill="white")
    assert daily_screen.is_question_screen(before)
    assert daily_screen.is_question_screen(after)
    assert daily_screen.is_question_stable(before, after)
    assert not daily_screen.is_same_question(before, after)
    assert daily_screen.is_same_question(before, before.copy())


def test_dry_run_emits_current_question_usage(desktop, monkeypatch):
    from weread_quiz.usage import UsageTracker

    vision = Vision()
    vision.usage = UsageTracker()

    def text(*args, **kwargs):
        vision.usage.record({"prompt_tokens": 10})
        return 1

    vision.answer_from_text = text
    emitted = []
    monkeypatch.setattr(daily, "log_usage", lambda label, usage, rec: emitted.append((label, usage.requests)))
    report = daily.run_daily(config(DailyHybridReader(vision, Ocr(read()), verbose=False), dry_run=True))
    assert report.outcome == "dry_run"
    assert emitted == [("第 1 题", 1)]


def test_expired_preread_has_only_own_short_text_budget(desktop, clock):
    clock.now = 110.0
    vision = Vision()
    reader = DailyHybridReader(vision, Ocr(read()), verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(), ocr_accepted=True, local_attempted=True)
    result = daily.answer_one_question(config(reader), desktop[0], None, 1, capture_context=context)
    assert vision.calls == [("text", 5.0, 115.0)]
    assert result["answer"] == 1 and not result["guessed"]


def test_expired_preread_text_failure_does_not_extend_image_budget(desktop, clock):
    clock.now = 110.0
    vision = Vision()
    vision.text_result = None
    reader = DailyHybridReader(vision, Ocr(read()), verbose=False)
    context = daily.DailyCaptureContext(captured_at=100.0, confirmed_at=100.0,
                                        ocr_read=read(), ocr_accepted=True, local_attempted=True)
    daily.answer_one_question(config(reader), desktop[0], None, 1, capture_context=context)
    assert vision.calls == [("text", 5.0, 115.0)]


def test_wait_counts_actual_requests_in_visual_fallback(desktop):
    from weread_quiz.usage import UsageTracker

    evidence = read()
    evidence.has_progress = False
    vision = Vision()
    vision.usage = UsageTracker()

    def failing_image(*args, **kwargs):
        vision.usage.record()
        vision.usage.record()
        raise AnswerUnavailable("offline")

    vision.read_question = failing_image
    logs = []
    cfg = config(DailyHybridReader(vision, Ocr(evidence), verbose=False),
                 recorder=SimpleNamespace(log=logs.append))
    daily.wait_for_question(cfg, timeout=5)
    assert any("模型请求 2次" in line for line in logs)


@pytest.mark.parametrize("dry_run", [False, True])
def test_daily_log_distinguishes_guess_from_model_answer(desktop, monkeypatch, dry_run):
    class Unavailable(Vision):
        def read_question(self, *args, **kwargs):
            raise AnswerUnavailable("invalid_response")

    logs = []
    recorder = SimpleNamespace(log=logs.append, shot=lambda *args, **kwargs: None)
    monkeypatch.setattr(daily, "_read_final", lambda cfg, **kwargs: "final_pending")
    cfg = config(Unavailable(), max_questions=1, recorder=recorder, dry_run=dry_run)
    report = daily.run_daily(cfg)
    assert report.answered == (0 if dry_run else 1)
    marker = "拟蒙答，未点击" if dry_run else "蒙答"
    assert any(marker in line and "第 1 个按钮" in line
               and "模型没给出合法答案下标" in line for line in logs)
    assert not any("→ ?" in line for line in logs)
