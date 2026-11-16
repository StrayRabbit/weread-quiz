"""道具接入、换题计数和点击后确认的离线回归。"""
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from pyautogui import FailSafeException

from weread_quiz import config as configuration
from weread_quiz.daily import runner as daily
from weread_quiz.daily.props import PropState
from weread_quiz.daily.vision import DailyReport
from weread_quiz.errors import WqError
from weread_quiz.models import Rect
from weread_quiz.record import format_question_line
from weread_quiz.vision import VisionQuestion
from test_daily_pipeline import clock, desktop  # noqa: F401


def cfg(**kwargs):
    return daily.DailyConfig(Rect(0, 0, 430, 788), SimpleNamespace(), verbose=False, **kwargs)


def question(answer=None):
    return VisionQuestion(stem="题干", options=["甲", "乙", "丙"], answer_index=answer,
                          screen="question", progress_no=1)


def state(kind="skip", count=2, enabled=True):
    return PropState(kind, count, enabled, Rect(44 if kind == "skip" else 224, 630, 163, 70))


@pytest.mark.parametrize("value, expected", [(None, False), ("1", True), ("0", False), ("true", True)])
def test_prop_config(monkeypatch, value, expected):
    monkeypatch.setattr(configuration, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(configuration.os, "environ", {} if value is None else {"DAILY_USE_PROPS_ON_FAILURE": value})
    assert configuration.AppConfig.from_env().daily_use_props_on_failure is expected


def test_bad_prop_config(monkeypatch):
    monkeypatch.setattr(configuration, "_load_dotenv", lambda path: None)
    monkeypatch.setenv("DAILY_USE_PROPS_ON_FAILURE", "maybe")
    with pytest.raises(ValueError, match="DAILY_USE_PROPS_ON_FAILURE"):
        configuration.AppConfig.from_env()


@pytest.mark.parametrize("enabled, dry_run, answer", [(False, False, None), (True, True, None), (True, False, 1)])
def test_prop_lookup_only_for_enabled_real_guess(desktop, monkeypatch, enabled, dry_run, answer):
    monkeypatch.setattr(daily, "read_props", lambda *a: pytest.fail("不该读道具"))
    result = daily._answer_one_question_inner(cfg(use_props_on_failure=enabled, dry_run=dry_run),
                                             desktop[0], question(answer))
    assert not result.get("prop")


def test_failed_model_clicks_prop_instead_of_answer(desktop, monkeypatch):
    monkeypatch.setattr(daily, "read_props", lambda *a: [state()])
    config = cfg(use_props_on_failure=True)
    result = daily._answer_one_question_inner(config, desktop[0], question())
    assert result["prop"] == "skip" and result["answer"] is None
    assert not result["guessed"] and desktop[1] == [665]
    # 点击前记账 —— 报告能看见用过一张卡。
    assert len(config.prop_uses) == 1
    assert config.prop_uses[0]["kind"] == "skip"
    assert config.prop_uses[0]["confirmed"] is False


def test_last_question_chooses_swap_even_if_skip_is_bright(desktop, monkeypatch):
    monkeypatch.setattr(daily, "read_props", lambda *a: [state(), state("swap")])
    result = daily._answer_one_question_inner(cfg(use_props_on_failure=True), desktop[0], question(),
                                             last_question=True)
    assert result["prop"] == "swap" and desktop[1] == [665]


def test_no_props_keeps_guess(desktop, monkeypatch):
    monkeypatch.setattr(daily, "read_props", lambda *a: [])
    result = daily._answer_one_question_inner(cfg(use_props_on_failure=True), desktop[0], question())
    assert result["guessed"] and not result.get("prop")


def test_skip_needs_question_number(desktop, monkeypatch):
    """跳题会让题号 +1 —— 题号读不出就不用跳题卡。"""
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip")])
    q = question()
    q.progress_no = None
    result = daily._answer_one_question_inner(cfg(use_props_on_failure=True), desktop[0], q,
                                             progress_no=None)
    assert result["guessed"] and not result.get("prop")


def test_swap_is_used_when_skip_is_unavailable(desktop, monkeypatch):
    """优先级是回退链：跳题卡用不了（题号未知）就改用换题卡，不能整条放弃。"""
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip"), state("swap", count=1)])
    q = question()
    q.progress_no = None
    result = daily._answer_one_question_inner(cfg(use_props_on_failure=True), desktop[0], q,
                                             progress_no=None)
    assert result["prop"] == "swap" and not result["guessed"]


def test_pre_click_recheck_failure_falls_back_to_guess(desktop, monkeypatch):
    """点击前复核不过 → 卡片没点，退回蒙答，不放弃整局。"""
    reads = iter([[state()], [state(count=1)]])
    monkeypatch.setattr(daily, "read_props", lambda *a: next(reads))
    result = daily._answer_one_question_inner(cfg(use_props_on_failure=True), desktop[0], question())
    assert result["guessed"] and not result.get("prop")
    assert desktop[1] == [320]


def test_emergency_abort_on_prop_click_propagates(desktop, monkeypatch):
    monkeypatch.setattr(daily, "read_props", lambda *a: [state()])
    monkeypatch.setattr(daily, "click_box", lambda *a, **k: (_ for _ in ()).throw(FailSafeException()))
    with pytest.raises(FailSafeException):
        daily._answer_one_question_inner(cfg(use_props_on_failure=True), desktop[0], question())


def test_prop_click_error_is_recorded_as_unconfirmed(desktop, monkeypatch):
    """点击抛错 → 记为用过一张卡（可能真被消耗了），整局停止。"""
    monkeypatch.setattr(daily, "read_props", lambda *a: [state()])
    monkeypatch.setattr(daily, "click_box", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    config = cfg(use_props_on_failure=True)
    with pytest.raises(WqError, match="道具点击失败"):
        daily._answer_one_question_inner(config, desktop[0], question())
    assert len(config.prop_uses) == 1 and config.prop_uses[0]["confirmed"] is False


def result(*, prop=None, no=1, count=2):
    payload = dict(stem="题干", options=["甲"], answer=None if prop else 0, guessed=False,
                   bare_index=False, error="", progress_no=no)
    if prop:
        payload.update(prop=prop, prop_count=count)
    return payload


def setup_run(monkeypatch, desktop, results, *, max_questions=2):
    config = cfg(use_props_on_failure=True, max_questions=max_questions)
    context = daily.DailyCaptureContext(captured_at=100, confirmed_at=100,
                                        ocr_read=SimpleNamespace(progress_no=1))
    monkeypatch.setattr(daily, "_wait_first_question", lambda *a, **k: (desktop[0], None, None, 0))
    calls = []
    queue = iter(results)

    def answer(config, image, q, no, *a, **k):
        calls.append((no, q, k["capture_context"]))
        item = next(queue)
        if item.get("prop"):
            # 真实路径在点击前记账 —— 这里照做，报告才有消耗记录。
            config.prop_uses.append({"kind": item["prop"], "question_no": no,
                                     "progress_no": item.get("progress_no"),
                                     "confirmed": False})
        return item

    monkeypatch.setattr(daily, "answer_one_question", answer)
    monkeypatch.setattr(daily, "wait_for_next_question",
                        lambda *a, **k: daily._WaitOutcome("next", desktop[0], captured_at=101))
    monkeypatch.setattr(daily, "_wait_after_prop",
                        lambda *a, **k: daily._WaitOutcome("next", desktop[0], captured_at=101))
    monkeypatch.setattr(daily, "_read_final", lambda *a, **k: "success")
    return config, calls


def test_swap_retries_same_slot_and_resets_capture_context(desktop, monkeypatch):
    config, calls = setup_run(monkeypatch, desktop,
                             [result(prop="swap"), result(prop="swap"), result(), result(no=2)])
    report = daily.run_daily(config)
    assert report.outcome == "success" and report.answered == 2
    assert [c[0] for c in calls] == [1, 1, 1, 2]
    assert len(report.questions) == 2 and len(report.prop_uses) == 2
    assert calls[0][2] is not calls[1][2] and calls[1][2] is not calls[2][2]
    assert all(c[1] is None for c in calls)
    assert report.progress_numbers == [1, 2]


def _card_frame(*, stem=True, option_text=True):
    """一张卡片帧：题号带 + 题干 + 选项按钮（选项文字可选）。"""
    image = Image.new("RGB", (430, 788), (7, 40, 79))
    draw = ImageDraw.Draw(image)
    draw.rectangle((170, 202, 240, 207), fill="white")
    if stem:
        draw.rectangle((100, 240, 320, 250), fill="white")
    for top in (310, 380, 450):
        draw.rectangle((40, top, 390, top + 44), fill=(42, 59, 74))
        if option_text:
            draw.rectangle((200, top + 15, 230, top + 25), fill="white")
    return image


def test_swap_waits_for_new_question_to_settle(desktop, monkeypatch):
    """换题后新题面还在渲染 —— 半渲染帧不能算换题成功。

    早一帧确认会把没画完的题面交出去：点击前复核会认为题面变了，
    那一题就白丢了（2026-11-12 离线模拟复现）。
    """
    before = _card_frame(stem=False, option_text=False)
    half = _card_frame(option_text=False)
    settled = _card_frame()
    frames = iter([half, settled, settled.copy()])
    monkeypatch.setattr(daily, "grab_screen", lambda region: next(frames, settled))
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: True)
    monkeypatch.setattr(daily, "is_question_screen", lambda image: True)
    monkeypatch.setattr(daily, "is_question_stable",
                        lambda a, b: a.tobytes() == b.tobytes())
    counts = iter([1, 1, 1])
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=next(counts))])
    config = cfg()
    config.vision.read_progress_number = lambda image: 1

    outcome = daily._wait_after_prop(config, before, result(prop="swap", no=1, count=2))

    assert outcome.status == "next"
    assert outcome.image.tobytes() == settled.tobytes(), "交出去的是渲染完成的帧"


def test_swap_rejects_changed_question_number(desktop, monkeypatch):
    """换题留在同一题位 —— 题号变了说明是自然进入下一题，不是换题成功。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 2
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: True)
    monkeypatch.setattr(daily, "is_question_screen", lambda image: True)
    counts = iter([2, 1, 1])
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=next(counts))])

    outcome = daily._wait_after_prop(config, desktop[0], result(prop="swap", no=1, count=2))

    assert outcome.status == "interrupted"
    assert "题号" in outcome.reason


def test_skip_waits_for_new_question_to_settle(desktop, monkeypatch):
    """跳题后的新题面同样要稳定 —— 半渲染帧会让下一题点不出去。"""
    before = _card_frame(stem=False, option_text=False)
    half = _card_frame(option_text=False)
    settled = _card_frame()
    frames = iter([half, settled, settled.copy()])
    monkeypatch.setattr(daily, "grab_screen", lambda region: next(frames, settled))
    monkeypatch.setattr(daily, "is_question_screen", lambda image: True)
    monkeypatch.setattr(daily, "is_question_stable",
                        lambda a, b: a.tobytes() == b.tobytes())
    counts = iter([1, 1, 1])
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=next(counts))])
    config = cfg()
    config.vision.read_progress_number = lambda image: 2

    outcome = daily._wait_after_prop(config, before, result(prop="skip", no=1))

    assert outcome.status == "next"
    assert outcome.image.tobytes() == settled.tobytes(), "交出去的是渲染完成的帧"


def test_skip_is_not_an_answer_click(desktop, monkeypatch):
    config, calls = setup_run(monkeypatch, desktop, [result(prop="skip"), result(no=2)])
    report = daily.run_daily(config)
    assert report.outcome == "success" and report.answered == 1
    assert report.prop_skipped == 1 and report.skipped == 0
    assert [c[0] for c in calls] == [1, 2]
    assert len(report.questions) == 2


def test_swap_at_last_slot_does_not_settle_until_answered(desktop, monkeypatch):
    config, calls = setup_run(monkeypatch, desktop, [result(prop="swap"), result()], max_questions=1)
    settlements = []
    monkeypatch.setattr(daily, "_read_final", lambda *a, **k: settlements.append(True) or "success")
    report = daily.run_daily(config)
    assert [c[0] for c in calls] == [1, 1]
    assert report.answered == 1 and settlements == [True]


def test_unconfirmed_prop_stops_without_retry_or_guess(desktop, monkeypatch):
    config, calls = setup_run(monkeypatch, desktop, [result(prop="swap")])
    monkeypatch.setattr(daily, "_wait_after_prop", lambda *a, **k: daily._WaitOutcome("interrupted", reason="未生效"))
    report = daily.run_daily(config)
    assert report.outcome == "interrupted" and report.answered == 0
    assert len(calls) == 1 and not report.prop_uses[0]["confirmed"]


def test_skip_is_confirmed_by_question_number(desktop, monkeypatch):
    """跳题的正面证据是题号 +1 —— 不要求库存当场扣减。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 2
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=2)])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="skip", no=1))
    assert outcome.status == "next"


def test_skip_without_number_change_is_not_success(desktop, monkeypatch):
    config = cfg()
    config.vision.read_progress_number = lambda image: 1
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=2)])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="skip", no=1))
    assert outcome.status == "interrupted"


def test_swap_needs_stem_change_and_stock_decrement(desktop, monkeypatch):
    """换题要两样证据：题干变了（本地像素判据）+ 库存扣减。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 1
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: True)
    counts = iter([2, 1])
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=next(counts))])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="swap", no=1, count=2))
    assert outcome.status == "next"


def test_swap_without_stock_decrement_is_not_success(desktop, monkeypatch):
    """题干变了但库存没少 —— 不能算换题成功（点击可能没生效）。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 1
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: True)
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=2)])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="swap", no=1, count=2))
    assert outcome.status == "interrupted"


def test_swap_without_stem_change_is_not_success(desktop, monkeypatch):
    """题干没变 —— 根本没换题。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 1
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: False)
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=1)])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="swap", no=1, count=2))
    assert outcome.status == "interrupted"


def test_swap_falls_back_to_stock_decrement_when_number_unreadable(desktop, monkeypatch):
    """末题显示「最后一题」读不出编号 —— 库存扣减是剩下的证据。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: None
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: True)
    counts = iter([2, 1])
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=next(counts))])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="swap", no=None, count=2))
    assert outcome.status == "next"


def test_swap_without_any_evidence_is_not_success(desktop, monkeypatch):
    config = cfg()
    config.vision.read_progress_number = lambda image: None
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: True)
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=2)])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="swap", no=None, count=2))
    assert outcome.status == "interrupted"


def test_prop_click_that_does_nothing_stops_the_run(desktop, monkeypatch):
    """点击没生效 → 题号没变、库存没少 → 停整局，不误点下一题。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 1
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("swap", count=2)])
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="swap", no=1, count=2))
    assert outcome.status == "interrupted"


def test_skip_needs_two_consecutive_number_reads(desktop, monkeypatch):
    """只读到一次题号不算数 —— 单次 OCR 误读不该决定跳题成败。"""
    config = cfg()
    numbers = iter([2, 7])
    config.vision.read_progress_number = lambda image: next(numbers)
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=2)])
    confirmed, reason, _frame, _at = daily._confirm_prop_effect(
        config, "skip", expected_number=2, expected_count=2)
    assert confirmed is False and "题号" in reason


def test_skip_confirms_when_number_changes_late(desktop, monkeypatch):
    """题号晚点才变也要等到 —— 2 秒上限会把成功的跳题判成失败。

    实测 2026-11-10：点击后 1.9 秒题号才变成 11，旧上限正好卡在
    边界上，差一次读数就整局中断。
    """
    config = cfg()
    numbers = iter([1] * 30 + [2] * 10)
    config.vision.read_progress_number = lambda image: next(numbers)
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=2)])
    confirmed, reason, _frame, _at = daily._confirm_prop_effect(
        config, "skip", expected_number=2, expected_count=2)
    assert confirmed is True and "题号" in reason


def test_skip_stock_drop_confirms_single_number_read(desktop, monkeypatch):
    """库存已经扣减时题号读到一次就够 —— 两份独立证据。"""
    config = cfg()
    numbers = iter([2, 7])
    config.vision.read_progress_number = lambda image: next(numbers)
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=1)])
    confirmed, _reason, _frame, _at = daily._confirm_prop_effect(
        config, "skip", expected_number=2, expected_count=2)
    assert confirmed is True


def test_skip_waits_for_toast_to_clear(desktop, monkeypatch):
    """提示条压着题号行 —— 它还在就不算生效。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 2
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=1)])
    toasts = iter([True, True, False])
    monkeypatch.setattr(daily, "has_toast", lambda image: next(toasts, False))
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="skip", no=1))
    assert outcome.status == "next"


def test_prop_frame_handed_over_has_no_toast(desktop, monkeypatch):
    """交出去的帧必须没有提示条 —— 题干上盖着提示条会让点击前复核
    误判成"题面变了"，那道题就白丢了。"""
    config = cfg()
    config.vision.read_progress_number = lambda image: 2
    monkeypatch.setattr(daily, "read_props", lambda *a: [state("skip", count=1)])
    clean_frame = Image.new("RGB", (430, 788), (7, 40, 79))
    toast_frame = clean_frame.copy()
    ImageDraw.Draw(toast_frame).rectangle((140, 200, 290, 265), fill=(1, 8, 16))
    frames = iter([toast_frame, clean_frame, clean_frame])
    monkeypatch.setattr(daily, "grab_screen", lambda region: next(frames, clean_frame))
    outcome = daily._wait_after_prop(config, desktop[0], result(prop="skip", no=1))
    assert outcome.status == "next" and outcome.image is clean_frame

    line = format_question_line(result(prop="skip"), ui_number=1, question_no=1)
    assert "跳题卡" in line and line.endswith("-") and "✓" not in line
    report = DailyReport(answered=11, prop_skipped=1, outcome="success",
                         prop_uses=[{"kind": "skip", "confirmed": True},
                                    {"kind": "swap", "confirmed": False}])
    assert "全部答对" not in report.summary()
    assert "跳题卡 1" in report.prop_summary()
    assert "未确认 1" in report.prop_summary()
