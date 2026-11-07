"""每日一答循环：本地确认、预读复用、限时识别、点击前复核及独立终态确认。"""
from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image
from pyautogui import FailSafeException

from .. import abort
from ..answer_feedback import SAMPLE_OFFSETS, AnswerFeedback
from ..capture import BLANK_REFRESH_ROUNDS, grab_screen, is_blank_frame
from ..click import click_box
from .buttons import find_option_buttons, pick_button
from .screen import (
    has_answer_reveal,
    has_progress_line,
    is_new_question,
    is_question_screen,
    is_question_stable,
    is_same_question,
)
from .vision import (
    DAILY_TOTAL_QUESTIONS,
    DailyReport,
    read_daily_answer,
    read_daily_result,
    text_path_deadline,
)
from ..errors import AnswerUnavailable, NotAQuestionScreen, OcrRejected, RecognitionTimeout, WqError
from ..record import format_question_line
from ..usage import PhaseTotals, Stopwatch, log_usage, safe_log, usage_since, usage_snapshot
from ..vision import VisionQuestion, VisionService, _check_deadline

# daily 无可读取的倒计时，仍保留固定 9 秒识别预算。
# 文本使用剩余预算，超时直接交点击前复核及蒙答，不预扣图片回退时间。
_IDENTIFY_BUDGET_SECONDS = 9.0
_TEXT_TIMEOUT = 9.0

# 仅本地判否或文本提前返回无效结果时考虑图片；余量不足则不发请求。
_IMAGE_PATH_MIN_BUDGET = 3.5

# 图龄超过这个值就**多问一次题号**，不是把答案作废。
#
# 以前这里是"超时即作废"（8 秒），而单题识别预算是 9 秒 —— 预算比
# 上限大，**图片路径一慢答案就必然被丢掉**（2026-11-06 实测：模型
# 7.45 秒才回，第 1 题被扔掉，整局输在第一题）。
#
# 现在改成按题号复核：题号是界面自己写的身份标记，题号没变就还是
# 那道题，答案照样能用（见 `_still_same_question`）。
_MAX_CAPTURE_AGE = 8.0
_LOCAL_FALLBACK_SECONDS = 3.0

# 等待进入答题界面的总超时
_WAIT_QUESTION_TIMEOUT = 120.0

# 轮询间隔
_POLL_INTERVAL = 0.25

_NEXT_QUESTION_TIMEOUT = 19.0
_INITIAL_WAIT_MAX = 6.0
_EXTRA_WAIT_MAX = 3.0
_REVEAL_WAIT_MAX = 13.0
_ABNORMAL_WAIT_MAX = 3.0
_OCR_MIN_ABNORMAL_SECONDS = 1.0
_OCR_MIN_ABNORMAL_FRAMES = 2

_FINAL_SETTLE_SECONDS = 3.0

# 结算等待期间的取帧时点（相对点击，单位秒）。
#
# 揭示色要等服务端回包才出现（PK 实测 0.2~0.5 秒），daily 同一套界面但服务端
# 可能更慢 —— 后半段取稀一点，把 3 秒等待都盖住。
_FINAL_SAMPLE_OFFSETS = SAMPLE_OFFSETS + (1.2, 1.5, 1.8, 2.1, 2.4, 2.7)


@dataclass
class DailyCaptureContext:
    """同一张截图的采集时刻与本地证据；确认时间不刷新原图年龄。"""

    captured_at: float | None = None
    confirmed_at: float | None = None
    ocr_read: object | None = None
    ocr_accepted: bool = False
    local_attempted: bool = False
    polls: int = 0


@dataclass
class _WaitOutcome:
    """换题等待结果：next 带已确认的新题帧；revive/success/fail 须有文字证据，未知返回 interrupted。"""

    status: str
    image: Image.Image | None = None
    reason: str = ""
    captured_at: float | None = None


@dataclass
class DailyConfig:
    region: object  # Rect
    vision: VisionService
    recorder: object | None = None
    verbose: bool = True
    max_questions: int = DAILY_TOTAL_QUESTIONS
    dry_run: bool = False
    ocr: object | None = None
    # 抓到空白帧时的急救回调（重新激活小程序窗口）。
    refresh: object | None = None

    def __post_init__(self) -> None:
        # region 类型在运行时是 Rect；这里只做属性存在性检查
        for attr in ("left", "top", "width", "height"):
            if not hasattr(self.region, attr):
                raise TypeError(f"DailyConfig.region 缺少 {attr} 属性")


def wait_for_question(cfg: DailyConfig, timeout: float | None = None, *,
                      capture_context: DailyCaptureContext | None = None):
    """本地确认首题，返回 (原图, 可选答案)；模型答题交给逐题阶段。"""
    from .screen import has_rendered_stem

    context = capture_context if capture_context is not None else DailyCaptureContext()
    context.captured_at = context.confirmed_at = None
    context.ocr_read = None
    context.ocr_accepted = context.local_attempted = False
    context.polls = 0
    timeout = _WAIT_QUESTION_TIMEOUT if timeout is None else timeout
    started = time.monotonic()
    deadline = started + timeout
    previous = None
    rejected_since = None
    fallback_used = False
    # 连续读到空白帧的轮数（窗口还没把内容画出来）。
    blank_rounds = 0
    totals = PhaseTotals()
    local = getattr(cfg.vision, "read_question_local", None)
    local_available = callable(local) and getattr(cfg.vision, "local_ocr_available", None) is not False

    def finish(image=None, question=None):
        safe_log(cfg.recorder, f"等待阶段明细：{totals.format()}"
                 f"（共 {(time.monotonic() - started) * 1000:.0f}ms/{context.polls}轮）")
        return image, question

    while time.monotonic() < deadline:
        abort.check()
        captured_at = time.monotonic()
        image = grab_screen(cfg.region)
        context.polls += 1
        if is_blank_frame(image):
            # 窗口还没把内容画出来（GPU 合成窗口失去焦点会停止渲染）——
            # 空白帧认不出题目，记着连续几次好去叫醒窗口。
            blank_rounds += 1
            if blank_rounds >= BLANK_REFRESH_ROUNDS and cfg.refresh is not None:
                blank_rounds = 0
                safe_log(cfg.recorder, "等待：连续读到空白帧，重新激活小程序窗口")
                cfg.refresh()
        else:
            blank_rounds = 0
        ready = (is_question_screen(image) and has_rendered_stem(image)
                 and not has_answer_reveal(image))
        totals.add("抓屏+门控", time.monotonic() - captured_at, unit="轮")
        if not ready:
            previous = None
            rejected_since = None
        elif previous is not None and is_question_stable(previous, image):
            context.captured_at = captured_at
            if not local_available:
                context.confirmed_at = time.monotonic()
                return finish(image)
            context.local_attempted = True
            local_started = time.monotonic()
            try:
                evidence = local(image)
                accepted = True
            except OcrRejected as exc:
                evidence, accepted = exc.read, False
                safe_log(cfg.recorder, f"等待：本地门控判否（{exc}）")
            finally:
                totals.add("本地OCR", time.monotonic() - local_started, unit="次")
            if time.monotonic() >= deadline:
                break
            if evidence is not None and evidence.has_progress:
                context.ocr_read = evidence
                context.ocr_accepted = accepted
                context.confirmed_at = time.monotonic()
                return finish(image)
            if rejected_since is None:
                rejected_since = time.monotonic()
            if not fallback_used and time.monotonic() - rejected_since >= _LOCAL_FALLBACK_SECONDS:
                fallback_used = True
                model_started = time.monotonic()
                request_start = usage_snapshot(cfg.vision)
                try:
                    question = read_daily_answer(
                        cfg.vision, image, visual_only=True,
                        timeout=5.0, deadline=min(deadline, model_started + 5.0),
                    )
                except (AnswerUnavailable, NotAQuestionScreen) as exc:
                    safe_log(cfg.recorder, f"等待：单次图片兜底失败（{exc}）")
                else:
                    if time.monotonic() < deadline:
                        context.confirmed_at = time.monotonic()
                        return finish(image, question)
                finally:
                    totals.add("模型请求", time.monotonic() - model_started,
                               count=usage_since(cfg.vision, request_start).requests, unit="次")
            previous = image
        else:
            previous = image
        poll_started = time.monotonic()
        time.sleep(_POLL_INTERVAL)
        totals.add("轮询等待", time.monotonic() - poll_started, unit="次")
    if cfg.verbose:
        print(f"\n  等待 {timeout:.0f}s 仍未进入答题界面")
    return finish()


def _identify_question(cfg, image, context):
    """可靠预读优先文本；文本到时直接蒙答，仅提前无效结果或本地判否可回退。"""
    anchor = context.confirmed_at if context.confirmed_at is not None else context.captured_at
    deadline = (anchor if anchor is not None else time.monotonic()) + _IDENTIFY_BUDGET_SECONDS
    _check_deadline(deadline)
    safe_log(cfg.recorder, f"识别预算：总 {_IDENTIFY_BUDGET_SECONDS:g}s，"
             f"当前余量 {max(0.0, deadline - time.monotonic()):.2f}s"
             f"（文本上限 {_TEXT_TIMEOUT:g}s，图片下限 {_IMAGE_PATH_MIN_BUDGET:g}s）")
    local = getattr(cfg.vision, "read_question_local", None)
    if not context.local_attempted and callable(local) and getattr(cfg.vision, "local_ocr_available", None) is not False:
        context.local_attempted = True
        try:
            context.ocr_read = local(image, deadline=deadline)
            context.ocr_accepted = True
        except OcrRejected as exc:
            context.ocr_read = exc.read
            context.ocr_accepted = False
            safe_log(cfg.recorder, f"图片路径 回退原因：{exc}")
    read = context.ocr_read
    usable = (context.ocr_accepted and read is not None
              and len(read.options) == len(find_option_buttons(image)))
    answer = getattr(cfg.vision, "answer_from_ocr_read", None)
    if usable and callable(answer):
        now = time.monotonic()
        text_deadline = text_path_deadline(deadline, now, _TEXT_TIMEOUT)
        _check_deadline(text_deadline)
        try:
            question = answer(read, timeout=text_deadline - now, deadline=text_deadline)
        except RecognitionTimeout:
            raise  # 文本到时直接交上层同题复核及蒙答，不再请求图片。
        except AnswerUnavailable as exc:
            safe_log(cfg.recorder, f"OCR 文本路径失败：{exc}")
            question = None
        _check_deadline(text_deadline)
        if question is not None:
            return question
    _check_deadline(deadline)
    # 图片路径值不值得试：**看还剩多少预算**。不够就别发 —— 发了也只是
    # 再等一次超时，然后照旧蒙答（与 PK 的判据一致）。
    remaining = deadline - time.monotonic()
    if remaining < _IMAGE_PATH_MIN_BUDGET:
        message = (f"图片路径 跳过：剩余预算 {remaining:.2f}s "
                   f"< {_IMAGE_PATH_MIN_BUDGET:.1f}s，走不了图片路径")
        safe_log(cfg.recorder, message)
        raise RecognitionTimeout(message)
    return read_daily_answer(cfg.vision, image, timeout=_IDENTIFY_BUDGET_SECONDS,
                             deadline=deadline, visual_only=context.local_attempted)


def _expected_progress_no(question, context) -> int | None:
    """这道题的题号：题面读数优先，其次模型读数。

    **读不出就是 None，绝不猜** —— 题号猜错会把另一道题的答案点出去。
    """
    for source in (question, getattr(context, "ocr_read", None)):
        number = getattr(source, "progress_no", None)
        if number is not None:
            return number
    return None


def _still_same_question(cfg, question, context, fresh) -> bool:
    """按题号复核：屏幕上现在还是刚才那道题吗？

    题号（「第N题·共12题」）是界面自己写的身份标记，比"截图是几秒前的"
    可靠 —— 截图可能过期，但题号没变就还是那道题，模型给的下标仍然
    对得上那个按钮。

    三条都要成立：
    1. 题号行还在（还在答题卡片上，没跑到别的页面）；
    2. 题面没揭示（揭示 = 已答或已超时，这道题已经结束了）；
    3. 题号和刚才那道题一致。
    """
    if not has_progress_line(fresh) or has_answer_reveal(fresh):
        return False
    expected = _expected_progress_no(question, context)
    if expected is None:
        return False
    reader = getattr(cfg.vision, "read_progress_number", None)
    if not callable(reader):
        return False
    try:
        current = reader(fresh)
    except Exception:  # noqa: BLE001 - 读不出当"不是同一道题"
        return False
    return current == expected


def _skip_result(question, reason):
    return {"stem": getattr(question, "stem", ""), "options": getattr(question, "options", []),
            "answer": None, "guessed": False, "bare_index": False,
            "error": reason, "skipped": True}


def answer_one_question(
    cfg: DailyConfig,
    image: Image.Image,
    question,
    question_no: int,
    watch: Stopwatch | None = None,
    *,
    capture_context: DailyCaptureContext | None = None,
    feedback: AnswerFeedback | None = None,
) -> dict:
    """识别并处理当前题，失败只在同题复核通过后蒙第一个；跳过留证，紧急中止向外传播。

    终端输出统一由 run_daily 的逐题行负责（那一行还要带上答后反馈），
    这里只把结果放进返回值 —— 不单独打印。
    """
    not_question = False
    context = capture_context if capture_context is not None else DailyCaptureContext()
    try:
        if context.captured_at is None and not cfg.dry_run:
            return _skip_result(question, "capture_time_unknown")
        if question is None:
            try:
                question = _identify_question(cfg, image, context)
            except AnswerUnavailable as exc:
                safe_log(cfg.recorder, f"模型识别失败，交点击前复核：{exc}")
                question = VisionQuestion(stem=getattr(context.ocr_read, "stem", ""),
                                          options=[], answer_index=None, screen="question")
            if watch:
                watch.mark("模型识别")
        if (not cfg.dry_run and not question.local_read
                and time.monotonic() - context.captured_at >= _MAX_CAPTURE_AGE):
            # 图龄超限**不再直接丢答案** —— 按题号复核一次。
            #
            # 题号是界面自己写的身份标记：截图可能过期，但题号没变就还是
            # 那道题，模型给的下标仍然对得上那个按钮；题号变了就不是同一
            # 道题，照旧不能点。
            #
            # 复核通过就用这一帧继续（点击前的同题校验还会再核一遍）。
            fresh = grab_screen(cfg.region)
            age = time.monotonic() - context.captured_at
            if not _still_same_question(cfg, question, context, fresh):
                safe_log(cfg.recorder,
                         f"图龄 {age:.1f}s，按题号复核未通过"
                         "（题号变了/题面已揭示/题号读不出），未点击")
                return _skip_result(question, "capture_too_old")
            safe_log(cfg.recorder,
                     f"图龄 {age:.1f}s，按题号复核仍是第"
                     f"{_expected_progress_no(question, context)} 题，继续作答")
            image = fresh
        result = _answer_one_question_inner(cfg, image, question, watch,
                                            feedback=feedback)
        result["progress_no"] = (getattr(question, "progress_no", None)
                                 or getattr(context.ocr_read, "progress_no", None))
        return result
    except NotAQuestionScreen:
        not_question = True
        raise
    finally:
        # 只保存模型输入原图；跳过也保留已知图。
        if not not_question and cfg.recorder and image is not None:
            _safe_shot(cfg, f"q{question_no:02d}_before", image)
            if watch:
                watch.mark("截图落盘")


def _answer_one_question_inner(
    cfg: DailyConfig,
    image: Image.Image,
    question,
    watch: Stopwatch | None = None,
    *, feedback: AnswerFeedback | None = None,
) -> dict:
    """答题本体。留证由 answer_one_question 的 finally 负责。

    终端不单独打印 —— 逐题行由 run_daily 统一打（那一行还要带上答后反馈）。
    """
    if not is_question_screen(image):
        # 几何只用于诊断描述；run_daily必须另读本地文字，不靠异常消息判错。
        raise NotAQuestionScreen(
            "复活页：不是答题界面，不能点"
            if _looks_like_revive_page(image)
            else "不是答题界面，不能点"
        )
    boxes = find_option_buttons(image)
    answer = question.answer_index
    guessed = False
    error = ""
    if answer is None:
        answer, guessed, error = 0, True, "模型没给出合法答案下标"
    elif not (0 <= answer < len(boxes)):
        answer, guessed, error = 0, True, f"模型答案下标越界（{len(boxes)} 个按钮）"
    elif len(question.options) not in (0, len(boxes)):
        # 数量不符说明至少一边认错，蒙第一个比不动强
        answer, guessed, error = 0, True, f"模型选项数 {len(question.options)} ≠ 按钮数 {len(boxes)}"
    box = pick_button(boxes, answer)
    if not cfg.dry_run:
        try:
            fresh = grab_screen(cfg.region)
            if not is_same_question(image, fresh):
                if watch:
                    watch.mark("点击前校验(跳过)")
                return {
                    "stem": question.stem,
                    "options": question.options,
                    "answer": answer,
                    "clicked_y": box.center_y,
                    "guessed": guessed,
                    "bare_index": question.bare_index,
                    "error": "screen_changed",
                    "skipped": True,
                }
            # 复核通过后**必须用 fresh 图重新定位**（PK 的做法，
            # pk.py:589）—— fresh 图上的按钮位置才是当前真实位置。
            fresh_boxes = find_option_buttons(fresh)
            box = pick_button(fresh_boxes, answer)
            if watch:
                watch.mark("点击前校验")
        except FailSafeException:
            raise
        except Exception as exc:  # noqa: BLE001
            if watch:
                watch.mark("点击前校验(失败)")
            return {
                "stem": question.stem,
                "options": question.options,
                "answer": answer,
                "clicked_y": box.center_y,
                "guessed": guessed,
                "bare_index": question.bare_index,
                "error": f"validation_failed: {exc}",
                "skipped": True,
            }
        click_box(box, cfg.region, pause=0.15)
        if feedback is not None:
            # 非末题：等待阶段每 0.25 秒就有一次取帧，不额外抓屏。
            feedback.start(fresh, box)
            if feedback.last_question:
                # 末题之后直接进结算页，颜色只亮一瞬：立刻取一帧。
                feedback.observe_now(lambda: grab_screen(cfg.region))
        if watch:
            watch.mark("蒙题点击" if guessed else "点击")
    if watch:
        watch.mark("定位按钮")
    return {
        "stem": question.stem,
        "options": question.options,
        "answer": answer,
        "clicked_y": box.center_y,
        "guessed": guessed,
        "bare_index": question.bare_index,
        "error": error,
    }


def _looks_like_revive_page(image: Image.Image) -> bool:
    """疑似终态的几何筛选：无题号、非揭示态且有按钮色块；不能据此判答错。"""
    from .buttons import _detect_any_buttons

    if has_progress_line(image):
        return False
    if has_answer_reveal(image):
        return False
    # 复活页可能只有一条深蓝灰带，不能要求选项定位器找到至少两个按钮。
    return _detect_any_buttons(image)


def _is_next_question_ready(candidate, current_image) -> bool:
    """依次排除揭示态、确认题干已换，再检查题号及按钮；返回后仍须连续两帧稳定。"""
    if has_answer_reveal(candidate):
        return False
    if current_image is not None and not is_new_question(current_image, candidate):
        return False
    return is_question_screen(candidate)


def _terminal_ocr(cfg: DailyConfig):
    """只使用显式注入的读取器；None不自动发现，也不初始化引擎。"""
    candidate = cfg.ocr
    return candidate if callable(getattr(candidate, "read_terminal", None)) else None


def _frames_consistent(before: Image.Image, after: Image.Image) -> bool:
    """保守工程策略：同尺寸原始灰度像素严格相等，不缩放、不用全图均差。"""
    try:
        return (before.size == after.size
                and before.convert("L").tobytes() == after.convert("L").tobytes())
    except Exception:  # noqa: BLE001
        return False


def _safe_shot(cfg: DailyConfig, label: str, image: Image.Image | None) -> None:
    if cfg.recorder and image is not None:
        try:
            cfg.recorder.shot(label, image=image)
        except Exception as exc:  # noqa: BLE001 - 留证不能覆盖业务结果
            safe_log(cfg.recorder, f"截图失败（{label}）：{type(exc).__name__}: {exc}")


def _confirm_terminal(cfg: DailyConfig, frame: Image.Image,
                      watch: Stopwatch | None = None):
    """先排除题号/揭示态，再一次本地OCR后重抓，返回(可选结论, 最新实际帧)。"""
    from .buttons import _detect_any_buttons

    reader = _terminal_ocr(cfg)
    if reader is None or has_progress_line(frame) or has_answer_reveal(frame):
        return None, frame
    started = time.monotonic()
    terminal = None
    detail = "无明确文字"
    try:
        terminal = reader.read_terminal(frame)
        if terminal is not None:
            detail = getattr(terminal, "kind", "未知类别")
    except Exception as exc:  # noqa: BLE001
        detail = f"异常 {type(exc).__name__}: {exc}"
    # 无论OCR是否有结论，都立即重抓。
    try:
        fresh = grab_screen(cfg.region)
    except Exception as exc:  # noqa: BLE001
        safe_log(cfg.recorder, f"终态OCR后重抓失败：{exc}")
        fresh = None
    elapsed = time.monotonic() - started
    safe_log(cfg.recorder, f"终态OCR 1次，耗时 {elapsed:.3f}s：{detail}")
    if watch:
        watch.mark(f"终态OCR({elapsed:.3f}s)")
    if fresh is None or terminal is None:
        return None, fresh
    if (not _frames_consistent(frame, fresh)
            or has_progress_line(fresh) or has_answer_reveal(fresh)
            or (_detect_any_buttons(fresh) and not _looks_like_revive_page(fresh))):
        safe_log(cfg.recorder, "终态OCR结论已过期：页面内容变化或恢复题面")
        return None, fresh
    kind = getattr(terminal, "kind", None)
    if kind not in ("revive", "success", "fail"):
        return None, fresh
    evidence = getattr(terminal, "evidence", "") or ""
    return _WaitOutcome(kind, fresh, f"读到明确文字「{evidence}」"), fresh


def wait_for_next_question(cfg: DailyConfig, current_image: Image.Image,
                          watch: Stopwatch | None = None, *,
                          feedback: AnswerFeedback | None = None) -> _WaitOutcome:
    """各阶段额度仅一次；OCR后fresh同轮处理，next要求连续两帧。"""
    from .buttons import _detect_any_buttons

    start = time.monotonic()
    hard_deadline = start + _NEXT_QUESTION_TIMEOUT
    previous = last_seen = None
    captured_at = None
    polls = no_buttons = same_stem = 0
    initial_used = normal_used = reveal_used = abnormal_used = 0.0
    abnormal_since = None
    abnormal_frames = 0
    revealed = saw_reveal = ocr_used = False
    ocr_count = 0
    ocr_seconds = 0.0
    last_at = start
    # 连续读到空白帧的轮数（窗口还没把内容画出来）。
    blank_rounds = 0

    def finish(status, image, reason):
        if status != "next":
            _safe_shot(cfg, "stop_status", image)
        safe_log(cfg.recorder, f"等待终止：{status}，{reason}；"
                 f"OCR {ocr_count}次/{ocr_seconds:.3f}s，等待 {time.monotonic() - start:.3f}s")
        if watch:
            watch.mark(f"等下一题({polls}帧 无按钮{no_buttons} 题干未变{same_stem}，{reason})")
        return _WaitOutcome(status, image, reason, captured_at=captured_at)

    safe_log(cfg.recorder, "进入等待下一题：总19s，初始6s，揭示13s/后转场3s，异常累计3s")
    while True:
        abort.check()
        if time.monotonic() >= hard_deadline:
            break
        time.sleep(_POLL_INTERVAL)
        captured_at = time.monotonic()
        candidate = grab_screen(cfg.region)
        blank = is_blank_frame(candidate)
        if blank:
            # 空白帧：窗口还没把内容画出来 —— 记着连续几次好去叫醒窗口。
            blank_rounds += 1
            if blank_rounds >= BLANK_REFRESH_ROUNDS and cfg.refresh is not None:
                blank_rounds = 0
                safe_log(cfg.recorder, "等下一题：连续读到空白帧，重新激活小程序窗口")
                cfg.refresh()
        else:
            blank_rounds = 0
        if feedback is not None and not blank:
            # 空白帧没有颜色信息，拿去观察只会误判成"题干变了"。
            feedback.observe(candidate)
        polls += 1
        last_seen = candidate
        now = time.monotonic()
        delta = now - last_at
        last_at = now
        if now >= hard_deadline:
            break
        has_buttons = _detect_any_buttons(candidate)
        abnormal = not has_buttons or _looks_like_revive_page(candidate)
        if not has_buttons:
            no_buttons += 1
        if abnormal:
            previous = None
            if abnormal_since is None:
                abnormal_since = now
                abnormal_frames = 0
                _safe_shot(cfg, "transition_unknown", candidate)
                safe_log(cfg.recorder, "进入异常等待：无按钮或疑似复活页，尚无文字证据")
            abnormal_used += delta
            abnormal_frames += 1
            if (not ocr_used and abnormal_frames >= _OCR_MIN_ABNORMAL_FRAMES
                    and now - abnormal_since >= _OCR_MIN_ABNORMAL_SECONDS
                    and not has_progress_line(candidate) and not has_answer_reveal(candidate)):
                ocr_used = True
                ocr_count = int(_terminal_ocr(cfg) is not None)
                ocr_start = time.monotonic()
                verdict, fresh = _confirm_terminal(cfg, candidate, watch)
                now = time.monotonic()
                ocr_seconds = now - ocr_start
                abnormal_used += ocr_seconds
                last_at = now
                if fresh is not None:
                    candidate = last_seen = fresh
                    if feedback is not None:
                        feedback.observe(candidate)
                # 同步OCR不能硬中断，超期返回后只能interrupted。
                if now >= hard_deadline:
                    break
                if fresh is None:
                    return finish("interrupted", last_seen, "终态OCR后无法重抓，当前页面无法确认")
                if verdict is not None:
                    return finish(verdict.status, fresh, verdict.reason)
                has_buttons = _detect_any_buttons(candidate)
                abnormal = not has_buttons or _looks_like_revive_page(candidate)
                delta = 0.0  # OCR用时不能转嫁给随后正常候选
            if abnormal:
                if abnormal_used >= _ABNORMAL_WAIT_MAX:
                    return finish("interrupted", last_seen,
                                  f"等待下一题超时（异常累计 {abnormal_used:.1f}s），当前页面无法确认")
                continue
        elif abnormal_since is not None:
            abnormal_used += delta
            delta = 0.0
        if abnormal_since is not None:
            safe_log(cfg.recorder, "异常等待恢复：重新检查最新题面/揭示态")
        abnormal_since = None
        abnormal_frames = 0
        now_revealed = has_answer_reveal(candidate)
        if now_revealed != revealed:
            revealed = now_revealed
            safe_log(cfg.recorder, f"等待{'进入' if revealed else '离开'}揭示态")
        if revealed:
            saw_reveal = True
            reveal_used += delta
            budget_used, budget_max = reveal_used, _REVEAL_WAIT_MAX
        elif saw_reveal:
            normal_used += delta
            budget_used, budget_max = normal_used, _EXTRA_WAIT_MAX
        else:
            initial_used += delta
            budget_used, budget_max = initial_used, _INITIAL_WAIT_MAX
        if not _is_next_question_ready(candidate, current_image):
            same_stem += 1
            previous = None
        elif previous is not None and is_question_stable(previous, candidate):
            return finish("next", candidate, "换到下一题")
        else:
            previous = candidate
        if budget_used >= budget_max:
            break
    return finish("interrupted", last_seen,
                  f"等待下一题超时（{time.monotonic() - start:.1f}s），当前页面无法确认")


def _stop_non_question(cfg, report, current, exc, watch):
    """中途非题面只采纳最新本地文字，不从几何或模型异常猜终态。"""
    # 复用3s异常确认预算；同步OCR不能硬中断，返回后必须先拒绝超期结论。
    deadline = time.monotonic() + _ABNORMAL_WAIT_MAX
    try:
        current = grab_screen(cfg.region)
    except Exception as capture_error:  # noqa: BLE001
        safe_log(cfg.recorder, f"异常界面重抓失败：{capture_error}")
        _safe_shot(cfg, "stop_status", current)
        report.outcome = "interrupted"
        report.error = f"当前页面无法确认：{exc}"
        return
    _safe_shot(cfg, "transition_unknown", current)
    safe_log(cfg.recorder, f"进入异常界面确认：{exc}")
    verdict, fresh = _confirm_terminal(cfg, current, watch=watch)
    if time.monotonic() >= deadline:
        verdict = None
        exc = f"异常界面确认超时（{_ABNORMAL_WAIT_MAX:.0f}s预算）：{exc}"
    latest = fresh if fresh is not None else current
    _safe_shot(cfg, "stop_status", latest)
    report.outcome = ("wrong" if verdict.status == "revive" else verdict.status) if verdict else "interrupted"
    report.error = verdict.reason if verdict else f"当前页面无法确认：{exc}"
    safe_log(cfg.recorder, f"异常界面终止：{report.outcome}，{report.error}；"
             f"OCR {int(_terminal_ocr(cfg) is not None)}次")


def run_daily(cfg: DailyConfig) -> DailyReport:
    """跑完一局每日一答。"""
    report = DailyReport()
    watch = Stopwatch(cfg.recorder)
    if cfg.verbose:
        print(f"等待进入答题界面（最多 {cfg.max_questions} 题）…")
    context = DailyCaptureContext()
    current, question, q_start, wait_polls = _wait_first_question(
        cfg, report, watch, capture_context=context,
    )
    if current is None:
        return report
    wait_seconds = watch.elapsed()
    for question_no in range(1, cfg.max_questions + 1):
        abort.check()
        feedback = AnswerFeedback("daily")
        # 末题不走换题等待：答后颜色只亮一瞬，必须提前取帧才抓得到。
        # 判据与结算页的读取分支一致（都用 question_no == max_questions）。
        feedback.last_question = question_no == cfg.max_questions
        watch.reset(cfg.recorder)
        if question_no == 1:
            watch.mark(f"等第一题开始(启动→题面共{wait_seconds:.1f}s/{wait_polls}轮像素)")
        try:
            result = answer_one_question(cfg, current, question, question_no, watch,
                                         capture_context=context, feedback=feedback)
        except NotAQuestionScreen as exc:
            watch.report(f"第 {question_no} 题（异常界面）")
            _stop_non_question(cfg, report, current, exc, watch)
            return report
        except (AnswerUnavailable, WqError) as exc:
            watch.report(f"第 {question_no} 题（异常退出）")
            report.outcome = "interrupted"
            report.error = str(exc)
            if cfg.verbose:
                print(f"  停止：{exc}")
            return report
        # 界面自己写的题号（「第N题·共12题」里的 N）。读不出是 None
        # （"不知道"），**绝不猜** —— 猜错比读不出更糟。
        #
        ui_number = result.get("progress_no")
        if ui_number is not None:
            report.progress_numbers.append(ui_number)
            if ui_number != question_no:
                # 不一致**只告警不中断** —— 题号读不出或不一致都不该
                # 让整局失败（答完这局比对齐题号重要得多）。
                message = (
                    f"第 {question_no} 题（界面第 {ui_number} 题）"
                    f" —— 题号与程序计数不一致"
                )
                if cfg.verbose:
                    print(f"  ⚠ {message}")
                if cfg.recorder:
                    safe_log(cfg.recorder, message)
        report.questions.append(result)
        skipped = result.get("skipped", False)
        if not skipped and not cfg.dry_run:
            # 跳过的题**不计入 answered** —— 它根本没点。
            report.answered += 1
        elif skipped:
            report.skipped += 1
        if cfg.recorder:
            label = f"第 {question_no} 题"
            if ui_number is not None:
                label = f"第 {question_no} 题（界面第 {ui_number} 题）"
            if skipped:
                safe_log(
                    cfg.recorder,
                    f"{label} {result['stem']} → 跳过"
                    f"（{result.get('error') or '界面已变'}），未点击",
                )
            else:
                chosen = (
                    result["options"][result["answer"]]
                    if result["answer"] < len(result["options"])
                    else f"第 {result['answer'] + 1} 个按钮"
                )
                decision = ""
                if result.get("guessed"):
                    marker = "拟蒙答，未点击" if cfg.dry_run else "蒙答"
                    decision = f"（{marker}：{result.get('error') or '未取到有效答案'}）"
                safe_log(
                    cfg.recorder,
                    f"{label} {result['stem']} → {chosen}{decision}",
                )
        # 每题的 token 明细只落盘 —— 终端只留本局合计（与 PK 一致）。
        log_usage(f"第 {question_no} 题", usage_since(cfg.vision, q_start),
                  cfg.recorder)
        if cfg.dry_run:
            watch.report(f"第 {question_no} 题（dry-run）")
            if cfg.verbose:
                # 没点击就没有答后反馈 —— 标记写「未点击」，不假装答对了。
                print(format_question_line(result, ui_number=ui_number,
                                           question_no=question_no,
                                           mark="（未点击）"))
            report.outcome = "dry_run"
            return report
        q_start = usage_snapshot(cfg.vision)
        # 最后一题不再等下一题：点完直接读结算页
        if question_no == cfg.max_questions:
            if skipped:
                watch.report(f"第 {question_no} 题（跳过未点击）")
                try:
                    latest = grab_screen(cfg.region)
                except Exception as exc:  # noqa: BLE001 - 留证失败不能启动结算或覆盖中断
                    latest = None
                    safe_log(cfg.recorder, f"末题跳过后重抓失败：{exc}")
                _safe_shot(cfg, "stop_status", latest)
                if cfg.verbose:
                    print(format_question_line(result, ui_number=ui_number,
                                               question_no=question_no))
                report.outcome = "interrupted"
                report.error = "末题跳过未点击，不能确认最终结果"
                return report
            watch.report(f"第 {question_no} 题（点击完成）")
            report.outcome = _read_final(cfg, feedback=feedback)
            feedback.finish(result, recorder=cfg.recorder, verbose=False,
                            label=f"第 {ui_number or question_no} 题", terminal=True)
            if cfg.verbose:
                print(format_question_line(result, ui_number=ui_number,
                                           question_no=question_no))
            return report
        outcome = wait_for_next_question(cfg, current, watch, feedback=feedback)
        feedback.finish(result, recorder=cfg.recorder, verbose=False,
                        label=f"第 {ui_number or question_no} 题")
        if cfg.verbose:
            # 逐题只打一行：题号. 题干 → 答案 ✓。
            #
            # 这行必须等答后反馈定稿（finish 里）才能打 —— 标记要写对错，
            # 提前打就成了「先写答案、后补对错」的两行。
            print(format_question_line(result, ui_number=ui_number,
                                       question_no=question_no))
        if outcome.status == "next":
            current = outcome.image
        elif outcome.status == "revive":
            # 读到了明确的复活文字 → 答错，机会用完。
            #
            # 这是**唯一**能判 wrong 的路径之一：它有正面证据（页面
            # 上的「可以通过以下方式复活」），不是靠"画面没按钮"猜的。
            watch.report(f"第 {question_no} 题（点击完成）")
            report.outcome = "wrong"
            report.error = outcome.reason
            return report
        elif outcome.status == "success":
            watch.report(f"第 {question_no} 题（点击完成）")
            report.outcome = "success"
            report.error = outcome.reason
            return report
        elif outcome.status == "fail":
            watch.report(f"第 {question_no} 题（点击完成）")
            report.outcome = "fail"
            report.error = outcome.reason
            return report
        else:
            watch.report(f"第 {question_no} 题（点击完成）")
            report.outcome = "interrupted"
            report.error = outcome.reason
            if cfg.verbose:
                print(f"  停止：{outcome.reason}")
            return report
        context = DailyCaptureContext(captured_at=outcome.captured_at,
                                      confirmed_at=time.monotonic())
        question = None
        watch.report(f"第 {question_no} 题（含等待下一题）")
    # 正常路径走不到这里；走完 12 题在循环里已 return
    return report


def _read_final(cfg: DailyConfig, *, feedback: AnswerFeedback | None = None) -> str:
    """点完最后一题后读结算页，返回 "success" / "fail" / "final_pending"。"""
    if feedback is not None:
        # 结算页要等一会儿才出现；这段等待里连取几帧 ——
        # 答后颜色只亮一瞬，等 3 秒再去抓只剩渐隐的暗色。
        # 总等待时间不变（抓屏本身约 20 毫秒，落在等待里）。
        feedback.sample_window(lambda: grab_screen(cfg.region),
                               offsets=_FINAL_SAMPLE_OFFSETS, delay=_FINAL_SETTLE_SECONDS)
    else:
        time.sleep(_FINAL_SETTLE_SECONDS)
    image = grab_screen(cfg.region)
    if feedback is not None:
        feedback.observe(image)
    if cfg.recorder:
        _safe_shot(cfg, "final_result", image)
    try:
        verdict = read_daily_result(cfg.vision, image)
    except Exception as exc:  # noqa: BLE001 - 判读是尽力而为，失败只降级不崩
        verdict = "无"
        if cfg.verbose:
            print(f"  读结算页失败：{exc}")
    if cfg.recorder:
        safe_log(cfg.recorder, f"结算页判读：{verdict}")
    if verdict == "成功":
        return "success"
    if verdict == "失败":
        return "fail"
    return "final_pending"


def _wait_first_question(cfg: DailyConfig, report: DailyReport,
                         watch: Stopwatch | None = None, *,
                         capture_context: DailyCaptureContext | None = None):
    """统一等待入口，保留首题请求的完整用量窗口。"""
    context = capture_context if capture_context is not None else DailyCaptureContext()
    q_start = usage_snapshot(cfg.vision)
    image, question = wait_for_question(cfg, capture_context=context)
    if watch:
        watch.mark(f"等待首题({context.polls}轮)")
    if image is None:
        report.outcome = "error"
        report.error = "没等到答题界面。请确认小程序已点『开始答题』或类似按钮。"
        safe_log(cfg.recorder, report.error)
    return image, question, q_start, context.polls
