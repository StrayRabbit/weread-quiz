"""PK 答题循环：本地确认、识别、按钮定位、点击复核、换题等待和结算。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import abort
from .answer_feedback import SAMPLE_OFFSETS, AnswerFeedback
from .buttons import find_option_buttons, pick_button
from .capture import grab_screen
from .click import click_box
from .errors import (
    AnswerUnavailable,
    NotAQuestionScreen,
    OcrRejected,
    RecognitionTimeout,
    WqError,
)
from pyautogui import FailSafeException

from .screen import (
    has_answer_reveal, has_rendered_stem, is_new_question, is_question_screen,
    is_same_question, looks_like_question_template, progress_cache_mask,
    progress_cache_changed,
)
from .pk_budget import PKQuestionBudget, _IMAGE_PATH_MIN_BUDGET, _TEXT_PATH_TIMEOUT
from .pk_number import parse_progress_text, read_question_number
from .ocr_core import OcrRead
from .record import format_question_line
from .vision import VisionQuestion, VisionService, is_question, PROMPT_PK
from .usage import (
    PhaseTotals,
    Stopwatch,
    log_usage,
    safe_log,
    usage_since,
    usage_snapshot,
)

# 常规一局的题数：9 道数字题 + 1 道「最后一题·分数翻倍」。
_NORMAL_QUESTIONS = 10

# 轮次硬上限。
#
# 平局时界面会在常规 10 题之后**继续出题**（加时赛），所以这个上限只作
# 兜底，不能当成"答满 10 题就停" —— 真正的结束信号是界面出现结算页。
_MAX_QUESTIONS = 30

# 图片识别结果的图龄上限；本地读数仍须通过点击前同题校验。
_MAX_CAPTURE_AGE = 8.0

# 正常点击后开始检查下一题的固定延迟，可由 PK_NEXT_QUESTION_DELAY 覆盖。
_NEXT_QUESTION_DELAY = 1.0

# 换题等待期间追加取帧的时点（相对固定延迟开始，单位秒）：答后反馈只亮很短
# 一瞬，要赶在渐隐之前。
#
# 揭示色要等服务端回包才出现（实测 0.2~0.5 秒），末题页面却不到 1 秒就开始
# 渐隐转场 —— 只取一帧多半还是原色。0.1 秒一帧铺满固定延迟那一段，
# 总等待时间不变（抓屏本身约 20 毫秒，落在等待里）。
#
# 取帧时点与 daily 共用一份（answer_feedback.SAMPLE_OFFSETS）。
_FEEDBACK_SAMPLE_OFFSETS = SAMPLE_OFFSETS

# 上一题被跳过时多等一会儿。跳过意味着抓到的是半渲染帧，界面推进比
# 正常情况慢 —— 立刻再抓一帧很可能还是过渡态。
_SKIP_QUESTION_DELAY = 4.0

_EXTRA_WAIT_MAX = 3.0

# 揭示态等待预算，覆盖单题倒计时；非揭示转场另计短预算。
_REVEAL_WAIT_MAX = 13.0

_EXTRA_WAIT_STEP = 0.1

# 等待答题界面（第一题）的轮询间隔。这个**仍在用** —— 那时候还没有
# 当前题可比对，必须靠帧稳定判据确认界面渲染完成。
_POLL_INTERVAL = 0.10

_GATE_FALLBACK_SECONDS = 30.0
_GATE_FALLBACK_POLLS = max(1, round(_GATE_FALLBACK_SECONDS / _POLL_INTERVAL))

# 兼容路径和纯视觉等待回退的单次短超时。
_WAIT_MODEL_TIMEOUT = 5.0

_PROGRESS_OCR_SECONDS = 1.5

# 题号 OCR 开始时刻之间的最小重读间隔，避免轮询被同步 OCR 占满。
_PROGRESS_CACHE_SECONDS = 4.0

# 本地连续判否后允许一次视觉兜底；0 表示关闭。
_OCR_FALLBACK_SECONDS = 3.0

# 本地 OCR 的**文本路径**最多连续失败几次。
#
# 取 3：连续失败 3 次就进入冷却。
#
# 3 次都失败说明服务端**真的**在慢（不是偶发抖动）—— 再打几十次请求
# 也没用，还烧 token。而 3 次 × 5 秒 = 15 秒，已经超过单题题窗（9~12 秒），
# 所以这个上限也是防止一次等待把整道题吃掉的闸门。
#
_TEXT_PATH_MAX_FAILURES = 3

_TEXT_PATH_COOLDOWN_POLLS = 30


@dataclass
class PKCaptureContext:
    """逐题采集、确认与答案时刻；原始采集时间不随补读刷新。"""

    captured_at: float | None = None
    confirmed_at: float | None = field(default=None, kw_only=True)
    model_started_at: float | None = field(default=None, kw_only=True)
    answer_received_at: float | None = field(default=None, kw_only=True)
    question_no: int | None = None
    ocr_read: OcrRead | None = None
    # 等待阶段读到的题号行**原文**（读到了才有）。
    progress_text: str = ""
    # 同一份原文里解析出的总题数（读不到为 None）。
    progress_total: int | None = None
    # 结算页出现前**最后一张还带着答题卡片**的画面。
    #
    # 结算页上只有称号、星级和奖励积分，**没有比分** —— 最终比分只在这
    # 张画面上，胜负靠它判。
    final_frame: object | None = None

    def reset(self) -> None:
        self.captured_at = None
        self.confirmed_at = None
        self.model_started_at = None
        self.answer_received_at = None
        self.question_no = None
        self.ocr_read = None
        self.progress_text = ""
        self.progress_total = None
        self.final_frame = None


def _store_progress(context, text: str) -> None:
    if context is not None and text:
        number, total, raw = parse_progress_text(text)
        context.question_no = number
        context.progress_total = total
        context.progress_text = raw or ""


def _record_answer(context, question) -> None:
    if (context is None or context.answer_received_at is not None
            or question is None or not is_question(question)):
        return
    index = question.answer_index
    if (isinstance(index, int) and index >= 0
            and (question.bare_index or index < len(question.options))):
        context.answer_received_at = time.monotonic()


def _log_capture_latency(context, recorder, *, first_at=None, stage="答题") -> None:
    if context is None or recorder is None:
        return
    origin = context.captured_at if first_at is None else first_at
    if origin is None:
        return
    def elapsed(value, missing):
        return missing if value is None else f"{(value - origin) * 1000:.0f}ms"
    safe_log(recorder, f"{stage}延迟（首次采集到候选帧起）："
             f"确认答题页 {elapsed(context.confirmed_at, '未确认')}，"
             f"首调模型 {elapsed(context.model_started_at, '未调用')}，"
             f"取得答案 {elapsed(context.answer_received_at, '未取得')}")


@dataclass
class PKReport:
    """一局 PK 的结果。"""

    answered: int = 0
    questions: list[dict] = field(default_factory=list)
    outcome: str = "unknown"  # victory / defeat / questions_exhausted / interrupted / error
    error: str = ""
    # 结算页读到的最终信息。胜负面板要等模型调用，慢了也不影响答题。
    final: dict = field(default_factory=dict)
    # 最终比分 (我方, 对手)。**胜负由它判定** —— 奖励积分档位与输赢无关。
    score: tuple[int, int] | None = None

    @property
    def guessed(self) -> list[int]:
        """哪些题是蒙的（模型没给出答案）。1-based。"""
        return [i for i, q in enumerate(self.questions, 1)
                if not q.get("skipped") and q.get("guessed")]

    @property
    def solid(self) -> int:
        """真正由模型判定的题数。"""
        return self.answered - len(self.guessed)

    @property
    def ui_numbers(self) -> list[int | None]:
        """每一轮界面自己写的题号（读不出的为 None）。"""
        return [q.get("ui_number") for q in self.questions]

    def ui_total(self) -> int | None:
        """界面自己写的总题数（读不出的为 None）。"""
        totals = [q.get("ui_total") for q in self.questions if q.get("ui_total")]
        # 取最大值：平局加时赛可能把总题数改大，早先读到的那个偏小。
        return max(totals) if totals else None

    def missing_questions(self) -> list[int]:
        """按已知界面题号列出漏答；已答的无数字末题按末题原文处理，未知题号不猜测。"""
        numbers = [n for n in self.ui_numbers if n is not None]
        if not numbers:
            return []
        start, end = numbers[0], numbers[-1]
        total = self.ui_total()
        if total:
            end = max(end, total)
        if end < start:
            return []
        covered = set(numbers)
        # 「最后一题」没有数字题号，但它是答过的 —— 把它占的题号补进已答集合，
        # 否则平局加时赛时会被误报成"漏答第 10 题"。
        last_number = self._last_question_number(total)
        if last_number is not None:
            covered.add(last_number)
        return [n for n in range(start, end + 1) if n not in covered]

    def _last_question_number(self, total: int | None) -> int | None:
        """「最后一题」那一轮占的题号（它自己没有数字题号，按位置推）。

        平局加时赛时它不是最后一轮，但仍是常规一局的第 10 题：取它前面
        出现过的最大数字题号 +1；一道数字题都没读到时才退回总题数。
        """
        highest = None
        for question in self.questions:
            number = question.get("ui_number")
            if number is not None:
                highest = number if highest is None else max(highest, number)
                continue
            if question.get("skipped"):
                continue
            if "最后一题" not in (question.get("ui_number_text") or ""):
                continue
            if highest is not None:
                return highest + 1
            return total
        return None

    def skipped_first_questions(self) -> list[int]:
        """程序进入答题界面时就**已经错过**的题（1..start-1）。"""
        numbers = [n for n in self.ui_numbers if n is not None]
        if not numbers:
            return []
        start = min(numbers)
        if start <= 1:
            return []
        return list(range(1, start))

    def duplicate_questions(self) -> list[int]:
        """同一题号被处理了多次；题号重新计数（加时赛可能从头编号）不算重复。"""
        seen: set[int] = set()
        duplicates: set[int] = set()
        last: int | None = None
        for number in self.ui_numbers:
            if number is None:
                continue
            if last is not None and number < last:
                # 题号往回走了 —— 重新计数，另起一段，不算重复处理。
                seen = set()
            if number in seen:
                duplicates.add(number)
            seen.add(number)
            last = number
        return sorted(duplicates)

    def _score_text(self) -> str:
        """比分的简短写法；读不到就是空串。"""
        if not self.score:
            return ""
        mine, theirs = self.score
        if mine is None or theirs is None:
            return ""
        return f"，比分 {mine}:{theirs}"

    def summary(self) -> str:
        # 结算页给出的结论最权威，优先用它
        final = self.final or {}
        verdict = final.get("result")
        score_text = self._score_text()
        if self.questions:
            skipped = sum(bool(q.get("skipped")) for q in self.questions)
            progress = (f"处理 {len(self.questions)} 轮，实际点击 {self.answered} 题，"
                        f"跳过 {skipped} 轮")
            progress += score_text
            skipped_first = self.skipped_first_questions()
            if skipped_first:
                progress += f"，漏答第 {'、'.join(str(n) for n in skipped_first)} 题"
            missing = self.missing_questions()
            if missing:
                progress += f"，漏答第 {'、'.join(str(n) for n in missing)} 题"
            duplicates = self.duplicate_questions()
            if duplicates:
                progress += f"，第 {'、'.join(str(n) for n in duplicates)} 题重复处理"
            if verdict:
                if final.get("title"):
                    progress += f"，称号「{final['title']}」"
                return f"{verdict}，{progress}"
            if final.get("score"):
                if final.get("title"):
                    progress += f"，称号「{final['title']}」"
                return f"结算奖励 {final['score']}（未判定胜负），{progress}"
            if self.outcome == "interrupted":
                return f"中断在第 {len(self.questions) + 1} 轮：{self.error}（{progress}）"
            return f"结束（{self.outcome}），{progress}" + (f"：{self.error}" if self.error else "")
        if verdict:
            tail = f"，共答 {self.answered} 题" + score_text
            if final.get("title"):
                tail += f"，称号「{final['title']}」"
            return verdict + tail
        if final.get("score"):
            tail = f"，共答 {self.answered} 题"
            if final.get("title"):
                tail += f"，称号「{final['title']}」"
            return f"结算奖励 {final['score']}（未判定胜负）" + tail

        if self.outcome == "victory":
            return f"VICTORY，答了 {self.answered} 题"
        if self.outcome == "questions_exhausted":
            return f"答完 {self.answered} 题，未读到结算结果"
        if self.outcome == "interrupted":
            return f"中断在第 {self.answered + 1} 题：{self.error}"
        return f"结束（{self.outcome}），答了 {self.answered} 题：{self.error}"


def _is_last_question(progress_text: str) -> bool:
    """末题在题号行写的是「最后一题·分数翻倍」而不是数字。

    末题之后直接进结算页 —— 答后颜色只亮一瞬，固定延迟之后只剩渐隐的暗色。
    其他题沿用原有等待，不额外抓屏。
    """
    return "最后一题" in (progress_text or "")


def _read_progress_line(image, ocr) -> tuple[bool | None, str]:
    """读题号行：这一帧有没有「第 N 题·共 M 题」。读不出返回 (None, "")。"""
    if ocr is None:
        return None, ""
    reader = getattr(ocr, "read_progress_text", None)
    if reader is None:
        reader = getattr(ocr, "read_progress", None)
        if reader is None:
            return None, ""
        try:
            return reader(image), ""
        except Exception:  # noqa: BLE001 - 读不出当"不知道"
            return None, ""
    try:
        return reader(image)
    except Exception:  # noqa: BLE001 - 读不出当"不知道"
        return None, ""


def _has_progress_line(read) -> bool:
    """只根据题号证据判断界面类型，不将选项数和置信度当作界面判据。"""
    if read is None:
        return False
    return bool(getattr(read, "has_progress", False))


def _try_ocr_first(vision, image, *, timeout: float, totals=None,
                   local_only: bool = False, capture_context=None):
    """本地确认可只交接读数；旧调用方保留求答案入口。"""
    if getattr(vision, "local_ocr_available", None) is False:
        return None, None
    local = getattr(vision, "read_question_local", None)
    if local is not None:
        # 门控判否时它抛 OcrRejected —— 由调用方继续轮询。
        local_started = time.perf_counter()
        try:
            read = local(image)
        finally:
            # 本地 OCR 的耗时**即使抛异常也要记** —— 门控判否的那几轮
            # 同样花了钱（OCR 跑过了），不记就低估了本地开销。
            if totals is not None:
                totals.add("本地OCR", time.perf_counter() - local_started, unit="次")
        if local_only:
            return None, read
        answer = getattr(vision, "answer_from_ocr_read", None)
        if answer is None:
            # 没有"只问下标"的入口 —— 本地读数仍然有用（它证明这是
            # 答题页），但没有模型下标。返回 (None, read)。
            return None, read
        if capture_context is not None:
            capture_context.model_started_at = time.monotonic()
        model_started = time.perf_counter()
        try:
            question = answer(read, timeout=timeout)
        finally:
            if totals is not None:
                totals.add("模型请求", time.perf_counter() - model_started, unit="次")
        _record_answer(capture_context, question)
        return question, read
    try_read = getattr(vision, "try_read_question", None)
    if try_read is None:
        return None, None
    if capture_context is not None:
        capture_context.model_started_at = time.monotonic()
    started = time.perf_counter()
    try:
        question = try_read(image, timeout=timeout)
    finally:
        if totals is not None:
            totals.add("本地OCR+模型", time.perf_counter() - started, unit="次")
    _record_answer(capture_context, question)
    return question, None


def _read_visual(vision, image, *, timeout: float):
    """问视觉模型读这道题（**不做本地 OCR 预处理**）。"""
    visual = getattr(vision, "read_question_visual", None)
    if visual is None:
        return vision.read_question(image, prompt=PROMPT_PK, timeout=timeout)
    return visual(image, prompt=PROMPT_PK, timeout=timeout)


def _poll_sleep(totals) -> None:
    """轮询间隔的 sleep —— 归到单独一类。"""
    started = time.perf_counter()
    time.sleep(_POLL_INTERVAL)
    try:
        totals.add("轮询等待", time.perf_counter() - started, unit="次")
    except Exception:  # noqa: BLE001 - 计时失败不该影响轮询
        pass


def wait_for_question(region, vision, timeout: float = 120.0, verbose: bool = True,
                       watch: "Stopwatch | None" = None, *,
                       capture_context: PKCaptureContext | None = None,
                       ocr=None, recorder=None,
                       ocr_fallback_seconds: float | None = None):
    """优先本地题号确认首题并交接采集上下文；兼容纯视觉回退，超时返回空结果。"""
    if capture_context is not None:
        capture_context.reset()
    local_only = (capture_context is not None
                  and callable(getattr(vision, "read_question_local", None))
                  and getattr(vision, "local_ocr_available", None) is not False)
    wait_started = time.monotonic()
    deadline = time.time() + timeout
    polls = 0
    skipped = 0
    ocr_attempts = 0
    model_calls = 0
    fallback_calls = 0
    text_path_failures = 0
    # 冷却期已经跳过的轮数。到 _TEXT_PATH_COOLDOWN_POLLS 就重新允许
    # 试一次文本路径（见下面 skip_visual 的说明）。
    cooldown_polls = 0
    progress: bool | None = None
    progress_at: float = 0.0
    progress_baseline = None
    progress_size = None
    progress_text = ""
    gated_polls = 0
    screens: dict[str, int] = {}
    # 连续被本地 OCR 判否的轮数。达到 _OCR_FALLBACK_POLLS 就允许
    # 发起**一次**视觉模型兜底（见 _OCR_FALLBACK_SECONDS）。
    #
    # 按**连续**计，不按累计：读出来一次就归零 —— 这样"平时偶尔判否
    # 一下"不会攒够额度而在某个倒霉时刻突然阻塞 5 秒。
    ocr_rejects = 0
    # 视觉模型兜底**只允许一次** —— 一次就要阻塞 5 秒，而单题题窗只有
    # 9~12 秒，再来一次就把它吃光了。
    ocr_fallback_used = False
    # 兜底额度的轮数门槛。由 ocr_fallback_seconds 换算（配置项
    # PK_OCR_FALLBACK_SECONDS），None 时用模块常量。
    #
    # 轮询间隔为 0 时（测试里会这么 monkeypatch）不能做除法 —— 那时
    # 用"至少 1 轮"的门槛兜住。
    #
    _interval = _POLL_INTERVAL or 0.1
    _fallback_seconds = (_OCR_FALLBACK_SECONDS if ocr_fallback_seconds is None
                         else ocr_fallback_seconds)
    _ocr_fallback_enabled = _fallback_seconds > 0
    _ocr_fallback_polls = max(
        1,
        round(_fallback_seconds / _interval),
    )
    # 在任何 OCR 前记下首次采集到候选帧的时刻。
    first_visual_at: float | None = None
    totals = PhaseTotals()
    while time.time() < deadline:
        abort.check()
        captured_at = time.monotonic()
        gate_started = time.perf_counter()
        image = grab_screen(region)
        polls += 1
        template_ok = looks_like_question_template(image)
        rendered = has_rendered_stem(image)
        # 兜底：连续 _GATE_FALLBACK_POLLS 轮不通过就强制问一次模型。
        # 判据万一因小程序改版失效，最坏结果是偶尔问一次模型（浪费 token），
        # 而**不会**永远卡在等待里 —— 这是这道闸门唯一的真实风险。
        #
        # **两个条件是互补的**：
        #   template_ok  —— 结构上是不是答题卡片（匹配页/结算页判否）
        #   rendered     —— 题干正文画出来了没有（挡住半渲染帧）
        # 两条都过才"优先"调模型；不满足时兜底按时间强制问一次，
        # 所以判据失效只会多花 token，不会卡死。
        ready = (template_ok is not False and rendered)
        totals.add("抓屏+门控", time.perf_counter() - gate_started, unit="轮")
        if first_visual_at is None and rendered:
            first_visual_at = captured_at
        if template_ok is False:
            changed = (progress_baseline is None or progress_cache_changed(
                progress_baseline, progress_size, image))
            retry_unknown = (progress is None
                             and time.monotonic() - progress_at >= _PROGRESS_CACHE_SECONDS)
            if changed or retry_unknown:
                progress_baseline = progress_cache_mask(image)
                progress_size = image.size
                progress_started = time.perf_counter()
                # **记开始时刻**（见 progress_at 的说明）：重读间隔
                # 要从"上次 OCR 是什么时候开始的"算起，才是真的间隔。
                progress_at = time.monotonic()
                progress, progress_text = _read_progress_line(image, ocr)
                totals.add("题号行OCR", time.perf_counter() - progress_started,
                           unit="次")
            if progress is False:
                # 模板判否 **且** 题号行判否 —— 两条独立判据一致否认，
                # 这一帧确定不是答题页。跳过，不调模型、不消耗兜底额度。
                gated_polls += 1
                _poll_sleep(totals)
                continue
            if progress is True:
                # 题号行在 → 这**确实是**答题页，只是模板失效了（改版
                # 换了卡片样式）。这时必须调模型 —— 模板判否不能当安全
                # 闸门，否则改版一次程序就永远卡在等待里。
                ready = True
        else:
            # 模板放行 —— 题号行的缓存作废（界面已经变了）。
            #
            # **原文也要一起作废**：它是上一帧的题号，拿来报"程序从第
            # N 题才开始"会报出**上一帧**的题号 —— 那是个假消息。
            progress = None
            progress_text = ""
            progress_baseline = None
            progress_size = None
        if not ready and skipped < _GATE_FALLBACK_POLLS:
            skipped += 1
            _poll_sleep(totals)
            continue
        if time.monotonic() - captured_at >= _MAX_CAPTURE_AGE:
            continue
        skipped = 0
        ocr_attempts += 1
        question = None
        usage_before = usage_snapshot(vision)

        def _count_requests():
            """用服务用量差值累计真实 HTTP 请求，异常退出也计数。"""
            nonlocal model_calls
            after = usage_snapshot(vision)
            if usage_before is not None and after is not None:
                model_calls += after.requests - usage_before.requests

        # 这一轮文本路径是否**成功**（拿到题目）。
        #
        # 三种结局要分清：
        #   返回题目      → 文本路径成功，网络往返正常 → 失败计数归零
        #   抛 OcrRejected → **本地门控判否，一次网络都没发** → 不动计数
        #   抛 AnswerUnavailable → 网络往返失败 → 失败计数 +1
        text_path_succeeded = False
        # 这一轮**本地已读出**的题干+选项（门控通过，但没有模型下标）。
        #
        local_read = None
        confirmed_by_progress = progress is True and local_only
        # 这一轮要不要**完全跳过**模型调用（纯像素轮询）。
        #
        # 服务端出问题时，两条网络路径都会失败 —— 一次最多 5 秒。而等待
        # 阶段的轮询是 100ms 一轮：只要还在发请求，就是 N×5 秒的连续
        # 阻塞，期间程序看不见新界面。
        #
        # **连续失败到上限就纯像素轮询**（见 _TEXT_PATH_MAX_FAILURES）。
        # 3 次都失败说明服务端真的在慢，再打几十次请求也没用（还烧 token）。
        #
        skip_visual = False
        try:
            if confirmed_by_progress:
                pass  # 独立题号已确认，不再追加整图 OCR。
            elif text_path_failures >= _TEXT_PATH_MAX_FAILURES:
                question = None
                skip_visual = True
            else:
                question, local_read = _try_ocr_first(vision, image,
                                                      timeout=_WAIT_MODEL_TIMEOUT,
                                                      totals=totals,
                                                      local_only=local_only,
                                                      capture_context=capture_context)
                text_path_succeeded = question is not None
        except OcrRejected as exc:
            # OCR 判否 —— 这不是错误，是"这一帧还不行"。
            #
            screens["OCR未读出"] = screens.get("OCR未读出", 0) + 1
            ocr_rejects += 1
            if _has_progress_line(exc.read):
                progress_text = getattr(exc.read, "progress_text", "") or progress_text
                confirmed_by_progress = True
                local_read = None
                screens["答题页(题号行)"] = screens.get("答题页(题号行)", 0) + 1
                if recorder:
                    safe_log(
                        recorder,
                        f"等待：题号行读到 → 这是答题页；本地读数不全"
                        f"（{exc}），仅交接题号证据，答题阶段重读",
                    )
            elif (not local_only and _ocr_fallback_enabled
                    and not ocr_fallback_used
                    and ocr_rejects >= _ocr_fallback_polls
                    and template_ok is True):
                ocr_fallback_used = True
                fallback_calls += 1
                if recorder:
                    safe_log(
                        recorder,
                        f"等待：本地 OCR 连续 {ocr_rejects} 轮读不出，"
                        f"发起一次视觉模型兜底",
                    )
                # 落到下面的图片路径（vision.read_question），只走这一次。
            else:
                _count_requests()
                if recorder:
                    safe_log(recorder, f"等待：本地 OCR 未读出（{exc}），继续等")
                _poll_sleep(totals)
                continue
        except AnswerUnavailable as exc:
            screens["模型异常"] = screens.get("模型异常", 0) + 1
            ocr_rejects += 1
            text_path_failures += 1
            _count_requests()
            if recorder:
                safe_log(
                    recorder,
                    f"等待：OCR 文本路径失败（{exc}），已耗时一次网络往返；"
                    f"冷却文本路径",
                )
            _poll_sleep(totals)
            continue
        # 完整读数可复用；残缺读数仅交接题号证据。
        confirmed_locally = (
            question is None
            and local_read is not None
            and bool(local_read.options)
        )
        # 题号确认不等于选项门控通过。
        confirmed_by_progress = confirmed_by_progress and question is None
        if confirmed_locally:
            progress_text = getattr(local_read, "progress_text", "") or progress_text
            if recorder:
                safe_log(recorder, "等待：本地读题门控通过，交接答题阶段求答案")
        # 走到这里说明本地 OCR 没能给出题目 —— 可能是被判否后走了兜底，
        # 也可能是压根没有 OCR 快路径（纯像素模型）。两条路都用图片路径。
        if question is None and skip_visual:
            # 服务端在慢，模型路径正在冷却 —— 这一轮**纯像素轮询**，
            # 不发任何网络请求。见上面 skip_visual 的说明。
            #
            screens["服务端慢"] = screens.get("服务端慢", 0) + 1
            cooldown_polls += 1
            if cooldown_polls >= _TEXT_PATH_COOLDOWN_POLLS:
                # 冷却期满 —— 重新允许试一次文本路径。
                cooldown_polls = 0
                text_path_failures = 0
                if recorder:
                    safe_log(
                        recorder,
                        f"等待：服务端慢，冷却 {_TEXT_PATH_COOLDOWN_POLLS} 轮后"
                        f"重新尝试文本路径",
                    )
            _poll_sleep(totals)
            continue
        if local_only and question is None and not confirmed_locally and not confirmed_by_progress:
            _count_requests()
            _poll_sleep(totals)
            continue
        if question is None and not confirmed_locally and not confirmed_by_progress:
            try:
                if capture_context is not None and capture_context.model_started_at is None:
                    capture_context.model_started_at = time.monotonic()
                visual_started = time.perf_counter()
                try:
                    question = _read_visual(vision, image, timeout=_WAIT_MODEL_TIMEOUT)
                    _record_answer(capture_context, question)
                finally:
                    # 失败也要记 —— 那次网络往返同样花掉了时间。
                    totals.add("模型请求", time.perf_counter() - visual_started,
                               unit="次")
            except RecognitionTimeout:
                # 识别耗光了单题预算（pk_hybrid 的级联预算检查）。
                #
                # 所以这里**不等了**：记一次、往下走，让上层用最新的画面重新
                # 判断。这比继续轮询到超时强 —— 后者会一直卡到 120 秒。
                screens["识别超时"] = screens.get("识别超时", 0) + 1
                if watch:
                    watch.mark(f"等待:识别超时({watch.elapsed():.1f}s)")
                _count_requests()
                _poll_sleep(totals)
                continue
            except AnswerUnavailable:
                screens["模型异常"] = screens.get("模型异常", 0) + 1
                _count_requests()
                _poll_sleep(totals)
                continue
        # 统计这一轮**真实**发出了几次 HTTP 请求（usage 的 requests 差值）。
        #
        # OCR 快路径成功时一次网络都不发；文本路径与图片路径各发一次。
        # 纯像素测试桩没有 usage（usage_snapshot 返回 None），那就跳过
        # —— 计数保持 0 是**诚实的**：没有 usage 就无法知道有没有发过请求。
        _count_requests()
        # 本地 OCR 读出题目了 → 连续判否的计数**归零**。
        #
        # 兜底额度按**连续**判否计（见 ocr_rejects 的说明）：读出来一次
        # 就重新开始数，这样"平时偶尔判否一下"不会攒够额度、在某个倒霉
        # 时刻突然阻塞 5 秒。
        ocr_rejects = 0
        if text_path_succeeded:
            text_path_failures = 0
        # 本地证据独立确认界面；点击仍需答案及 fresh 身份校验。
        if confirmed_locally or confirmed_by_progress or is_question(question):
            confirmed_at = time.monotonic()
            if capture_context is not None:
                capture_context.captured_at = captured_at
                capture_context.confirmed_at = confirmed_at
                capture_context.ocr_read = local_read
                _store_progress(capture_context, progress_text)

            if watch:
                watch.mark(f"等待:命中答题页({watch.elapsed():.1f}s)")
                watch.report(
                    f"等待阶段（{model_calls} 次模型调用/{ocr_attempts} 次本地 OCR/"
                    f"{polls} 轮像素，其中 {gated_polls} 轮被本地判据拦下，"
                    f"{fallback_calls} 次视觉模型兜底）"
                )
            if recorder:
                detail = totals.format()
                if detail:
                    # watch 可能是 None（dry-run 那条路没传）—— 那就不
                    # 报"共多少"，只报归因明细。
                    waited = f"{(time.monotonic() - wait_started) * 1000:.0f}ms"
                    safe_log(
                        recorder,
                        f"等待阶段明细：{detail}"
                        f"（已归因 {totals.total() * 1000:.0f}ms/共 {waited}）"
                    )
            _log_capture_latency(capture_context, recorder,
                                 first_at=first_visual_at, stage="等待")
            if time.time() >= deadline:
                screens["总超时"] = screens.get("总超时", 0) + 1
                if recorder:
                    safe_log(recorder, "等待：模型调用后已过总超时，收工")
            return image, question
        screens[question.screen] = screens.get(question.screen, 0) + 1
        _poll_sleep(totals)

    if watch:
        watch.report(
            f"等待超时（{model_calls} 次模型调用/{ocr_attempts} 次本地 OCR/"
            f"{polls} 轮像素，其中 {gated_polls} 轮被本地判据拦下，"
            f"{fallback_calls} 次视觉模型兜底）"
        )
    # 超时那条路也要报归因 —— 超时往往正是"时间花在哪"最需要答案
    # 的时候（程序干等了两分钟却什么都没等到）。
    if recorder:
        detail = totals.format()
        if detail:
            waited = f"{(time.monotonic() - wait_started) * 1000:.0f}ms"
            safe_log(
                recorder,
                f"等待阶段明细：{detail}"
                f"（已归因 {totals.total() * 1000:.0f}ms/共 {waited}）"
            )
    if verbose:
        print(f"  等待 {timeout:.0f}s 仍未进入答题界面")
        if screens:
            detail = "，".join(f"{name} {count} 次" for name, count in sorted(screens.items()))
            print(f"  （期间模型判定：{detail}）")
        if polls and not model_calls:
            print(f"  ⚠ 像素门控拦下了全部 {polls} 轮，模型一次没被调用 —— "
                  f"判据可能已失效或界面改版了")
    return None, None


def _guess_after_timeout(region, scale, image, question, skip, guess_on_failure,
                         watch, error: str, *, ocr=None, progress_text="",
                         feedback: AnswerFeedback | None = None) -> dict:
    """识别失败后重抓并定位按钮，按当前题号和画面证据限制蒙答，未知页面不点击。"""
    if not guess_on_failure:
        return skip("capture_expired", error)
    try:
        fresh = grab_screen(region)
        boxes = find_option_buttons(fresh)
        box = pick_button(boxes, 0)
    except FailSafeException:
        raise
    except Exception as exc:  # noqa: BLE001 - 抓屏/定位失败就是蒙不了
        return skip("validation_failed", f"{error}；且蒙答失败：{exc}")
    if watch is not None:
        watch.mark("超时蒙题")
    # 仅同题校验通过时复用旧图题号；变化帧仍读取真正点击的 fresh 图。
    cached = parse_progress_text(progress_text)
    reuse_progress = False
    if cached[0] is not None or "最后一题" in (cached[2] or ""):
        try:
            reuse_progress = image is not None and is_same_question(image, fresh)
        except Exception:  # noqa: BLE001 - 不能确认同题就读取新图
            pass
    click_box(box, region, scale, pause=0.15)
    if feedback is not None:
        feedback.last_question = _is_last_question(progress_text)
        feedback.start(fresh, box)
        if feedback.last_question:
            feedback.observe_now(lambda: grab_screen(region))
    if watch is not None:
        watch.mark("超时蒙题点击")
    if reuse_progress:
        number, total, raw = cached
    else:
        number, total, raw = read_question_number(fresh, ocr=ocr)
    if watch is not None:
        watch.mark("题号复用" if reuse_progress else "题号OCR")
    return {
        "stem": getattr(question, "stem", ""),
        "options": getattr(question, "options", []),
        "answer": 0,
        "clicked_y": box.center_y,
        "guessed": True,
        "skipped": False,
        "bare_index": False,
        "error": error,
        # 实际点到的那道题的题号（读不出为 None）。
        "ui_number": number,
        "ui_total": total,
        "ui_number_text": raw.strip(),
    }


def _answer_from_pre_read(vision, ocr_read, *, deadline: float | None = None):
    """复用预读请求一次文本答案；单次最多 _TEXT_PATH_TIMEOUT 秒。

    剩下的预算留给图片路径 —— 文本路径一慢就整题蒙答太浪费（见
    pk_budget._TEXT_PATH_TIMEOUT）。
    """
    answer = getattr(vision, "answer_from_ocr_read", None)
    if answer is None:
        return None
    if deadline is None:
        return answer(ocr_read, timeout=_TEXT_PATH_TIMEOUT)
    now = time.monotonic()
    remaining = deadline - now
    if remaining <= 0:
        # 保留完整本地预读的旧时间戳豁免，点击前仍须通过同题校验。
        remaining = 3.0
        deadline = now + remaining
    question = answer(ocr_read, timeout=min(remaining, _TEXT_PATH_TIMEOUT),
                      deadline=deadline)
    if time.monotonic() >= deadline:
        raise RecognitionTimeout("deadline_exceeded")
    return question


class _ImagePathFailed(Exception):
    """图片路径彻底失败（模型没给出可用答案）。"""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _enough_for_image(budget: "PKQuestionBudget | None") -> bool:
    """还够不够跑一次图片路径。

    图片路径要编码整张图再让模型读（实测 1.5~3 秒）。剩下的预算不够
    就别试 —— 试了也只是再等一次超时，然后照旧蒙答。
    """
    if budget is None or budget.deadline is None:
        return False
    return budget.deadline - time.monotonic() >= _IMAGE_PATH_MIN_BUDGET


def _read_image_only(vision, image, deadline, capture_context):
    """只走图片路径：不重跑本地 OCR，也不重试文本路径。"""
    started = time.monotonic()
    try:
        visual = getattr(vision, "read_question_visual", None)
        if callable(visual):
            question = visual(image, prompt=PROMPT_PK, deadline=deadline)
        else:
            question = vision.read_question(image, prompt=PROMPT_PK, deadline=deadline)
    except RecognitionTimeout:
        raise
    except AnswerUnavailable as exc:
        raise _ImagePathFailed(exc) from exc
    finally:
        if capture_context is not None:
            _store_progress(capture_context, getattr(
                vision, "last_question_progress_text", ""))
            if capture_context.confirmed_at is None:
                capture_context.confirmed_at = getattr(
                    vision, "last_question_local_finished_at", None)
            if capture_context.model_started_at is None:
                capture_context.model_started_at = getattr(
                    vision, "last_question_model_started_at", started)
    return question


def _read_by_image(vision, image, deadline, capture_context, *, budget=None):
    """按截图预算读取答案；混合读取器可用本地倒计时更新共享预算。"""
    model_started = time.monotonic()
    try:
        timed_read = getattr(vision, "read_question_with_budget", None)
        if budget is not None and callable(timed_read):
            question = timed_read(image, prompt=PROMPT_PK, budget=budget)
        else:
            question = vision.read_question(image, prompt=PROMPT_PK, deadline=deadline)
    except RecognitionTimeout:
        raise
    except AnswerUnavailable as exc:
        raise _ImagePathFailed(exc) from exc
    finally:
        if capture_context is not None:
            _store_progress(capture_context, getattr(
                vision, "last_question_progress_text", ""))
            if capture_context.confirmed_at is None:
                capture_context.confirmed_at = getattr(
                    vision, "last_question_local_finished_at", None)
            if capture_context.model_started_at is None:
                capture_context.model_started_at = getattr(
                    vision, "last_question_model_started_at", model_started)
    return question


def answer_one_question(
    region,
    scale: float,
    vision: VisionService,
    image=None,
    question=None,
    guess_on_failure: bool = True,
    recorder=None,
    question_no: int = 0,
    verbose: bool = True,
    watch: "Stopwatch | None" = None,
    *,
    captured_at: float | None = None,
    ocr=None,
    ocr_read=None,
    capture_context: PKCaptureContext | None = None,
    feedback: AnswerFeedback | None = None,
) -> dict:
    """识别、复核并点击一道题；复用预读和采集时刻，失败按蒙答策略处理，未知或变化画面跳过。

    终端输出统一由 run_pk 的逐题行负责（那一行还要带上答后反馈），
    这里只把结果放进返回值 —— verbose 参数保留给调用方，不再单独打印。
    """
    own_watch = watch is None
    watch = watch or Stopwatch(recorder)
    start = usage_snapshot(vision)
    reused_usage = getattr(vision, "last_question_usage", None) if question is not None else None
    not_question = False
    progress_confirmed = _has_progress_line(ocr_read)
    if capture_context is not None:
        if captured_at is None:
            captured_at = capture_context.captured_at
        _store_progress(capture_context, getattr(ocr_read, "progress_text", ""))

    deadline = None
    budget = None

    confirmed_at = getattr(capture_context, "confirmed_at", None) if capture_context else None
    deadline_anchor = captured_at
    if confirmed_at is not None and confirmed_at > deadline_anchor:
        deadline_anchor = confirmed_at

    def expired():
        if budget is not None and budget.countdown_seconds is not None:
            return time.monotonic() >= budget.deadline
        if ocr_read is not None and question is None:
            return False
        if question is not None and getattr(question, "local_read", False):
            return False
        if deadline is not None and time.monotonic() >= deadline:
            return True
        return captured_at is not None and time.monotonic() - captured_at >= _MAX_CAPTURE_AGE

    def skip(reason, error=""):
        # 终端不再单独打「跳过」—— 逐题行（run_pk 打印）里会带上原因。
        return {
            "stem": getattr(question, "stem", ""),
            "options": getattr(question, "options", []),
            "answer": None,
            "guessed": False,
            "skipped": True,
            "reason": reason,
            "error": error,
        }

    # 题号行原文：既用来判末题，也用来在超时蒙答时复用题号。
    #
    # **每次现读，不取快照**：题号行常常是在模型调用期间才由本地 OCR 读到的
    # （_read_by_image 的 finally 会把它写回 capture_context）。开头取一次
    # 快照的话，模型超时后拿到的仍是空串 —— 末题判不出来，点击后就不会提前
    # 取帧，唯一一次取色落在固定延迟之后，那时末题页面已经在渐隐。
    def current_progress_text() -> str:
        return (getattr(capture_context, "progress_text", "")
                or getattr(ocr_read, "progress_text", ""))

    def guess(error):
        return _guess_after_timeout(
            region, scale, image, question, skip, guess_on_failure, watch,
            error, ocr=ocr, progress_text=current_progress_text(), feedback=feedback,
        )

    try:
        if image is None:
            captured_at = time.monotonic()
            image = grab_screen(region)
            if capture_context is not None:
                capture_context.captured_at = captured_at
            watch.mark("抓屏")
        if captured_at is None:
            return skip("capture_time_unknown")
        if (ocr_read is not None
                and (not ocr_read.options
                     or any(not text.strip() for text in ocr_read.options)
                     or len(ocr_read.options) != ocr_read.button_count)):
            safe_log(recorder, "残缺 OCR 预读仅保留题号证据，直接重读图片，不请求文本答案")
            ocr_read = None
        deadline = deadline_anchor + 6.0
        budget = PKQuestionBudget(captured_at, deadline)
        budget.observe(ocr_read)
        deadline = budget.deadline
        if budget.countdown_seconds is not None:
            safe_log(recorder, f"倒计时预算：截图剩余 {budget.countdown_seconds}s，"
                     f"当前识别余量 {max(0.0, deadline - time.monotonic()):.2f}s（已预留复核及点击时间）")
        if expired():
            return guess("采集时已过期（不调用模型）")
        preread_suspect = False
        if question is None:
            if ocr_read is not None:
                if capture_context is not None:
                    if _has_progress_line(ocr_read) and capture_context.confirmed_at is None:
                        capture_context.confirmed_at = time.monotonic()
                    if capture_context.model_started_at is None:
                        capture_context.model_started_at = time.monotonic()
                try:
                    question = _answer_from_pre_read(vision, ocr_read, deadline=deadline)
                except RecognitionTimeout:
                    return guess("deadline_exceeded")
                if question is None:
                    # 模型这次还是没给出下标。
                    #
                    # 这样"选项数与按钮数不一致"的检查也照样跑 ——
                    # 那条检查对预读读数同样必要（下标映射不可靠时点了
                    # 会点到别的选项上）。
                    question = VisionQuestion(
                        stem=ocr_read.stem,
                        options=list(ocr_read.options),
                        answer_index=None,
                        screen="question",
                        local_read=True,
                    )
                watch.mark("模型识别")
                preread_suspect = True
            else:
                preread_suspect = False
                try:
                    question = _read_by_image(vision, image, budget.deadline, capture_context, budget=budget)
                except RecognitionTimeout:
                    # 识别把单题预算耗光了（见 pk_hybrid 的级联预算检查）。
                    # 这时截图早超龄，再做任何识别都是白等 —— 直接蒙。
                    return guess("recognition_timeout")
                except _ImagePathFailed as exc:
                    return guess(f"识别失败：{exc}")
                watch.mark("模型识别")
        if expired():
            return guess("模型返回时已过期")
        _record_answer(capture_context, question)

        classification_error = ""
        if not is_question(question):
            progress_text = current_progress_text()
            progress_confirmed = progress_confirmed or bool(progress_text)
            if not progress_confirmed:
                progress_confirmed, progress_text = _read_progress_line(image, ocr)
                _store_progress(capture_context, progress_text)
            if not progress_confirmed:
                message = f"这不是答题界面（模型判定为 {question.screen}，本地无题号证据），不能点"
                safe_log(recorder, message)
                raise NotAQuestionScreen(message)
            classification_error = (
                f"模型判定为 {question.screen}，但本地题号行确认答题页；未取得合法答案"
            )
            safe_log(recorder, f"界面分类冲突：{classification_error}，按答案不可用处理")
            # 丢弃冲突模型的答案，仍须遵守蒙答开关及下方同题、按钮校验。
            question = VisionQuestion(stem="", options=[], screen="question")
            preread_suspect = False

        boxes = find_option_buttons(image)
        watch.mark("定位按钮")
        guessed = False
        error = ""
        try:
            answer = question.answer_index
            if answer is None:
                raise AnswerUnavailable(classification_error or "模型没给出合法答案下标")
            if not (0 <= answer < len(boxes)):
                raise AnswerUnavailable(
                    f"模型答案下标 {answer} 越界（共 {len(boxes)} 个按钮）"
                )
            if not question.bare_index and len(question.options) != len(boxes):
                source = "本地 OCR" if question.local_read else "图片模型"
                raise AnswerUnavailable(
                    f"{source}读到 {len(question.options)} 个选项，检测到 {len(boxes)} 个按钮，不一致"
                )
        except AnswerUnavailable as exc:
            safe_log(recorder, f"答案校验失败：{exc}")
            answer = None
            if preread_suspect and _enough_for_image(budget):
                # 只走图片路径：**不重跑本地 OCR，也不重试文本路径**。
                #
                # 文本路径刚刚失败过（多半是慢），再来一遍只会把剩余
                # 预算也吃掉 —— 到头来还是蒙答，而蒙答只有一半机会。
                try:
                    question = _read_image_only(vision, image, budget.deadline,
                                               capture_context)
                except RecognitionTimeout:
                    pass  # 预算耗光 —— 落到下面的蒙题分支
                except _ImagePathFailed:
                    pass  # 图片路径也失败 —— 落到下面的蒙题分支
                else:
                    if is_question(question):
                        answer = question.answer_index
                        if (answer is not None and 0 <= answer < len(boxes)
                                and (question.bare_index
                                     or len(question.options) == len(boxes))):
                            # 图片路径给出了能用的答案 —— 用它。
                            error = ""
                            watch.mark("模型识别(图片回退)")
                            _record_answer(capture_context, question)
                            preread_suspect = False
                    if preread_suspect:
                        answer = None
            if answer is None or preread_suspect:
                if not guess_on_failure:
                    raise
                # 仅候选答案校验失败可蒙答，不扩大到网络、年龄或身份失败。
                answer = 0
                guessed = True
                error = str(exc)

        # 正常、裸序号、首题复用及蒙答共用此门。fresh 仅在内存中使用，
        # 留证和下一题等待的基准始终保留模型输入原图。
        if expired():
            return guess("点击前已过期")
        try:
            fresh = grab_screen(region)
            same = is_same_question(image, fresh)
            if not same:
                return skip("screen_changed", error)
            fresh_boxes = find_option_buttons(fresh)
            box = pick_button(fresh_boxes, answer)
            watch.mark("点击前校验")
        except FailSafeException:
            raise
        except Exception as exc:
            return skip("validation_failed", str(exc))

        if expired():
            if not guess_on_failure:
                return skip("capture_expired", error)
            try:
                box = pick_button(fresh_boxes, 0)
            except WqError:
                return skip("capture_expired", error)
            guessed = True
            answer = 0
            error = error or "识别超时，已蒙第一个按钮"
            watch.mark("超时蒙题")
        # 不在校验异常捕获范围内；点击失败和紧急停止必须交给调用方。
        click_box(box, region, scale, pause=0.15)
        if feedback is not None:
            feedback.last_question = _is_last_question(current_progress_text())
            feedback.start(fresh, box)
            if feedback.last_question:
                # 末题之后直接进结算页，颜色只亮一瞬：立刻取一帧。
                feedback.observe_now(lambda: grab_screen(region))
        watch.mark("蒙题点击" if guessed else "点击")

        # 终端不再在这里逐题打印 —— 统一由 run_pk 打「题号. 题干 → 答案」一行，
        # 那行还要带上答后反馈（在换题等待之后才知道）。
        return {
            "stem": question.stem,
            "options": question.options,
            "answer": answer,
            "clicked_y": box.center_y,
            "guessed": guessed,
            "skipped": False,
            "bare_index": question.bare_index,
            "error": error,
        }
    except NotAQuestionScreen:
        not_question = True
        raise
    finally:
        _log_capture_latency(capture_context, recorder)
        if not not_question:
            # 只保存模型输入原图；正常路径在点击后落盘，skip 也保留已知图。
            if recorder and image is not None:
                try:
                    recorder.shot(f"q{question_no:02d}_before", image=image)
                except FailSafeException:
                    raise
                except Exception as exc:
                    safe_log(recorder, f"第 {question_no} 题留证失败（{type(exc).__name__}）")
                else:
                    watch.mark("截图落盘")
            if own_watch:
                watch.report(f"第 {question_no} 题")
            # 每题的 token 明细只写日志 —— 终端留本局合计就够了。
            log_usage(f"第 {question_no} 题",
                      reused_usage if reused_usage is not None else usage_since(vision, start),
                      recorder)


def wait_for_next_question(region, current_image, watch: "Stopwatch | None" = None, *,
                           capture_context: PKCaptureContext | None = None,
                           delay: float | None = None,
                           feedback: AnswerFeedback | None = None):
    """固定延迟后补等非揭示态的新题；揭示态使用独立长预算，返回 (截图, 可选答案)。"""
    if capture_context is not None:
        capture_context.reset()
    wait_seconds = _NEXT_QUESTION_DELAY if delay is None else delay
    if feedback is not None and feedback.last_question:
        # 只有末题需要提前取帧：固定延迟中间连取几帧，晚了只剩渐隐的暗色。
        # 这段时间本来就在等，不额外拉长总等待。
        feedback.sample_window(lambda: grab_screen(region), delay=wait_seconds)
    else:
        time.sleep(wait_seconds)
    captured_at = time.monotonic()
    candidate = grab_screen(region)
    if feedback is not None:
        feedback.observe(candidate)

    revealed = has_answer_reveal(candidate)
    phase_waited = 0.0
    total_waited = 0.0
    if revealed and watch:
        # 首帧就是揭示态 —— 这时没有"进入"的转变可记，但必须留痕，
        # 否则日志里看不出这一轮是在揭示态里度过的。
        watch.mark("等下一题:处于揭示态")
    # 记下**最后一张还带着答题卡片**的画面：结算页上没有比分，
    # 最终比分只在这张画面上（对局结束那一轮最要紧）。
    last_frame = candidate if (revealed or is_question_screen(candidate)) else None
    while not _is_next_question_ready(candidate, current_image):
        abort.check()
        budget = _REVEAL_WAIT_MAX if revealed else _EXTRA_WAIT_MAX
        if phase_waited >= budget:
            break
        time.sleep(_EXTRA_WAIT_STEP)
        phase_waited += _EXTRA_WAIT_STEP
        total_waited += _EXTRA_WAIT_STEP
        captured_at = time.monotonic()
        candidate = grab_screen(region)
        if feedback is not None:
            feedback.observe(candidate)
        now_revealed = has_answer_reveal(candidate)
        # 揭示态和答题态都还有卡片在 —— 只有结算页没有。
        if now_revealed or is_question_screen(candidate):
            last_frame = candidate
        # 揭示态的出现和消失都换一次预算：出现是因为要等倒计时走完
        # （长预算），消失是因为回到普通渲染节奏（短预算）。
        if now_revealed != revealed:
            revealed = not revealed
            phase_waited = 0.0
            if watch:
                watch.mark(f"等下一题:{'进入' if revealed else '离开'}揭示态")

    if watch:
        watch.mark(f"等下一题(固定{wait_seconds:.2f}s"
                   + (f" + 补等{total_waited:.1f}s)" if total_waited else ")"))
    if not is_question_screen(candidate):
        # 补等到底还是**没有按钮** → 对局真的结束了（结算页）。
        if watch:
            watch.mark("等下一题:无按钮")
        if capture_context is not None:
            capture_context.final_frame = last_frame
        return None, None
    if not _is_next_question_ready(candidate, current_image):
        if watch:
            watch.mark("等下一题:题干未变(按重复题处理)")
    if capture_context is not None:
        capture_context.captured_at = captured_at
    return candidate, None


def _is_next_question_ready(candidate, current_image) -> bool:
    """依次排除揭示态、检查题干已换、题干已渲染及按钮可用。"""
    if has_answer_reveal(candidate):
        return False
    if current_image is not None and not is_new_question(current_image, candidate):
        return False
    return is_question_screen(candidate)


def run_pk(
    region,
    scale: float,
    vision: VisionService,
    max_questions: int = _MAX_QUESTIONS,
    guess_on_failure: bool = True,
    recorder=None,
    verbose: bool = True,
    next_question_delay: float | None = None,
    ocr=None,
    ocr_fallback_seconds: float | None = None,
) -> PKReport:
    """跑完一局 PK。"""
    report = PKReport()
    watch = Stopwatch(recorder)

    if verbose:
        print(f"等待进入答题界面（常规一局 {_NORMAL_QUESTIONS} 题，"
              f"平局加时赛会继续出题）…")

    capture_context = PKCaptureContext()
    # 平局加时赛的提示只打一次。
    tiebreak_notified = False
    current, question = wait_for_question(region, vision, timeout=120.0, verbose=verbose,
                                          watch=watch, capture_context=capture_context,
                                          ocr=ocr, recorder=recorder,
                                          ocr_fallback_seconds=ocr_fallback_seconds)
    if current is None:
        report.outcome = "error"
        report.error = "没等到答题界面。请确认小程序已点『自动匹配』或『邀请好友」。"
        if recorder:
            safe_log(recorder, "没等到答题界面")
        return report

    for question_no in range(1, max_questions + 1):
        abort.check()
        feedback = AnswerFeedback("pk")
        watch.reset(recorder)
        if question_no == 1:
            # 等待阶段那次识别的耗时要算进第 1 题，所以 reset 之后再补一刀。
            watch.mark("首题识别(复用等待阶段)")
        # 等待阶段带出来的本地读数**只给第 1 题用**。
        #
        pre_read = capture_context.ocr_read
        capture_context.ocr_read = None
        try:
            # 复用等待阶段确认好的识题图，每题仅保存一张；换题轮询图不落盘。
            result = answer_one_question(
                region,
                scale,
                vision,
                image=current,
                question=question,
                guess_on_failure=guess_on_failure,
                recorder=recorder,
                question_no=question_no,
                verbose=verbose,
                watch=watch,
                captured_at=capture_context.captured_at,
                ocr=ocr,
                ocr_read=pre_read,
                capture_context=capture_context,
                feedback=feedback,
            )
        except NotAQuestionScreen as exc:
            # 界面不是答题页（匹配等待 / 结算）—— 不是错误，是正常结束
            report.outcome = "questions_exhausted"
            if verbose:
                print("  " + str(exc))
            _read_final(region, report, vision, recorder, verbose,
                        capture_context=capture_context)
            return report
        except (AnswerUnavailable, WqError) as exc:
            report.outcome = "interrupted"
            report.error = str(exc)
            if verbose:
                print(f"  停止：{exc}")
            _read_final(region, report, vision, recorder, verbose,
                        capture_context=capture_context)
            return report

        report.questions.append(result)
        skipped = result.get("skipped", False)
        if not skipped:
            report.answered += 1
        if result.get("guessed") and result.get("ui_number_text") is not None:
            ui_number = result.get("ui_number")
            ui_total = result.get("ui_total")
            number_text = result.get("ui_number_text") or ""
        elif capture_context.progress_text:
            # 复用等待阶段的原文 —— 不再 OCR 一遍。
            ui_number, ui_total, number_text = parse_progress_text(
                capture_context.progress_text)
            # 总题数：等待阶段解析出来的优先（同一份原文，值相同）。
            if ui_total is None:
                ui_total = capture_context.progress_total
            result["ui_number"] = ui_number
            result["ui_total"] = ui_total
            result["ui_number_text"] = number_text.strip()
        else:
            ui_number, ui_total, number_text = read_question_number(current, ocr=ocr)
            result["ui_number"] = ui_number
            result["ui_total"] = ui_total
            result["ui_number_text"] = number_text.strip()
        if recorder:
            label = f"第 {ui_number} 题" if ui_number else f"第 {question_no} 轮"
            if skipped:
                safe_log(recorder, f"{label} {result['stem']} → "
                         f"跳过（{result['reason']}），未点击")
            else:
                chosen = (result['options'][result['answer']]
                          if result['answer'] < len(result['options']) else '?')
                safe_log(recorder, f"{label} {result['stem']} → {chosen}"
                         f"{'（蒙的）' if result.get('guessed') else ''}，已点击")

        # 跳过说明界面没正常推进（多半是半渲染帧被拦下），多等一会儿
        # 再看，比立刻抓一帧更可能撞上真正的下一题。
        delay = next_question_delay
        if delay is None:
            delay = _NEXT_QUESTION_DELAY
        if skipped:
            delay = max(delay, _SKIP_QUESTION_DELAY)
        current, question = wait_for_next_question(
            region, current, watch=watch, capture_context=capture_context, delay=delay,
            feedback=feedback,
        )
        feedback.finish(result, recorder=recorder, verbose=False,
                        label=f"第 {ui_number or question_no} 题", terminal=current is None)
        if verbose:
            # 逐题只打一行：题号. 题干 → 答案 ✓。
            #
            # 这行必须等答后反馈定稿（finish 里）才能打 —— 标记要写对错，
            # 提前打就成了「先写答案、后补对错」的两行。
            print(format_question_line(result, ui_number=ui_number,
                                       question_no=question_no))
        # 一行涵盖"点击本题 → 等到下一题就绪"的整段，这才是能和倒计时
        # 9~12s 对账的量。分段明细（等下一题多少帧、模型多少 ms）都在里面。
        watch.report(f"第 {question_no} 题（含等待下一题）")
        if current is None:
            # 没等到下一题 —— 对局结束，去读结算页
            report.outcome = "questions_exhausted"
            if not skipped and len(report.questions) < 3:
                report.outcome = "interrupted"
                report.error = (
                    f"处理完第 {len(report.questions)} 轮后 {delay + _EXTRA_WAIT_MAX:.1f} 秒内"
                    "没有出现下一题，对局可能已中断（网络掉线、对手退出等）"
                )
            _read_final(region, report, vision, recorder, verbose,
                        capture_context=capture_context)
            return report

        if not tiebreak_notified and "最后一题" in (result.get("ui_number_text") or ""):
            # 「最后一题」之后界面还在出题 —— 平局加时赛，接着答。
            tiebreak_notified = True
            if verbose:
                print("  「最后一题」之后界面仍出题 —— 平局加时赛，继续")
            if recorder:
                safe_log(recorder, f"第 {question_no} 轮（最后一题）之后界面仍出题，"
                                   "判定为平局加时赛，继续答题")

    if report.questions and current is not None:
        # 轮次用尽时界面还是答题页 —— 对局没结束，不能当成"答完了"。
        report.error = (f"已达轮次上限 {max_questions} 轮，界面仍是答题页"
                        "（对局未结束）")
    report.outcome = "questions_exhausted"
    _read_final(region, report, vision, recorder, verbose,
                capture_context=capture_context)
    return report


def _read_final(region, report: PKReport, vision, recorder, verbose: bool, *,
                capture_context=None) -> None:
    """对局结束后读奖励积分和最终比分，把结果写进 report。

    胜负由**比分**判定 —— 结算页上的奖励积分档位与输赢无关（实测 +1 的
    那一局是 125:60 的大胜）。比分只能从结算前最后一张答题卡片上读。
    """
    from .result import read_result

    time.sleep(1.5)  # 等结算页动画播完
    image = grab_screen(region)
    if recorder:
        recorder.shot("final_result", image=image)

    ocr = getattr(vision, "_ocr", None)
    report.score = _read_final_score(capture_context, ocr)
    if recorder:
        safe_log(recorder, f"最终比分：{report.score[0]}:{report.score[1]}"
                 if report.score else "最终比分：未读到")
    report.final = read_result(image, vision=vision,
                               ocr=ocr, recorder=recorder,
                               score=report.score)
    verdict = report.final.get("result", "")
    if verdict in ("胜利", "落败"):
        report.outcome = "victory" if verdict == "胜利" else "defeat"
    if recorder:
        safe_log(recorder, f"结算：{verdict or '未识别'}，奖励积分：{report.final.get('score') or '未读到'}")
    if verbose:
        print(f"\n结算页：{verdict or '未识别（未读到明确胜负或奖励积分）'}")
        if report.final.get("title"):
            print(f"  称号：{report.final['title']}")
        if report.final.get("score"):
            print(f"  {report.final['score']}")
        if report.score:
            print(f"  比分：{report.score[0]}:{report.score[1]}")


def _read_final_score(capture_context, ocr):
    """读最终比分 (我方, 对手)；读不到返回 None。"""
    if capture_context is None or ocr is None:
        return None
    frame = getattr(capture_context, "final_frame", None)
    if frame is None:
        return None
    try:
        from .pk_score import read_score

        mine, theirs = read_score(frame, ocr)
    except Exception:  # noqa: BLE001 - 读不到比分不影响其它结算信息
        return None
    if mine is None or theirs is None:
        return None
    return (mine, theirs)
