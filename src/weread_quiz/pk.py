"""每日一答 PK 的答题循环。
功能：识别当前题目 → 定位正确选项 → 点下去 → 等下一题。
PK 是双人实时对战：选项 2~3 个、无字母标记、答错不停、倒计时 9~12 秒。
OCR 读不出选项（深蓝底 + 发光艺术字），走视觉模型或混合路径。
本模块不判对错。
识别的两条路径与置信度门控见 pk_ocr.py / pk_hybrid.py。
换题判定见 screen.py：靠题干区二值化帧差。
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
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
from .pk_number import parse_progress_text, read_question_number
from .ocr_core import OcrRead
from .vision import VisionQuestion, VisionService, is_question, PROMPT_PK
from .usage import (
    PhaseTotals,
    Stopwatch,
    emit_usage,
    safe_log,
    usage_since,
    usage_snapshot,
)
# 一局固定的题数。
_MAX_QUESTIONS = 10
# 采集图的有效期（秒），超过则认为画面可能已变化。
_MAX_CAPTURE_AGE = 8.0
# 点击后等下一题出现的固定延迟（秒）。
# 配合下方补等：「至少等非这么久，且等到有按钮为止」。
# 默认 1.0 秒：短但安全。
_NEXT_QUESTION_DELAY = 1.0
# 上一题跳过（半渲染帧）时额外多等。
_SKIP_QUESTION_DELAY = 4.0
# 固定延迟到期后，没有选项按钮时最多再补等。
# 覆盖"按钮回来"和"题干画完"两阶段。
_EXTRA_WAIT_MAX = 3.0
# 揭示态（答案已给出、等倒计时结束）的补等上限。
# 揭示态不是渲染而是倒计时，需要更长等待。
_REVEAL_WAIT_MAX = 13.0
# 补等步长。0.1 秒比 0.05 秒更划算：多等 25ms 但抓屏次数减半。
_EXTRA_WAIT_STEP = 0.1
# 等待答题界面（第一题）的轮询间隔。
_POLL_INTERVAL = 0.10
# 像素门控连续拒绝多少轮后，强制问一次模型兜底。
# 兜底防止判据因改版失效时程序永远卡在等待里。
_GATE_FALLBACK_SECONDS = 30.0
_GATE_FALLBACK_POLLS = max(1, round(_GATE_FALLBACK_SECONDS / _POLL_INTERVAL))
# 纯视觉路径等待超时（秒）；正常答题保持原预算。
_WAIT_MODEL_TIMEOUT = 5.0
# 题号行缓存失效间隔（秒）；画面不变时不重复 OCR。
_PROGRESS_CACHE_SECONDS = 1.0
# 本地 OCR 连续判否多少秒后，允许发起一次视觉模型兜底。
# 兜底每次最多阻塞 8 秒，总耗时仍应在 PK 单题倒计时内。
_OCR_FALLBACK_SECONDS = 3.0
# 本地 OCR 文本路径最多连续失败次数。超过即进入冷却。
_TEXT_PATH_MAX_FAILURES = 3
# 冷却期跳过多少轮轮询后重新允许试文本路径。
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
    def reset(self) -> None:
        self.captured_at = None
        self.confirmed_at = None
        self.model_started_at = None
        self.answer_received_at = None
        self.question_no = None
        self.ocr_read = None
        self.progress_text = ""
        self.progress_total = None
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
        """界面自己写的总题数（读不出的为 None）。
        题号行是「第 N 题·共 M 题」，M 就是这一局的题数。它比"最后
        一个读到的题号"更可信 —— 程序可能在最后一题之前就退出了，
        那时最后一个题号并不是总题数。
        """
        for q in self.questions:
            total = q.get("ui_total")
            if total:
                return total
        return None
    def missing_questions(self) -> list[int]:
        """按题号推算漏了哪几题。
        读得到总题数时用其上界；读不到时退回"最后一个题号"。
        末轮已答题但题号读不出（如「最后一题·分数翻倍」）时不计漏答。
        """
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
        if self._last_round_is_last_question():
            end -= 1
        return [n for n in range(start, end + 1) if n not in covered]
    def _last_round_is_last_question(self) -> bool:
        """最后一轮是否是「最后一题·分数翻倍」（界面无数字）。"""
        if not self.questions:
            return False
        last = self.questions[-1]
        if last.get("ui_number") is not None or last.get("skipped"):
            return False
        text = last.get("ui_number_text") or ""
        return "最后一题" in text
    def skipped_first_questions(self) -> list[int]:
        """程序进入答题界面时就**已经错过**的题（1..start-1）。
        和 `missing_questions` 互补：那个算的是"处理过的题号之间的
        缺口"，这个算的是"还没开始处理就已经漏掉的"。
        """
        numbers = [n for n in self.ui_numbers if n is not None]
        if not numbers:
            return []
        start = min(numbers)
        if start <= 1:
            return []
        return list(range(1, start))
    def duplicate_questions(self) -> list[int]:
        """同一题号被处理了多次。"""
        seen: dict[int, int] = {}
        for number in self.ui_numbers:
            if number is None:
                continue
            seen[number] = seen.get(number, 0) + 1
        return sorted(n for n, count in seen.items() if count > 1)
    def summary(self) -> str:
        # 结算页给出的结论最权威，优先用它
        final = self.final or {}
        verdict = final.get("result")
        if self.questions:
            skipped = sum(bool(q.get("skipped")) for q in self.questions)
            progress = (f"处理 {len(self.questions)} 轮，实际点击 {self.answered} 题，"
                        f"跳过 {skipped} 轮")
            # 题号能读出来时，把漏题/重读摆在明面上 —— 轮数相同不
            # 代表题数相同（可能重读了上一题）。
            #
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
            # 读到了奖励积分但判不出胜负（+3/+4 等没确证过的档位）。
            # 不要丢掉积分信息 —— 旧实现只认 +5/+1，+3 整段全丢。
            if final.get("score"):
                if final.get("title"):
                    progress += f"，称号「{final['title']}」"
                return f"结算奖励 {final['score']}（未判定胜负），{progress}"
            if self.outcome == "interrupted":
                return f"中断在第 {len(self.questions) + 1} 轮：{self.error}（{progress}）"
            return f"结束（{self.outcome}），{progress}" + (f"：{self.error}" if self.error else "")
        if verdict:
            tail = f"，共答 {self.answered} 题"
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
def _read_progress_line(image, ocr) -> tuple[bool | None, str]:
    """读题号行：这一帧有没有「第 N 题·共 M 题」。读不出返回 (None, "")。
    模板预筛认卡片**样式**（框线转角），题号行认**语义**（功能性文字）。
    改版会改样式、不改语义 —— 两者都判否时这一帧确定不是答题页。
    没有 OCR 引擎时返回 (None, "")，由兜底处理。
    
    题号行原文顺带带回：确认答题页之后要报"程序从第 N 题才开始"，
    来自同一条带，复用原文省一次同区域 OCR。
    """
    if ocr is None:
        return None, ""
    reader = getattr(ocr, "read_progress_text", None)
    if reader is None:
        # 没有带原文的入口（测试桩、旧引擎）—— 退回只读判据。
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
    """这份读数里有没有题号行 —— 也就是"这是不是答题页"。
    **只用题号行判断**（2026-10-30 19:58 那局的修）。
    题号行（「第 N 题·共 M 题」/「最后一题」）是**界面类型**的判据：
    答题页有它，匹配等待页和结算页都没有。它与选项数、置信度**无关**
    —— 这正是它能独立作证的原因。
    为什么这条判据能独立成立（实测 150 张真实 PK 截图）：48 张答题帧
    全部读到题号行，2 张匹配页一张都没读到 —— 零假阴性。
    **为什么必须与"选项数"分开**（2026-10-30 19:58 那局）：
    门控原来把两件事绑在一起 —— 选项数与按钮数对不上时整个读数被丢掉，
    "这是不是答题页"的结论也跟着丢了。那局的实测：10 轮「选项数不符
    （OCR 1 vs 按钮 2）」，期间一次模型都没调，第 1 题整题错过 ——
    而那 10 帧的题号行**全都读得到**。
    read 为 None 时返回 False（"不知道"，不是"不是答题页"）。
    """
    if read is None:
        return False
    return bool(getattr(read, "has_progress", False))
def _try_ocr_first(vision, image, *, timeout: float, model_timeout_seconds: float | None = None, totals=None,
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
        # 纯像素模型（测试桩、或没接混合读取器）—— 走原来的路径。
        return None, None
    if capture_context is not None:
        capture_context.model_started_at = time.monotonic()
    started = time.perf_counter()
    try:
        question = try_read(image, timeout=timeout)
    finally:
        # 旧入口把本地 OCR 与模型调用捆在一起 —— 只能合成一类。
        if totals is not None:
            totals.add("本地OCR+模型", time.perf_counter() - started, unit="次")
    _record_answer(capture_context, question)
    return question, None
def _read_visual(vision, image, *, timeout: float, model_timeout_seconds: float | None = None):
    """问视觉模型读这道题（**不做本地 OCR 预处理**）。
    等待阶段用（2026-10-30）。
    **为什么绕开本地 OCR**：走到这一步时，本地 OCR **已经跑过了** ——
    整图路（~800ms）+ 拼接路（~775ms）+ 合并带（1.0~2.3s）合计约
    3.9 秒。若再用 `vision.read_question`（混合读取器），那些 OCR 会
    **全部重做一遍**，然后才发起那次最多 8 秒的阻塞调用 —— 合计最坏
    约 15.75 秒 > PK 单题倒计时 12 秒。
    兜底的意义就是"本地读不出来，那就问模型" —— 本地那部分不必再来
    一遍。
    没接混合读取器时（纯 VisionService、测试桩），直接用
    `read_question` —— 那条路上本来就没有本地 OCR 可重做。
    """
    visual = getattr(vision, "read_question_visual", None)
    if visual is None:
        return vision.read_question(image, prompt=PROMPT_PK, timeout=timeout)
    return visual(image, prompt=PROMPT_PK, timeout=timeout)
def _poll_sleep(totals) -> None:
    """轮询间隔的 sleep —— 归到单独一类。
    等待阶段每轮都 sleep 一次 `_POLL_INTERVAL`。单看一轮是 100ms，
    但一轮里还夹着别的操作，10 轮累计就是 1 秒。归因出来才知道
    "轮询本身"占多少 —— 那是纯等待，不产生任何进展。
    **sleep 恰好一次**：计时失败绝不能导致多睡一轮（那会让等待阶段
    悄悄变慢）。所以计时包在 try 里，而 sleep 在 try 之外。
    """
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
    """等一张"正在答题"的截图出现。超时返回 None。
    优先用本地题号证据确认界面，求答案交给答题阶段。
    匹配等待页也有青色按钮（"邀请好友"/"自动匹配"），几何上跟答题页的
    选项按钮一模一样 —— 早期版本只看按钮，把它们当成选项点了下去，
    于是误点「取消匹配」，对局刚开始就被自己取消（2026-10-17 真实发生）。
    watch 传入 Stopwatch 时记录等待阶段的模型调用次数与总耗时。
    ocr 传入 RapidOcrReader（已预热），用来读界面上的题号 —— 见
    capture_context.question_no，也用来在模板判否时做第二判据
    （见 _read_progress_line）。
    ocr_fallback_seconds 覆盖「本地 OCR 连续判否多久后允许一次视觉
    模型兜底」（见 _OCR_FALLBACK_SECONDS）。None = 用模块常量。
    """
    if capture_context is not None:
        capture_context.reset()
    local_only = (capture_context is not None
                  and callable(getattr(vision, "read_question_local", None))
                  and getattr(vision, "local_ocr_available", None) is not False)
    wait_started = time.monotonic()
    deadline = time.time() + timeout
    polls = 0
    skipped = 0
    # 等待阶段的三种调用统计：ocr_attempts（本地 OCR 尝试）、
    # model_calls（真实 HTTP 请求数）、fallback_calls（视觉兜底次数）。
    # model_calls 用 usage 差值计算，反映真实网络调用，不掩盖"等待期
    # 几乎不调模型"的事实。
    ocr_attempts = 0
    model_calls = 0
    fallback_calls = 0
    # 本地 OCR 文本路径连续失败次数。达上限后停止尝试、走视觉兜底，
    # 把「连续 N×8 秒阻塞」压成「8 秒 + 一次兜底」。
    text_path_failures = 0
    # 冷却期跳过的轮数。到 _TEXT_PATH_COOLDOWN_POLLS 重新允许试一次。
    cooldown_polls = 0
    # 题号行判据缓存。模板判否时才读，避免匹配页上重复浪费 800ms。
    # 缓存绑定画面状态，界面变了就失效。
    progress: bool | None = None
    progress_at: float = 0.0
    progress_baseline = None
    progress_size = None
    progress_text = ""
    # 被本地判据判否、未发起模型调用的轮数。用于判断等待期是否反复阻塞。
    gated_polls = 0
    screens: dict[str, int] = {}
    # 连续被本地 OCR 判否的轮数，达阈值允许一次视觉兜底。按连续计、
    # 读出来一次归零，避免"偶尔判否"攒够额度突然阻塞。
    ocr_rejects = 0
    ocr_fallback_used = False
    _interval = _POLL_INTERVAL or 0.1
    _fallback_seconds = (_OCR_FALLBACK_SECONDS if ocr_fallback_seconds is None
                         else ocr_fallback_seconds)
    _ocr_fallback_enabled = _fallback_seconds > 0
    _ocr_fallback_polls = max(
        1,
        round(_fallback_seconds / _interval),
    )
    first_visual_at: float | None = None
    # 等待阶段耗时归因：区分抓屏门控、题号行 OCR、本地 OCR、
    # 模型请求（网络）、轮询等待各自的耗时。
    totals = PhaseTotals()
    while time.time() < deadline:
        captured_at = time.monotonic()
        # 抓屏 + 三条像素门控合起来算一类 —— 每轮都做，毫秒级。
        gate_started = time.perf_counter()
        image = grab_screen(region)
        polls += 1
        # 卡片结构预筛：认答题卡片左上角框线转角。便宜（~1ms）。
        # 匹配页在此判否；None 是"不知道"需兜底，False 是确凿否定。
        # 永远不授权点击 —— 点击只认模型判定的 screen。
        template_ok = looks_like_question_template(image)
        # 渲染完成？按钮最先出现，只等"有按钮"会在半渲染界面停住。
        # 判据用题干正文区必须有实际文字。省掉每轮 is_question_stable。
        rendered = has_rendered_stem(image)
        # 连续门控不通过就强制问模型：判据失效时不至于永远卡在等待。
        # 两条件互补：template_ok（结构）让 rendered（题干画完）独立判断。
        # 两条都过才优先调模型；不满足时兜底按时间强制问。
        ready = (template_ok is not False and rendered)
        totals.add("抓屏+门控", time.perf_counter() - gate_started, unit="轮")
        if first_visual_at is None and rendered:
            first_visual_at = captured_at
        # **模板判否时，要用题号行佐证，才真的不调模型。**
        #
        # 模板认的是卡片**样式**（框线转角），题号行认的是**语义**
        # （「第 N 题·共 M 题」）。改版会改样式、不会改语义 —— 所以
        # 两者都判否时，这一帧**确定**不是答题页，跳过它不调模型；
        # 两者不一致（模板判否但题号行在）说明**模板失效**了，
        # 那正是必须调模型的情形。
        #
        # 实测（全部真实帧）：题号行判据 67 张答题帧全 True、2 张匹配页
        # 全 False、9 张结算页 None —— 零假阴性。
        #
        # 题号行要跑 OCR（约 800ms，RapidOCR 的固定开销），所以**只在
        # 模板判否时才跑** —— 那是少数情况（匹配页 + 结算页）。
        if template_ok is False:
            changed = (progress_baseline is None or progress_cache_changed(
                progress_baseline, progress_size, image))
            retry_unknown = (progress is None
                             and time.monotonic() - progress_at >= _PROGRESS_CACHE_SECONDS)
            if changed or retry_unknown:
                progress_baseline = progress_cache_mask(image)
                progress_size = image.size
                progress_started = time.perf_counter()
                progress, progress_text = _read_progress_line(image, ocr)
                totals.add("题号行OCR", time.perf_counter() - progress_started,
                           unit="次")
                progress_at = time.monotonic()
            if progress is False:
                # 模板与题号行都判否 → 确定不是答题页。跳过，不调模型。
                gated_polls += 1
                _poll_sleep(totals)
                continue
            if progress is True:
                # 题号行在但模板判否 → 模板失效（改版），必须调模型。
                ready = True
            # progress is None 时落到正常门控逻辑。
        else:
            # 模板放行 → 题号行缓存作废（界面变了）。
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
        # 先跑本地 OCR，快路径读出来就直接用（文本路径 ~1.5 秒）；
        # 读不出抛 OcrRejected，继续轮询不阻塞。
        question = None
        # 记一次 usage —— 之后用差值算真实 HTTP 请求数。
        usage_before = usage_snapshot(vision)
        def _count_requests():
            """把这一轮真实发出的 HTTP 请求数累加进 model_calls。
            每条退出路径都必须调它：异常时请求已发出，不加会漏计。
            """
            nonlocal model_calls
            after = usage_snapshot(vision)
            if usage_before is not None and after is not None:
                model_calls += after.requests - usage_before.requests
        # 文本路径三种结局：返回题目=成功、OcrRejected=本地门控判否、
        # AnswerUnavailable=网络往返失败。
        text_path_succeeded = False
        local_read = None
        # 仅靠题号行确认"这是答题页"（选项数对不上时，不要丢掉题号信息）。
        # 确认了也不拿这份读数答题：答题阶段会重新读一遍。
        confirmed_by_progress = progress is True and local_only
        # 服务端出问题时两条网络路径都会失败，连续失败到上限就纯像素轮询，
        # 避免 N×3 秒的连续阻塞。
        skip_visual = False
        try:
            # **文本路径连续失败够多次就冷却**（2026-10-30 14:01 那局的教训）。
            #
            # 冷却期里跳过**两条**网络路径：图片路径同样是网络往返，
            # 换一条路并不能躲开"服务端在慢"这件事。
            if confirmed_by_progress:
                pass  # 独立题号已确认，不再追加整图 OCR。
            elif text_path_failures >= _TEXT_PATH_MAX_FAILURES:
                question = None
                skip_visual = True
            else:
                effective_timeout = model_timeout_seconds if model_timeout_seconds is not None else _WAIT_MODEL_TIMEOUT
                question, local_read = _try_ocr_first(vision, image,
                                                      timeout=effective_timeout,
                                                      model_timeout_seconds=model_timeout_seconds,
                                                      totals=totals,
                                                      local_only=local_only,
                                                      capture_context=capture_context)
                text_path_succeeded = question is not None
        except OcrRejected as exc:
            screens["OCR未读出"] = screens.get("OCR未读出", 0) + 1
            ocr_rejects += 1
            # 题号行读到 → 这确实是答题页（只用题号行判断）。
            # 选项数对不上时不能丢掉"这是不是答题页"的结论 → 分开处理。
            if _has_progress_line(exc.read):
                progress_text = getattr(exc.read, "progress_text", "") or progress_text
                confirmed_by_progress = True
                screens["答题页(题号行)"] = screens.get("答题页(题号行)", 0) + 1
                if recorder:
                    safe_log(
                        recorder,
                        f"等待：题号行读到 → 这是答题页；本地读数不全"
                        f"（{exc}），确认界面、读数交答题阶段重读",
                    )
            elif (not local_only and _ocr_fallback_enabled
                    and not ocr_fallback_used
                    and ocr_rejects >= _ocr_fallback_polls
                    and template_ok is True):
                # 连续判否够久 → 允许一次视觉模型兜底。
                # 要求模板明确放行，模板判不出时由门控兜底（30 秒）接手。
                # 只允许一次：两次 8 秒 > 12 秒题窗。
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
            # OCR 路径已发出请求但没拿到答案（超时/网络/模型没给合法下标）。
            # 记一次失败、冷却文本路径，下一轮不再发文本请求。
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
        confirmed_by_progress = confirmed_by_progress and question is None
        if confirmed_locally:
            progress_text = getattr(local_read, "progress_text", "") or progress_text
            if recorder:
                safe_log(recorder, "等待：本地读题门控通过，交接答题阶段求答案")
        if question is None and skip_visual:
            # 服务端在慢 → 纯像素轮询，不发任何网络请求。
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
            # 没有走 OCR 快路径 → 就走图片路径 + 短超时。
            # confirmed_locally 时不走这里：读数已够答题阶段用，不发重复请求。
            try:
                # 单次网络往返上限（见 _WAIT_MODEL_TIMEOUT）。
                # 在「OCR 文本路径 → 图片路径」级联中，两段各 5 秒 ≤ 题目倒计时。
                # 真正的修复是"匹配页不调模型"，短超时是第二道保险。
                if capture_context is not None and capture_context.model_started_at is None:
                    capture_context.model_started_at = time.monotonic()
                visual_started = time.perf_counter()
                try:
                    effective_timeout = model_timeout_seconds if model_timeout_seconds is not None else _WAIT_MODEL_TIMEOUT
                    question = _read_visual(vision, image, timeout=effective_timeout,
                                            model_timeout_seconds=model_timeout_seconds)
                    _record_answer(capture_context, question)
                finally:
                    # 失败也要记 —— 那次网络往返同样花掉了时间。
                    totals.add("模型请求", time.perf_counter() - visual_started,
                               unit="次")
            except RecognitionTimeout:
                # 识别耗光了单题预算。记一次、让上层用最新画面重新判断，
                # 比继续轮询到 120 秒超时强。
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
        # 统计这一轮真实发出的 HTTP 请求数（usage 差值）。
        # OCR 快路径成功时不发声；纯像素测试桩没有 usage 则跳过计数。
        _count_requests()
        # 本地 OCR 读出题目 → 连续判否计数归零；
        # 文本路径成功 → 重新允许用它（视觉兜底不算文本恢复）。
        ocr_rejects = 0
        if text_path_succeeded:
            text_path_failures = 0
        # 本地证据独立确认界面；点击仍需答案及 fresh 身份校验。
        if confirmed_locally or confirmed_by_progress or is_question(question):
            confirmed_at = time.monotonic()
            if capture_context is not None:
                capture_context.captured_at = captured_at
                capture_context.confirmed_at = confirmed_at
                capture_context.ocr_read = local_read if confirmed_locally else None
                _store_progress(capture_context, progress_text)
            if watch:
                watch.mark(f"等待:命中答题页({watch.elapsed():.1f}s)")
                watch.report(
                    f"等待阶段（{model_calls} 次模型调用/{ocr_attempts} 次本地 OCR/"
                    f"{polls} 轮像素，其中 {gated_polls} 轮被本地判据拦下，"
                    f"{fallback_calls} 次视觉模型兜底）"
                )
            # **等待阶段的耗时归因**（2026-10-31 增补）。
            #
            # 上面那行只给一个总数。"等待阶段 耗时 12485ms" —— 这 12 秒
            # 花在哪？**看不出来**。而实测那 12 秒里只有 2.2 秒是网络，
            # 其余是本地 OCR 与轮询 —— 这个差别只有分开归因才看得见，
            # 也只有看得见才会有人去改它。
            #
            # 各类的含义见上面 totals 的注释。归因是**观测**，不参与门控。
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
            # **调用后检查总超时**（2026-10-30 修）。
            #
            # 所以这里**只查总超时**：超了就立刻收工，把手上这张图交出去
            # （宁可蒙题也别把整局拖死）。
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
                         watch, error: str, *, ocr=None) -> dict:
    """识别失败/超时后，蒙第一个按钮。
    超时惩罚与答错相同（不得分），蒙答有 25~33% 胜率。
    必须重新抓屏定位按钮：超时后屏幕可能已换题，旧坐标会点到别的题上。
    题号从 fresh 图读，确保记录的题号与实际点到的题目一致。
    guess_on_failure 为假时保持跳过行为。
    """
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
    click_box(box, region, scale, pause=0.15)
    if watch is not None:
        watch.mark("超时蒙题点击")
    # **题号读自 fresh 图** —— 它才是我们真正点到的那道题。
    # 读不出就留 None（"不知道"，绝不猜）。
    number, total, raw = read_question_number(fresh, ocr=ocr)
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
    """用**已经读好的**题干与选项走一次文本路径，只要下标。
    模型没给出合法下标时返回 None（不抛异常）。
    走的是混合读取器的 `answer_from_ocr_read` —— 它内部复用与 OCR 快
    路径**完全相同**的那段代码（`_answer_from_text`），所以：
    - 越界下标照样被挡住（越界映射回按钮会点错）；
    - 返回的 VisionQuestion 带 `local_read=True`，于是 8 秒图龄闸门
      对它不生效（题干和按钮是同一时刻的像素）。
    没有这个入口时（纯 VisionService、测试桩）返回 None —— 调用方按
    "没有预读"处理，走原来的路径。
    """
    answer = getattr(vision, "answer_from_ocr_read", None)
    if answer is None:
        return None
    if deadline is None:
        return answer(ocr_read, timeout=3.0)
    deadline = min(deadline, time.monotonic() + 3.0)
    if time.monotonic() >= deadline:
        raise RecognitionTimeout("deadline_exceeded")
    question = answer(ocr_read, timeout=3.0, deadline=deadline)
    if time.monotonic() >= deadline:
        raise RecognitionTimeout("deadline_exceeded")
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
) -> dict:
    """答一道题：识别 → 定位 → 点击。返回这道题的处理结果。
    image / question 可以传入已抓好的截图和已识别好的题目 ——
    wait_for_question 已经做过一次识别，直接复用，零额外耗时。
    这是速度的关键：PK 只有 9~12 秒，多一次模型调用就可能超时。
    识别失败时：guess_on_failure=True 会点第一个按钮（标记 guessed=True），
    否则抛 AnswerUnavailable 停止。
    watch 传入 Stopwatch 时记录本阶段的分段耗时（纯观测，不影响流程）。
    captured_at 是模型输入图采集前的 monotonic 时刻；传图但缺少时刻、
    年龄达到 8 秒或 fresh 校验失败时返回 skipped，不猜测、不点击。
    ocr_read 是等待阶段通过门控的读数，答题阶段复用一次文本入口，
    不重做 OCR；未传预读的正常题保留原有级联预算。
    2026-10-30 16:46 那局：等待阶段本地早已读准（题干、选项、题号行
    全有），模型失败 → 程序继续等 → 连发三次 3 秒的请求 + 冷却，
    第 1 题出现又走完的整个窗口正好被吃光。带这份读数进来，第 1 题
    就能被真正处理掉。
    """
    own_watch = watch is None
    watch = watch or Stopwatch(recorder)
    start = usage_snapshot(vision)
    reused_usage = getattr(vision, "last_question_usage", None) if question is not None else None
    not_question = False
    if capture_context is not None:
        if captured_at is None:
            captured_at = capture_context.captured_at
        _store_progress(capture_context, getattr(ocr_read, "progress_text", ""))
    deadline = None
    def expired():
        if deadline is not None and time.monotonic() >= deadline:
            return True
        # OCR 路径不受图龄约束：题干和选项是本地从同一张图读出的，
        # 它们和按钮是同一时刻的像素，多少秒后依然成立。闸门是给
        # "模型看图"设计的 —— 那条路上模型读的是一张已经过期的截图。
        #
        # 保留的校验：点击前那一张 fresh 图的 is_same_question 对两条路
        # 都还在跑（见下面），所以"界面真的没变"依然被验证。
        #
        # 而且这一条尤其重要：那一局等待阶段花了 31.7 秒（16:46:14 →
        # 16:46:46），远超 8 秒图龄 —— 若闸门照常生效，这份读数会被
        # 当成过期图丢掉，第 1 题**仍然**会丢。
        if question is not None and getattr(question, "local_read", False):
            return False
        if ocr_read is not None and question is None:
            return False
        return captured_at is not None and time.monotonic() - captured_at >= _MAX_CAPTURE_AGE
    def skip(reason, error=""):
        if verbose:
            print(f"  跳过（{reason}），未点击" + (f"：{error}" if error else ""))
        return {
            "stem": getattr(question, "stem", ""),
            "options": getattr(question, "options", []),
            "answer": None,
            "guessed": False,
            "skipped": True,
            "reason": reason,
            "error": error,
        }
    try:
        if image is None:
            captured_at = time.monotonic()
            image = grab_screen(region)
            if capture_context is not None:
                capture_context.captured_at = captured_at
            watch.mark("抓屏")
        if captured_at is None:
            return skip("capture_time_unknown")
        deadline = captured_at + 6.0
        if expired():
            # 图进来就已经超龄 —— 调模型纯属浪费（一次最多 11 秒），
            # 直接蒙。注意这里不能省：2026-10-20 的日志里有两道题
            # 「截图落盘 21041ms」，就是在一个注定超龄的图上干等了 21 秒。
            return _guess_after_timeout(
                region, scale, image, question, skip, guess_on_failure, watch,
                "采集时已过期（不调用模型）", ocr=ocr)
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
                    return _guess_after_timeout(
                        region, scale, image, question, skip, guess_on_failure, watch,
                        "deadline_exceeded", ocr=ocr)
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
            else:
                try:
                    model_started = time.monotonic()
                    try:
                        question = vision.read_question(image, prompt=PROMPT_PK, deadline=deadline)
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
                except RecognitionTimeout:
                    # 识别把单题预算耗光了（见 pk_hybrid 的级联预算检查）。
                    # 这时截图早超龄，再做任何识别都是白等 —— 直接蒙。
                    #
                    # 它和下面那个通用的 AnswerUnavailable 分支的差别是：
                    # 通用分支还要看 expired() 判断"是不是顺便超龄了"，
                    # 而超时**必然**意味着超龄，不必再判一次。
                    return _guess_after_timeout(region, scale, image, question,
                                               skip, guess_on_failure, watch,
                                               "recognition_timeout", ocr=ocr)
                except AnswerUnavailable as exc:
                    if expired():
                        # 模型调用本身失败（超时/网络）—— 这时连题目都没读出来，
                        # 但**按钮还在屏幕上**，可以蒙一个。跳过的收益是 0%，
                        # 蒙的收益是 25~33%。
                        return _guess_after_timeout(region, scale, image, question,
                                                   skip, guess_on_failure, watch,
                                                   f"识别失败：{exc}", ocr=ocr)
                    raise
                watch.mark("模型识别")
        if expired():
            return _guess_after_timeout(
                region, scale, image, question, skip, guess_on_failure, watch,
                "模型返回时已过期", ocr=ocr)
        _record_answer(capture_context, question)
        # 匹配页也有青色按钮；模型的界面分类仍是点击前置条件。
        if not is_question(question):
            raise NotAQuestionScreen(f"这不是答题界面（模型判定为 {question.screen}），不能点")
        boxes = find_option_buttons(image)
        watch.mark("定位按钮")
        guessed = False
        error = ""
        try:
            answer = question.answer_index
            if answer is None:
                raise AnswerUnavailable("模型没给出合法答案下标")
            if not (0 <= answer < len(boxes)):
                raise AnswerUnavailable(
                    f"模型答案下标 {answer} 越界（共 {len(boxes)} 个按钮）"
                )
            if not question.bare_index and len(question.options) != len(boxes):
                raise AnswerUnavailable(
                    f"模型读到 {len(question.options)} 个选项，检测到 {len(boxes)} 个按钮，不一致"
                )
        except AnswerUnavailable as exc:
            if not guess_on_failure:
                raise
            # 仅候选答案校验失败可蒙答，不扩大到网络、年龄或身份失败。
            answer = 0
            guessed = True
            error = str(exc)
        # 正常、裸序号、首题复用及蒙答共用此门。fresh 仅在内存中使用，
        # 留证和下一题等待的基准始终保留模型输入原图。
        if expired():
            return _guess_after_timeout(
                region, scale, image, question, skip, guess_on_failure, watch,
                "点击前已过期", ocr=ocr)
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
            # 超时改成蒙答（2026-10-20）。PK 里超时的惩罚与答错相同 ——
            # 这题不得分 —— 所以"点第一个"有 25~33% 机会，什么都不点
            # 是 0%。原来这里直接 skip，白丢的是实打实的期望收益。
            #
            # 这里可以直接用上面刚校验过的 box —— is_same_question 已确认
            # 界面没换题，fresh_boxes 就是当前屏幕的按钮位置，比
            # _guess_after_timeout 再抓一次更省也更准。
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
        watch.mark("蒙题点击" if guessed else "点击")
        if verbose:
            print(f"  {question.stem or '(模型未给题干)'}")
            if guessed:
                print(f"    {error or '识别失败'}，蒙第一个按钮（y={box.center_y:.0f}）")
            elif question.bare_index:
                print(f"    模型只回序号 {answer + 1}（y={box.center_y:.0f}）")
            else:
                print(f"    选项：{question.options}")
                print(f"    点：{question.options[answer]}（下标 {answer}，y={box.center_y:.0f}）")
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
            emit_usage(f"第 {question_no} 题",
                       reused_usage if reused_usage is not None else usage_since(vision, start),
                       recorder)
def wait_for_next_question(region, current_image, watch: "Stopwatch | None" = None, *,
                           capture_context: PKCaptureContext | None = None,
                           delay: float | None = None):
    """等换到下一题。返回 (截图, 识别结果)；界面不对返回 (None, None)。
    **固定延迟 + 补等，且要求题干已换**（2026-10-20 晚，实测修正）。
    原来是「抓帧 + 题干区帧差 + 相邻两帧稳定」的轮询，理由是「按钮会在
    过渡期消失，只看按钮会误判对局结束」。实测这个理由站不住：
        轮询间隔 0.25s → 9 帧、2.42 秒
        轮询间隔 0.10s → 20 帧、2.35 秒
    帧数随间隔**倒数缩放** —— 界面渲染时间固定在 ~2.0 秒，间隔越小
    越早撞上过渡帧，需要更多帧才走得完。旧注释说的「每题省 1.4 秒」
    在实测里只有 70ms。每帧的固定开销（实测抓屏 16.6ms + 判定 6.8ms）
    乘以帧数，正是降间隔省下的那部分 —— 一来一回基本抵消。
    固定等待把 20 次抓屏判定换成 1 次 sleep + 1 次抓屏：实现更简单，
    总耗时相同。
    **但只看"有没有按钮"是不够的**（2026-10-20 18:19 那局的实测）：
    按钮在渐显动画里**最先**画完，题干最后画完。固定延迟 1.0 秒时按钮
    已经回来了，但题干还是上一题 —— 于是返回半成品帧，下一轮识别出
    同一道题，点击前校验判 screen_changed 跳过，**每道真题被处理两次**。
    那局的实测：10 轮上限只答了 5 题，比分 50:105 落败。
    所以补等条件改成**两个都要满足**：
        1. 帧里有选项按钮（是答题界面，不是结算页）
        2. 题干区相对**上一题**已经变了（渲染真的完成了）
    第 2 条用 `is_new_question`（题干区二值化帧差 > 20）。实测区分度很高：
    同一题严格为 0，换题为 1513~2856 像素。倒计时在 y214-227，题干带
    是 y250-345，两者不重叠 —— 倒计时跳动不会污染这条判据。
    current_image 是**上一题**的图，用作题干比对的基准。
    等待上限 = delay + _EXTRA_WAIT_MAX，正常题 1.0 + 3.0 = 4.0 秒。
    之前这里还有一个 timeout 参数（6.0 / 15.0），函数体里一次都没用过，
    2026-10-20 删掉 —— 留着只会让人以为超时由它控制。
    watch 传入 Stopwatch 时记录本阶段耗时 —— 这是每题耗时的大头。
    """
    if capture_context is not None:
        capture_context.reset()
    wait_seconds = _NEXT_QUESTION_DELAY if delay is None else delay
    time.sleep(wait_seconds)
    captured_at = time.monotonic()
    candidate = grab_screen(region)
    # 补等分**两个阶段**，各自计时：
    #
    #   1. 普通阶段（_EXTRA_WAIT_MAX = 3 秒）—— 等渲染完成：按钮回来、
    #      题干换新。
    #   2. 揭示阶段（_REVEAL_WAIT_MAX = 13 秒）—— 认出"答后揭示态"时
    #      进入。揭示态不是渲染过程，是小程序在等本题倒计时走完。
    #
    # **为什么必须分开**：2026-10-20 21:24 那局，21:25:25 点完第 6 题，
    # 21:25:29（固定 1.0s + 补等 3.0s 全部耗尽）抓到的 007 号截图
    # **仍然是第 6 题** —— 题号行写着「第6题·共10题」，只是正确项
    # 「哺乳类」染了绿、比分从 65:60 变成 80:70、倒计时从 00:12 走到
    # 00:06。小程序根本没推进，它在等这 12 秒走完。
    #
    # 阶段切换时 phase_waited 归零，所以离开揭示态后仍有完整的 3 秒
    # 给渲染 —— 两个阶段的预算不互相挤占。
    #
    # 首帧先看一眼是不是揭示态：固定延迟 1.0 秒时揭示态往往已经出现，
    # 先认出来才能一上来就用长预算 —— 否则会白等 3 秒普通预算才升级。
    revealed = has_answer_reveal(candidate)
    phase_waited = 0.0
    total_waited = 0.0
    if revealed and watch:
        # 首帧就是揭示态 —— 这时没有"进入"的转变可记，但必须留痕，
        # 否则日志里看不出这一轮是在揭示态里度过的。
        watch.mark("等下一题:处于揭示态")
    while not _is_next_question_ready(candidate, current_image):
        budget = _REVEAL_WAIT_MAX if revealed else _EXTRA_WAIT_MAX
        if phase_waited >= budget:
            break
        time.sleep(_EXTRA_WAIT_STEP)
        phase_waited += _EXTRA_WAIT_STEP
        total_waited += _EXTRA_WAIT_STEP
        captured_at = time.monotonic()
        candidate = grab_screen(region)
        # 揭示态的出现和消失都换一次预算：出现是因为要等倒计时走完
        # （长预算），消失是因为回到普通渲染节奏（短预算）。
        if has_answer_reveal(candidate) != revealed:
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
        return None, None
    if not _is_next_question_ready(candidate, current_image):
        # 有按钮但题干始终没变 —— 三种可能：
        #   1. 下一题和上一题**题干相同**（题库里有重复题）；
        #   2. 界面卡在上一题；
        #   3. 揭示态比预算还长（实测每题倒计时 12 秒，13 秒预算够用，
        #      走到这一步说明界面出问题了）。
        #
        # 这时**不能**判"对局结束" —— 那会白白丢掉后面几题。
        # 宁可返回这一帧：万一真是重复题，答它是对的；万一是界面
        # 卡住，下一轮的 is_same_question 校验会拦下（记为跳过）。
        if watch:
            watch.mark("等下一题:题干未变(按重复题处理)")
    if capture_context is not None:
        capture_context.captured_at = captured_at
    return candidate, None
def _is_next_question_ready(candidate, current_image) -> bool:
    """这一帧是不是"下一题已经渲染完成"。
    三个条件都要满足：
      1. **不在答后揭示态** —— 见下，这是 2026-10-20 22:27 那局的教训
      2. 题干已经换了 —— 渲染真的做完了（按钮比题干先画完）
      3. 有选项按钮 —— 是答题界面，不是结算页
    **条件 1 为什么必需**（2026-10-20 22:27 那局）：
    那一局第 9、10 轮都是「最后一题·分数翻倍」，两道不同的题。但
    第 10 轮抓到的是第 9 题的**揭示帧** —— 正确项染绿、比分更新，
    题干和题号纹丝不动。此时题干帧差只有 **89**（揭示态下按钮文字
    的渲染微动），**刚好越过 20 像素的换题阈值**，于是
    "题干已换"判据放行了它。
    结果：第 10 轮重新识别出第 9 题，点击前校验判 screen_changed
    跳过 —— 那局的最后一题就这么被吃掉了。
    上一轮修复（21:24 那局）只把揭示态接到了**预算切换**上，没接到
    **就绪判据**上：等待循环第一轮就判就绪直接返回，补等只花了 4 毫秒。
    揭示帧必须在这里就被挡住，不能只靠预算。
    **判定顺序：先揭示态，后题干，最后按钮**（2026-10-20 优化）。
    实测题干判据 0.89ms、按钮判据 2.99ms —— 补等循环每步都要调，
    而"题干没变"是最常见的"还没准备好"情形。先判便宜的，
    就不用为"注定不合格"的帧付按钮检测的钱。
    current_image 为 None 时退化成只看条件 2、3 —— 没有基准可比对时，
    总比什么都不看强。
    """
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
    model_timeout_seconds: float | None = None,
) -> PKReport:
    """跑完一局 PK。
    每题：等界面 → 识别 → 点击 → 等下一题（固定延迟 + 补等，按钮判据）。
    结束后读结算页，给出明确的胜负结论。
    ocr 传入 RapidOcrReader（已预热），用来读界面上的题号。题号是
    **唯一**能告诉程序"这是第几题"的信号 —— 2026-10-20 22:27 那局
    程序从第 2 题才开始（第 1 题在等待阶段就开跑了），光看题干和
    选项完全发现不了。
    ocr_fallback_seconds 覆盖「本地 OCR 连续判否多久后允许一次视觉
    模型兜底」（见 _OCR_FALLBACK_SECONDS）。None = 用模块常量。
    """
    report = PKReport()
    watch = Stopwatch(recorder)
    if verbose:
        print(f"等待进入答题界面（最多处理 {max_questions} 轮，含跳过；未必对应实际题号）…")
    capture_context = PKCaptureContext()
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
        # 每题重新计时。report() 只累加不清零，所以不在这里 reset 的话，
        # 第 N 题那行会把前 N-1 题的分段全部重列一遍（2026-10-18 实测
        # 耗时从 320ms 一路涨到 50216ms，完全没法读）。
        watch.reset(recorder)
        if question_no == 1:
            # 等待阶段那次识别的耗时要算进第 1 题，所以 reset 之后再补一刀。
            watch.mark("首题识别(复用等待阶段)")
        # 等待阶段带出来的本地读数**只给第 1 题用**。
        #
        pre_read = capture_context.ocr_read
        capture_context.ocr_read = None
        if verbose:
            print(f"\n[第 {question_no} 轮处理]")
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
            )
        except NotAQuestionScreen as exc:
            # 界面不是答题页（匹配等待 / 结算）—— 不是错误，是正常结束
            report.outcome = "questions_exhausted"
            if verbose:
                print("  " + str(exc))
            _read_final(region, report, vision, recorder, verbose)
            if verbose:
                print("")
                print(report.summary())
            return report
        except (AnswerUnavailable, WqError) as exc:
            report.outcome = "interrupted"
            report.error = str(exc)
            if verbose:
                print(f"  停止：{exc}")
            _read_final(region, report, vision, recorder, verbose)
            return report
        report.questions.append(result)
        skipped = result.get("skipped", False)
        if not skipped:
            report.answered += 1
        # 读这一轮界面上的题号（2026-10-20 22:27 那局）。
        #
        # 走本地 OCR（~600ms），不调模型、不花 token。读不出就留 None。
        #
        # **第 1 题复用等待阶段那份读数**（2026-10-31）：等待阶段读
        # 题号行时已经把原文带回来了，而第 1 题的图就是那张图 ——
        # 同一条带 OCR 两遍是纯浪费（实测 2744ms）。
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
        )
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
            _read_final(region, report, vision, recorder, verbose)
            if verbose:
                print(f"\n{report.summary()}")
            return report
        # 这里**不再调模型读题号**（2026-10-17 实测删掉）。
        #
    report.outcome = "questions_exhausted"
    _read_final(region, report, vision, recorder, verbose)
    if verbose:
        print(f"\n{report.summary()}")
    return report
def _read_final(region, report: PKReport, vision, recorder, verbose: bool) -> None:
    """对局结束后读明确胜负文字或奖励积分，把结果写进 report。"""
    from .result import read_result
    time.sleep(1.5)  # 等结算页动画播完
    image = grab_screen(region)
    if recorder:
        recorder.shot("final_result", image=image)
    report.final = read_result(image, vision=vision)
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
