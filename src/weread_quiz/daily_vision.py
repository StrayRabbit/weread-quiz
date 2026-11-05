"""每日一答模型适配：独立 prompt、请求预算、裁图和明确文字结算判读。"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

from PIL import Image

from .errors import NotAQuestionScreen
from .vision import (
    VisionService,
    VisionQuestion,
    is_question,
    strip_think,
    parse_vision_response,
    _check_deadline,
)

# 每日一答的精简 prompt。
#
DAILY_PROMPT = """这是微信读书每日一答的答题界面截图。只输出一个 JSON 对象，不要任何解释：
{"stem":"题干原文","options":["从上到下第1个选项",...],"answer":正确选项在options中的下标，从0开始}
options 必须按画面从上到下的顺序完整抄下，不要遗漏、不要改写。"""

# 每日一答固定 12 题（与小程序规则一致；不在 prompt 里问，避免答非所问）。
DAILY_TOTAL_QUESTIONS = 12

DAILY_VISION_TIMEOUT = 11.5
DAILY_VISION_RETRIES = 0

# 文本路径（OCR 快路径）与图片回退之间的预算切分。
#
# 文本路径慢一次要吃掉 5 秒，而识别总预算 9 秒 —— 回退若被饿死到
# 2.5 秒，模型慢一点就又超时，两条路同时空，只能蒙答（2026-11-02
# 实测：第 5 题「中东地区石油出口量…占世界的百分比」就是这样错的）。
#
# 所以文本路径最多只能用「剩余预算减去这份保留」，图片回退始终拿得到
# FALLBACK_RESERVE_SECONDS —— 数值与 PK 的 _IMAGE_PATH_MIN_BUDGET 对齐。
FALLBACK_RESERVE_SECONDS = 3.5


def text_path_deadline(deadline: float, now: float, timeout: float) -> float:
    """文本路径的截止时刻 —— 必须给图片回退留出预算。

    剩余预算不够两条路分时全给文本路径：它更快更便宜（输入 token 只有
    图片路径的六分之一），而此时已经没有回退的余地。

    识别预算已经耗光（deadline 早于 now）时，文本路径仍给自己一份
    独立的短预算 —— 预读的钱已经付了，文本请求又便宜又快；而图片
    回退**不**因此延长，它仍受原 deadline 约束。
    """
    remaining = deadline - now
    if remaining <= 0:
        return now + timeout
    budget = min(timeout, remaining - FALLBACK_RESERVE_SECONDS)
    if budget <= 0:
        budget = remaining
    return now + budget


@dataclass
class DailyReport:
    """每日一答的一局结果。"""

    answered: int = 0
    questions: list[dict] | None = None
    outcome: str = "unknown"  # success / fail / questions_exhausted / wrong / final_pending / interrupted / error
    error: str = ""
    skipped: int = 0
    # 界面自己写的题号（「第N题·共12题」里的 N），按出现顺序。
    # 读不出就是没这一项 —— **绝不猜**：题号猜错比读不出更糟。
    #
    progress_numbers: list[int] | None = None

    def __post_init__(self) -> None:
        if self.questions is None:
            self.questions = []
        if self.progress_numbers is None:
            self.progress_numbers = []

    @property
    def guessed(self) -> list[int]:
        if self.outcome == "dry_run":
            return []
        return [i for i, q in enumerate(self.questions, 1)
                if q.get("guessed") and not q.get("skipped")]

    @property
    def solid(self) -> int:
        return self.answered - len(self.guessed)

    def summary(self) -> str:
        if self.outcome == "dry_run":
            return "dry-run：只识别了这一题的答案，未做任何点击"
        if self.outcome == "success":
            return f"✅ 全部答对，成功拿满今日奖励（共答 {self.answered} 题）"
        if self.outcome == "fail":
            return f"❌ 答题失败，今日机会已用完（共答 {self.answered} 题）"
        if self.outcome == "wrong":
            return f"第 {self.answered} 题答错，今日机会已用完（共答 {self.answered} 题）"
        if self.outcome == "questions_exhausted":
            return f"答完 12 题（共答 {self.answered} 题）"
        if self.outcome == "final_pending":
            return f"最后题点击完成，未读到最终结果（共答 {self.answered} 题）"
        if self.outcome == "interrupted":
            return f"流程中断（已点击 {self.answered} 题）：{self.error}"
        return f"结束（{self.outcome}），答了 {self.answered} 题：{self.error}"

    def progress_summary(self) -> str:
        """题号对齐的汇总：漏了哪几题。"""
        if not self.progress_numbers:
            return ""
        numbers = sorted(set(self.progress_numbers))
        if not numbers:
            return ""
        highest = numbers[-1]
        missing = [n for n in range(1, highest + 1) if n not in numbers]
        text = f"界面题号 {','.join(str(n) for n in numbers)}"
        if missing:
            text += f" —— 漏了第 {'、'.join(str(n) for n in missing)} 题"
        return text


def build_daily_vision_service(cfg, *, timing_sink=None) -> VisionService:
    """构造模型服务：默认超时 11.5 秒、零重试、严格截断处理，思考开关来自配置。

    协议由代码自己认（见 model_service 模块）—— 主模型一组，视觉模型可选。
    """
    from .model_service import build_model_service

    return build_model_service(
        cfg,
        timeout=DAILY_VISION_TIMEOUT,
        retries=DAILY_VISION_RETRIES,
        recover_truncated_response=False,
        timing_sink=timing_sink,
    )


def read_daily_answer(vision: VisionService, image: Image.Image, *,
                      timeout: float | None = None, deadline: float | None = None,
                      visual_only: bool = False) -> VisionQuestion:
    """读取 daily 答案，转发预算并复用完整截图 OCR；纯图片路径只裁模型输入，非题面抛异常。"""
    if getattr(vision, "expects_full_image", False):
        cropped = image
    else:
        cropped = _DAILY_CROP(image) if image.size[1] > 600 else image
    reader = getattr(vision, "read_question_visual", None) if visual_only else None
    if not callable(reader):
        reader = getattr(vision, "read_question", None)
    if callable(reader):
        question = _call_reader(reader, cropped, timeout=timeout, deadline=deadline)
        if not is_question(question):
            raise NotAQuestionScreen(f"模型判定为 {question.screen}，不是答题界面")
        # 题号（界面自己写的「第N题」）。模型那条路读不出来 —— prompt
        # 里没问题号；OCR 路径已经在 `daily_hybrid` 里带上了。读不出
        # 就是 None，绝不猜。
        if deadline is None and not visual_only and getattr(question, "progress_no", None) is None:
            progress = getattr(vision, "read_progress_number", None)
            if callable(progress):
                try:
                    question.progress_no = progress(cropped)
                except Exception:  # noqa: BLE001 - 读不出当"不知道"
                    question.progress_no = None
        return question
    text = _call_reader(vision.recognize, cropped, timeout=timeout, deadline=deadline)
    question = parse_vision_response(text)
    if not is_question(question):
        raise NotAQuestionScreen(f"模型判定为 {question.screen}，不是答题界面")
    return question


def _call_reader(reader, image, *, timeout=None, deadline=None):
    """按签名传递预算；不以捕获 TypeError 重发请求。"""
    options = {"prompt": DAILY_PROMPT}
    if timeout is not None:
        options["timeout"] = timeout
    if deadline is not None:
        options["deadline"] = deadline
    try:
        params = inspect.signature(reader).parameters
    except (TypeError, ValueError):
        params = None
    if params is not None and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        options = {key: value for key, value in options.items() if key in params}
    _check_deadline(deadline)
    result = reader(image, **options)
    _check_deadline(deadline)
    return result


def _DAILY_CROP(image: Image.Image) -> Image.Image:
    """裁掉顶/底装饰区，只留"答题卡片"。"""
    return image.crop((30, 100, 400, 600))


# 判读结算页的 prompt。只抄画面上明确出现的文字，不许从星级、
# 进度条、颜色等间接线索推测成败 —— 猜错会让用户以为答完了。
RESULT_PROMPT = """这是微信读书每日一答答完 12 题之后的结算页截图。

只看画面上**明确出现的文字**，判断这一局是否成功。只输出一个词：
- 成功：画面明确写着"恭喜完成"、"答题完成"、"今日已完成"、"挑战成功"、"全部答对"、"获得奖励"、"领取奖励"这类表示通关并拿到奖励的文字
- 失败：画面明确写着"答题失败"、"挑战失败"、"答错"、"很遗憾"、"再接再厉"这类表示没通关的文字
- 无：画面上没有以上任何一类文字（空白、只有图形、只有奖励图标等）

不要根据星星数量、进度条、颜色深浅或弹窗样式推测。只输出上述三个词之一。"""

# 模型有时不照指令只回一个词，而是把画面上的原文抄回来
# （"恭喜完成今日答题"、"很遗憾，答题失败"）。这些短语就是上面
# RESULT_PROMPT 里列举过的判据，这里做同样的匹配。
#
_FAIL_PHRASES = ("失败", "答错", "很遗憾", "再接再厉", "不正确")
_SUCCESS_PHRASES = (
    "成功", "恭喜", "完成", "全部答对", "获得奖励", "领取奖励",
    "今日已完成", "通关",
)


def read_daily_result(vision: VisionService, image: Image.Image) -> str:
    """读结算页，返回 "成功" / "失败" / "无"。"""
    text = vision.recognize(image, prompt=RESULT_PROMPT)
    body = strip_think(text)
    # 先看模型给的结论词，再退到画面原文短语，最后才是英文兜底。
    for phrase in _FAIL_PHRASES:
        if phrase in body:
            return "失败"
    for phrase in _SUCCESS_PHRASES:
        if phrase in body:
            return "成功"
    lowered = body.upper()
    if "SUCCESS" in lowered or "COMPLETE" in lowered:
        return "成功"
    if "FAIL" in lowered:
        return "失败"
    return "无"
