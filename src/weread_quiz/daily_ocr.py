"""每日一答 OCR：卡片及拼接读取、独立几何、置信度门控和严格终态文字识别。"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from PIL import Image

from .advance import read_text_lines
from .blank import TextSpan, compose_stem, find_blank_lines
from .daily_buttons import _OPTION_BOTTOM, _OPTION_TOP, find_blue_gray_bands
from .daily_screen import has_answer_reveal
from .ocr_core import OcrDecision, OcrRead, RapidOcrEngine

# 置信度门控阈值。低于它就回退图片路径。
#
# 0.85 正好把"读对了但没把握"的那批挡在门外（单字选项、"小篆"），
# 而把读得稳的全放行。**这些被挡下的其实都读对了** —— 门控在这里
# 牺牲的是速度不是正确性：读不准就回退图片路径，模型自己看图。
#
# 样本仍然偏少，跑几局真实 daily 后看 run.log 里每题记录的置信度
# 分布再校准 —— 调高更保守（更多回退），调低更激进。
DEFAULT_MIN_CONFIDENCE = 0.85

# 答题卡片的裁剪范围。裁的是"题号行 + 题干 + 选项按钮"这一整块。
#
# 上界 185：倒计时圆（"12"，y122）和等级（"LV.2"，y176）在上面。
# 下界 545："这题有问题！"按钮在 y575，在下面。
# 横向 30..400 与按钮的横向范围一致 —— 卡片边框之外没有文字。
_CARD_TOP = 185
_CARD_BOTTOM = 545
_CARD_LEFT = 30
_CARD_RIGHT = 400

_CARD_SCALE = 2.0

# 按钮的纵向区间。用于把 OCR 行归位到按钮。
#
# 数值来自 `daily_buttons._OPTION_TOP / _OPTION_BOTTOM`（点击路径的
# 权威来源）—— 这里直接引用，不再各写一份，改版时不会两处漂移。
_BUTTON_TOP = _OPTION_TOP
_BUTTON_BOTTOM = _OPTION_BOTTOM
_BUTTON_LEFT = 30
_BUTTON_RIGHT = 400

_STRIP_PAD_V = 6      # 按钮内文字区的上下留白
_STRIP_LEFT = 38
_STRIP_RIGHT = 392
_STRIP_SCALE = 2.0

# 拼接时按钮带之间的**垂直间隙**。
#
# 各带的文字区原本是紧挨着拼起来的（np.vstack 不留空隙），RapidOCR 的
# 文本检测器会把三个单字当成**一条竖排文字**读出来 —— 实测
# 「数/面/唠」读成「敷面磅」，「滔/湛/淼」读成「滔湛淼」。于是拼接路
# 在最需要它的单字选项上必然失效（它本来就是为单字选项准备的）。
#
# 16px（原图尺度，放大后 32px）足以让检测器把三行分开。
_STRIP_GAP = 16

_CENTER_TOLERANCE_RATIO = 0.5

_BAND_EDGE_TOLERANCE = 6

_X_TOLERANCE = 12

# 题干区要剔除的非题目文字。
#
# RapidOCR 会把整个卡片都读出来，混进题干的噪声有：等级（"LV.2"）、
# 倒计时（"12"）、题号（"第7题·共12题"）。这些都有固定形态，按正则
# 剔除比按 y 坐标切更稳 —— 题号行的纵向位置会随题型浮动。
_NOISE_PATTERNS = (
    re.compile(r"^\d+$"),
    re.compile(r"^LV\.?\s*\d*$", re.IGNORECASE),
    re.compile(r"^第\s*\d+\s*题"),
    re.compile(r"共\s*\d+\s*题"),
)

# 题号行里的题号。
_RE_PROGRESS_NO = re.compile(r"第\s*(\d+)\s*题")

# 「最后一题」也是题号行，只是没有编号。
#
# 每日一答的卡片在最后一题把「第N题·共12题」换成「最后一题」。认不出
# 它的话，这一题就没有题号证据 —— 门控判否、被迫回退图片路径，题干里
# 还会混进「最后一题」三个字（2026-11-02 实测：选项读得又准又全，
# 却因为这一行被判否）。
_RE_PROGRESS_LAST = re.compile(r"最后一题")


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    return any(pattern.search(stripped) for pattern in _NOISE_PATTERNS)


@dataclass
class DailyOcrRead(OcrRead):
    """每日一答 OCR 读数，包含界面类型证据及可选题号；题号不依赖选项门控是否通过。"""

    has_progress: bool = False
    progress_no: int | None = None


_TERMINAL_REVIVE_PHRASES = frozenset({
    "可以通过以下方式复活",
})

# 成功页完整短语白名单，尚缺真实结算夹具验证；仅作保守文字确认。
_TERMINAL_SUCCESS_PHRASES = frozenset({
    "恭喜完成",
    "答题完成",
    "今日已完成",
    "挑战成功",
    "全部答对",
})

# 失败结算页的完整短语。
#
# 失败页完整短语白名单，必须整行匹配且有有效位置证据。
_TERMINAL_FAIL_PHRASES = frozenset({
    "答题失败",
    "挑战失败",
})

# 短语 → 终态类别。
#
# 值标注成 Literal 是为了让 `TerminalRead.kind` 的约束真正生效 ——
# 声明成 `dict[str, str]` 的话，构造 TerminalRead 时就什么也管不着。
_TERMINAL_PHRASE_KIND: dict[str, Literal["revive", "success", "fail"]] = {
    **{phrase: "revive" for phrase in _TERMINAL_REVIVE_PHRASES},
    **{phrase: "success" for phrase in _TERMINAL_SUCCESS_PHRASES},
    **{phrase: "fail" for phrase in _TERMINAL_FAIL_PHRASES},
}

# 规范化时要剥掉的标点（首尾的所有标点字符，不止一层）。
#
_TERMINAL_PUNCTUATION = (
    set(string.punctuation)
    | set("，。、；：？！“”‘’（）《》〈〉【】〔〕…—－·～「」『』")
)


@dataclass(frozen=True)
class TerminalRead:
    """带类别和完整文字证据的终态结论；调用方须在 OCR 后重抓并验证画面仍一致。"""

    kind: Literal["revive", "success", "fail"]
    evidence: str


def _normalize_terminal_text(text: str) -> str:
    """规范化一行文字：去掉所有空白，再剥掉首尾标点。"""
    compact = "".join(text.split())   # split() 无参 → 去掉所有空白
    return compact.strip("".join(_TERMINAL_PUNCTUATION))


def _is_point_sequence(box) -> bool:
    """验证四个二维点的几何结构，拒绝非有限数值和退化边界。"""
    try:
        points = tuple(box)
    except TypeError:
        return False
    if len(points) != 4:
        return False
    for point in points:
        try:
            coords = tuple(point)
        except TypeError:
            return False
        if len(coords) != 2:
            return False
        for value in coords:
            # RapidOCR 的坐标为 NumPy 实数；布尔值不能作为坐标。
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value)):
                return False
    return True


def _is_option_band_line(box) -> bool:
    """检查文字行是否位于选项区；选项文字不能作为终态证据，缺失有效几何时保守排除。"""
    try:
        points = tuple(box)
        ys = [point[1] for point in points]
        center_y = (min(ys) + max(ys)) / 2.0
    except (TypeError, IndexError, KeyError, ValueError):
        # 框不可用 → 不据此排除（见上面的取舍说明）。
        return False
    if not math.isfinite(center_y):
        return False
    return _OPTION_TOP <= center_y <= _OPTION_BOTTOM


def _terminal_lines(raw) -> list[tuple[str, float, tuple]]:
    """统一原始 OCR 行的文字、置信度和几何结构；读取器只调用一次。"""
    usable = []
    for entry in raw or []:
        try:
            box = entry[0]
            text = entry[1]
            float(entry[2])
        except (TypeError, IndexError, KeyError, ValueError):
            continue
        # 文字必须是 str —— read_text_lines 会调 text.strip()，
        # 传个数字进去会抛 AttributeError，把整批好行带走。
        if not isinstance(text, str):
            continue
        # box 必须是"四个二维数值点"。这里不能靠
        # `tuple(tuple(p) for p in box)` 抛 TypeError 来兜底 ——
        # 字符串能迭代，不抛异常（见 _is_point_sequence 的说明）。
        if not _is_point_sequence(box):
            continue
        usable.append(entry)
    try:
        return read_text_lines(usable)
    except Exception:  # noqa: BLE001 - 畸形行不该把整批带走
        return []


def find_option_bands(image: Image.Image) -> list[tuple[int, int]]:
    """找出选项按钮的纵向区间，返回 [(top, bottom), ...]。"""
    arr = np.asarray(image.convert("RGB"))
    height = arr.shape[0]
    return find_blue_gray_bands(arr, _BUTTON_TOP, min(_BUTTON_BOTTOM, height))


def has_highlighted_button(image: Image.Image) -> bool:
    """按钮区里有没有"已作答"的彩色按钮。"""
    return has_answer_reveal(image)


class DailyOcrReader(RapidOcrEngine):
    """每日一答的 RapidOCR 读取器。"""

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        super().__init__(min_confidence=min_confidence)

    # -------------------------------------------------------------- 整图路

    def read(self, image: Image.Image) -> DailyOcrRead | None:
        """识别答题卡片。"""
        # 答后高亮帧直接放弃（见 has_highlighted_button 的说明）：
        # 深蓝灰检测器看不见被染色的按钮，OCR 会在缺一个选项的情况下
        # 作答，而正确答案可能正是那个看不见的。
        if has_highlighted_button(image):
            return None
        bands = find_option_bands(image)
        if not bands:
            return None
        result = self._engine_run(self._card(image))
        if result is None:
            return None
        lines = self._to_image_space(result)
        options = self._map_options(lines, bands)
        stem, has_progress, progress_no = self._split_stem(image, lines)
        if options is None:
            # 结构不合法（某个带漏读，或某个带挤进两行文字）。
            # 返回一个空选项的 read：门控会因选项数对不上而拒绝它，
            # 而拼接路还能借它带出来的题干。
            return DailyOcrRead(
                stem=stem,
                options=[],
                confidences=[],
                button_count=len(bands),
                has_progress=has_progress,
                progress_no=progress_no,
            )
        return DailyOcrRead(
            stem=stem,
            options=[text for _, text, _ in options],
            confidences=[confidence for _, _, confidence in options],
            button_count=len(bands),
            has_progress=has_progress,
            progress_no=progress_no,
        )

    # --------------------------------------------------------- 终态页识别

    def read_terminal(self, image: Image.Image) -> TerminalRead | None:
        """一次 OCR 严格匹配规范化后的完整终态短语；排除选项区文字及无效位置证据，不发网络请求。"""
        # 终态路径只用已初始化且未预热失败的引擎，绝不在这里加载。
        # 测试替身须显式注入ready的_engine；真实引擎由CLI提前warmup。
        if self._engine is None or self.warm_error:
            return None
        raw = self._engine_run(image)
        if raw is None:
            return None
        lines = _terminal_lines(raw)

        kinds: dict[str, str] = {}
        for text, confidence, box in lines:
            # NaN/inf 置信度一律当无效 —— `float('nan') < 阈值` 是
            # False，NaN 会**穿过**置信度门，被当成有效证据。
            if not math.isfinite(confidence):
                continue
            if confidence < self.min_confidence:
                continue
            normalized = _normalize_terminal_text(text)
            if not normalized:
                continue
            kind = _TERMINAL_PHRASE_KIND.get(normalized)
            if kind is None:
                continue
            # 命中了白名单，但位置在选项带里 → 这是选项文字，不是标题。
            if _is_option_band_line(box):
                continue
            kinds.setdefault(kind, text)

        if not kinds:
            return None
        if len(kinds) > 1:
            # 成功和失败同时出现 → 无法判断是哪种终态。
            return None
        kind, evidence = next(iter(kinds.items()))
        return TerminalRead(kind=kind, evidence=evidence)

    # -------------------------------------------------------------- 拼接路

    def read_strip(self, image: Image.Image,
                   whole: DailyOcrRead | None = None) -> DailyOcrRead | None:
        """按点击检测器的按钮带裁剪文字并拼接识别；复用整图题干和题号，每带必须恰好一个选项。"""
        bands = find_option_bands(image)
        if not bands:
            return None
        # 高亮帧同样要放弃（见 has_highlighted_button 的说明）。这一路
        # 只读按钮文字区，深蓝灰检测器看不见被染色的那个按钮 ——
        # 拼出来会是缺一个选项的列表，正好是错位的来源。
        if has_highlighted_button(image):
            return None
        if whole is None:
            whole = self.read(image)
        strip = self._build_strip(image, bands)
        if strip is None:
            return None
        result = self._engine_run(strip)
        if result is None:
            return None
        lines = [(text.strip(), float(confidence))
                 for _, text, confidence in result or [] if text.strip()]
        if not lines:
            return None
        return DailyOcrRead(
            stem=whole.stem if whole is not None else "",
            options=[text for text, _ in lines],
            confidences=[confidence for _, confidence in lines],
            button_count=len(bands),
            has_progress=whole.has_progress if whole is not None else False,
            progress_no=whole.progress_no if whole is not None else None,
        )

    def decide(self, read: DailyOcrRead | None) -> OcrDecision:
        """在通用门控之外要求题号证据，避免将复活或结算按钮作为选项。"""
        decision = super().decide(read)
        if not decision.accepted:
            return decision
        if read is not None and not read.has_progress:
            return OcrDecision(
                False,
                "没读到题号行（可能不是答题卡片）",
                read,
                needs_strip=False,
            )
        return decision

    def decide_merged(self, whole: DailyOcrRead | None,
                      strip: DailyOcrRead | None) -> OcrDecision:
        """两路合并裁决。题号行只在整图路里有，所以界面类型判据按整图路算。"""
        decision = super().decide_merged(whole, strip)
        if not decision.accepted:
            return decision
        reference = whole if whole is not None else strip
        if reference is not None and not reference.has_progress:
            return OcrDecision(
                False,
                "没读到题号行（可能不是答题卡片）",
                reference,
                needs_strip=False,
            )
        return decision

    # ------------------------------------------------------------ 内部实现

    def _card(self, image: Image.Image) -> Image.Image:
        """裁出答题卡片并放大 2 倍。"""
        crop = image.crop((_CARD_LEFT, _CARD_TOP, _CARD_RIGHT, _CARD_BOTTOM))
        return crop.resize(
            (int((_CARD_RIGHT - _CARD_LEFT) * _CARD_SCALE),
             int((_CARD_BOTTOM - _CARD_TOP) * _CARD_SCALE)),
            Image.LANCZOS,
        )

    def _build_strip(self, image: Image.Image,
                     bands: list[tuple[int, int]]) -> Image.Image | None:
        """将各按钮文字区纵向拼接，并放大 2 倍供 OCR 识别。

        带与带之间留 `_STRIP_GAP` 的黑缝 —— 不留缝的话检测器会把几个
        单字当成一条竖排文字读（见 `_STRIP_GAP` 的说明）。
        """
        array = np.asarray(image.convert("RGB"))
        parts = []
        for index, (top, bottom) in enumerate(bands):
            upper, lower = top + _STRIP_PAD_V, bottom - _STRIP_PAD_V
            if upper >= lower:
                return None
            if index and _STRIP_GAP:
                parts.append(np.zeros((_STRIP_GAP, _STRIP_RIGHT - _STRIP_LEFT, 3),
                                      dtype=array.dtype))
            parts.append(array[upper:lower, _STRIP_LEFT:_STRIP_RIGHT])
        if not parts:
            return None
        strip = np.vstack(parts)
        try:
            import cv2  # noqa: PLC0415
        except ImportError:  # pragma: no cover - opencv 是硬依赖
            return None
        strip = cv2.resize(strip, None, fx=_STRIP_SCALE, fy=_STRIP_SCALE,
                           interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(strip)

    def _map_options(self, lines, bands: list[tuple[int, int]]):
        """按文字中心归位到按钮带，输出从上到下的选项；每带必须恰好一个有效选项。"""
        # 先把每行文字归到"所在的按钮带"。
        per_band: dict[int, list[tuple[int, str, float]]] = {
            index: [] for index in range(len(bands))
        }
        for center_y, x0, x1, _y0, _y1, text, confidence in lines:
            if not text.strip():
                continue
            # 横向必须落在按钮内 —— 题干、题号行都在按钮区之外。
            if x0 < _BUTTON_LEFT - _X_TOLERANCE or x1 > _BUTTON_RIGHT + _X_TOLERANCE:
                continue
            # 纵向必须**落在某个按钮带之内**（不是"离得最近的那个"）。
            #
            # 用"最近"来判断会有个真实的漏洞：题干行离第一个按钮带
            # 中心只有几十像素，而容差是按钮高度的一半（约 22~24），
            # 一旦题干更长更靠近按钮带，就会被当成第一个选项。
            # 按"必须落在带内"判断，题干永远进不了选项。
            best = None
            for index, (top, bottom) in enumerate(bands):
                if top - _BAND_EDGE_TOLERANCE <= center_y <= bottom + _BAND_EDGE_TOLERANCE:
                    best = index
                    break
            if best is None:
                continue
            per_band[best].append((center_y, text.strip(), confidence))

        # 一一对应才成立。
        if any(len(items) != 1 for items in per_band.values()):
            return None
        options = [items[0] for items in per_band.values()]
        options.sort()
        return options

    def _to_image_space(self, result):
        """把卡片上的 OCR 行坐标换算回原图坐标。"""
        lines = []
        for box, text, confidence in result or []:
            if not text.strip():
                continue
            center_y = sum(point[1] for point in box) / len(box) / _CARD_SCALE + _CARD_TOP
            x0 = min(point[0] for point in box) / _CARD_SCALE + _CARD_LEFT
            x1 = max(point[0] for point in box) / _CARD_SCALE + _CARD_LEFT
            y0 = min(point[1] for point in box) / _CARD_SCALE + _CARD_TOP
            y1 = max(point[1] for point in box) / _CARD_SCALE + _CARD_TOP
            lines.append((center_y, x0, x1, y0, y1, text.strip(), float(confidence)))
        return lines

    def _split_stem(self, image: Image.Image, lines):
        """从卡片 OCR 结果里分出题干、题号行。"""
        parts: list[TextSpan] = []
        spans: list[TextSpan] = []
        has_progress = False
        progress_no = None
        for center_y, x0, x1, y0, y1, text, _confidence in lines:
            span = TextSpan(x0=x0, x1=x1, y0=y0, y1=y1, text=text)
            spans.append(span)
            match = _RE_PROGRESS_NO.search(text)
            if match:
                has_progress = True
                progress_no = int(match.group(1))
                continue
            if _RE_PROGRESS_LAST.search(text):
                # 有题号证据，但读不出编号 —— **不猜**，编号留 None。
                has_progress = True
                continue
            if _is_noise(text):
                continue
            # 题号行之上、按钮区之下的是题干。按钮区之下还可能有
            # "这题有问题！"之类的装饰，但卡片裁剪到 545 已经排除了。
            if center_y < _BUTTON_TOP:
                parts.append(span)
        # 题干按画面上的阅读顺序拼，填空横线补在对应位置。
        stem = compose_stem(
            parts,
            find_blank_lines(image, (_CARD_TOP, _BUTTON_TOP), spans),
        )
        return stem, has_progress, progress_no


# 纯文本路径的 prompt。
#
# **这里用 % 格式化而不是 .format()**：JSON 里的 {"answer_index": ...}
# 会被 str.format 当成占位符而抛 KeyError。选项文字里也可能出现 { }，
# 用 %s 占位同样能安全代入。
TEXT_PROMPT = """回答下面这道选择题。只输出一个 JSON，不要解释：
{"answer_index": 正确选项的编号，从 0 开始}

题目：%s

选项：
%s
"""
