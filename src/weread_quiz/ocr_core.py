"""共享 OCR 引擎、置信度门控与两路裁决；各模式分别定义几何和阈值。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# 置信度门控阈值。低于它就回退图片路径。
#
# **默认值是 PK 的值（0.7）**，每日一答有自己的阈值
# （`daily_ocr.DEFAULT_MIN_CONFIDENCE`）—— 两种界面的字体和字号不同，
# 置信度分布也不同，混用一个阈值会有一边过松或过严。
DEFAULT_MIN_CONFIDENCE = 0.7


# 界面文字均为正向横排，禁用 180° 角度分类器。
_ENGINE_KWARGS = {"use_cls": False}

# RapidOCR 自带 INFO 级日志（引擎初始化、模型文件检查）会刷屏，
# 把它压到 error —— 真正的错误仍然会打出来。
#
# **只能走配置项**：RapidOCR.__init__ 会用配置里的级别覆盖 logger，
# 事后调 logging 没用。
_ENGINE_PARAMS = {"Global.log_level": "error"}


def build_rapid_ocr():
    """建一个 RapidOCR 引擎（日志级别已压到 error）。"""
    from rapidocr import RapidOCR  # noqa: PLC0415 - 缺依赖时由调用方降级
    try:
        return RapidOCR(params=_ENGINE_PARAMS)
    except TypeError:
        # 老版本 RapidOCR 不吃 params 参数。
        return RapidOCR()


@dataclass
class OcrRead:
    """RapidOCR 从一张截图里读出的一道题。"""

    stem: str
    options: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    # 像素检测到的按钮数。必须等于 len(options)，否则下标映射不可靠。
    button_count: int = 0
    # 读到了题号行（「第 N 题·共 M 题」/「最后一题」）。
    #
    has_progress: bool = False
    progress_text: str = field(default="", kw_only=True)
    # 仅 PK 整图读取器填充；其他模式默认未知，不参与题目类型判定。
    countdown_seconds: int | None = field(default=None, kw_only=True)

    @property
    def min_confidence(self) -> float:
        return min(self.confidences) if self.confidences else 0.0


@dataclass
class OcrDecision:
    """门控结论及原因；needs_strip 表示允许尝试拼接读取，不等于门控通过。"""

    accepted: bool
    reason: str
    read: OcrRead | None = None
    needs_strip: bool = False


def normalize_options(options: list[str]) -> list[str]:
    """比对两路选项用的归一化：去空白与标点。"""
    return [re.sub(r"[\s，,。、；;？?：:！!（）()《》\"'“”‘’]", "", o)
            for o in options]


def decide_read(read: OcrRead | None, min_confidence: float) -> OcrDecision:
    """校验选项数、按钮数和最低置信度；空读数或低置信度可试拼接，部分选项数量不符直接判否。"""
    if read is None:
        return OcrDecision(False, "OCR 未读出题干或选项", needs_strip=True)
    if len(read.options) == 0:
        # **一个选项都没读出** —— 这不是"漏了一个"，是"全漏了"。
        #
        # 第二路仍值得跑：daily 侧用空选项表达"结构不合法"，靠拼接路
        # 带出来的题干（见 daily_ocr.read 的文档）。
        return OcrDecision(
            False,
            "OCR 未读出任何选项",
            read,
            needs_strip=True,
        )
    if len(read.options) != read.button_count:
        return OcrDecision(
            False,
            f"选项数不符（OCR {len(read.options)} vs 按钮 {read.button_count}）",
            read,
            needs_strip=False,
        )
    if read.min_confidence < min_confidence:
        return OcrDecision(
            False,
            f"置信度 {read.min_confidence:.2f} < 阈值 {min_confidence:.2f}"
            f"（选项 {read.options}）",
            read,
            needs_strip=True,
        )
    return OcrDecision(True, "", read)


def structurally_ok(read: OcrRead | None, min_confidence: float) -> bool:
    """这条路的结果能不能直接用：行数等于按钮数，且置信度达标。"""
    if read is None:
        return False
    if len(read.options) != read.button_count:
        return False
    return read.min_confidence >= min_confidence


def decide_merged(whole: OcrRead | None, strip: OcrRead | None,
                  min_confidence: float) -> OcrDecision:
    """合并整图和拼接读数：按合法性、选项一致性与置信度裁决，保留判否原因。"""
    whole_ok = structurally_ok(whole, min_confidence)
    strip_ok = structurally_ok(strip, min_confidence)

    if whole_ok and strip_ok:
        if normalize_options(whole.options) == normalize_options(strip.options):
            return OcrDecision(True, "", whole)
        if whole.min_confidence > strip.min_confidence:
            return OcrDecision(True, "两路结果不一致，取整图路（置信度更高）", whole)
        if strip.min_confidence > whole.min_confidence:
            return OcrDecision(True, "两路结果不一致，取拼接路（置信度更高）", strip)
        return OcrDecision(True, "两路结果不一致且置信度持平，取整图路", whole)
    if whole_ok:
        if strip is None:
            # **第二路根本没跑** —— 按需触发时整图路已经合法就不会再跑它。
            # 这里**不能**说"拼接路结构不合法"：那是在陈述一件没发生的事。
            #
            # 整图路合法时 reason 就是空串 —— 调用方（pk_hybrid）会把
            # 空串显示成"整图路"，那才是事实。
            return OcrDecision(True, "", whole)
        return OcrDecision(True, "拼接路结构不合法（相邻选项粘连），取整图路", whole)
    if strip_ok:
        return OcrDecision(True, "整图路结构不合法（单字选项漏读），取拼接路", strip)

    # 都不合法：报出各自的原因，便于按 run.log 定位是哪类失败。
    #
    # 整图路一定跑过（它是第一路），所以它的失败原因照实报；拼接路
    # 只有真的跑了才有失败原因。
    reasons = []
    whole_sub = decide_read(whole, min_confidence)
    if whole_sub.reason:
        reasons.append(f"整图{whole_sub.reason}")
    if strip is None:
        reasons.append("拼接路未运行")
    else:
        strip_sub = decide_read(strip, min_confidence)
        if strip_sub.reason:
            reasons.append(f"拼接{strip_sub.reason}")
    return OcrDecision(
        False,
        "；".join(reasons) or "两路都读不出可用结果",
        whole,
        needs_strip=(
            strip is None
            and (
                whole is None
                # 整图路读出了但只是置信度不够 / 一个选项都没读出
                # -> 值得跑（4/5）。
                or decide_read(whole, min_confidence).needs_strip
            )
        ),
    )


class RapidOcrEngine:
    """RapidOCR 引擎的惰性单例。"""

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence
        self._engine = None
        self._load_error: str = ""
        self._warm_error: str = ""

    @property
    def load_error(self) -> str:
        """加载失败的原因；成功或还没加载时为空串。"""
        return self._load_error

    @property
    def warm_error(self) -> str:
        """预热失败的原因；预热成功或没预热时为空串。"""
        return self._warm_error

    def warmup(self, sample: Image.Image | None = None) -> bool:
        """预加载模型并预热推理。成功返回 True；失败记录原因并返回 False。"""
        if self._engine is not None:
            return True
        try:
            self._engine = build_rapid_ocr()
        except ImportError as exc:  # noqa: BLE001 - 缺依赖/装不上都要降级
            self._load_error = f"rapidocr 不可用：{type(exc).__name__}: {exc}"
            return False
        except Exception as exc:  # noqa: BLE001 - 模型文件损坏等
            self._load_error = f"RapidOCR 初始化失败：{type(exc).__name__}: {exc}"
            return False
        if sample is not None:
            self._warm_inference(sample)
        return True

    def _warm_inference(self, image: Image.Image) -> None:
        """跑一次推理把 onnx 的预热成本吃掉。失败只记录不抛出。"""
        try:
            self._engine(np.asarray(image.convert("RGB")), **_ENGINE_KWARGS)
        except Exception as exc:  # noqa: BLE001 - 预热失败不影响真实识别
            self._warm_error = f"{type(exc).__name__}: {exc}"

    def _engine_run(self, image: Image.Image):
        """跑一次引擎。识别失败等价于"这次没读出来"，返回 None。"""
        if not self.warmup():
            return None
        try:
            output = self._engine(np.asarray(image.convert("RGB")), **_ENGINE_KWARGS)
            if output.boxes is None or output.txts is None or output.scores is None:
                return None
            # RapidOCR 3.x 返回结果对象；下游仍使用 (box, text, score) 行。
            return list(zip(output.boxes, output.txts, output.scores, strict=True)) or None
        except Exception:  # noqa: BLE001
            return None

    def decide(self, read: OcrRead | None) -> OcrDecision:
        """单路门控（见模块级 `decide_read`）。"""
        return decide_read(read, self.min_confidence)

    def decide_merged(self, whole: OcrRead | None,
                      strip: OcrRead | None) -> OcrDecision:
        """合并整图和拼接读数：按合法性、选项一致性与置信度裁决，保留判否原因。"""
        return decide_merged(whole, strip, self.min_confidence)
