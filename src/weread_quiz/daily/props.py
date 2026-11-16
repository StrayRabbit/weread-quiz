"""每日一答道具：只读底部数量和可用颜色，不发模型请求、不初始化 OCR。

判定规则来自实机截图（data/runs/daily/*/，核对脚本 tools/check_props.py）：
底部并排两个卡片，左边「跳题卡·N」右边「换题卡·N」，N 是剩余数量
（0 表示没有）。数量为 0 时按钮变暗；最后一题时跳题卡也会变暗 —— 所以
**可用性必须同时看数量和颜色**，只看数量会误用。

历史帧里换题卡**恒为 0 且暗色**，没有它的可用态样本 —— 换题分支
（点击落点、生效判据）尚未经真实界面验证。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..models import Rect

# 道具区在标定坐标（430 宽）下的位置。
#
# 截图已由 capture 归一化到 430px 宽，因此这些坐标不随显示缩放变化。
# 跳题卡在左、换题卡在右；文字紧跟在图标右侧。
_PROP_ROW_TOP = 630
_PROP_ROW_BOTTOM = 700
_PROP_BOXES = {
    "skip": Rect(44, _PROP_ROW_TOP, 163, _PROP_ROW_BOTTOM - _PROP_ROW_TOP),
    "swap": Rect(224, _PROP_ROW_TOP, 163, _PROP_ROW_BOTTOM - _PROP_ROW_TOP),
}
_PROP_LABELS = {"skip": "跳题卡", "swap": "换题卡"}

# 只识别文字带（跳过左侧图标）—— 图标是装饰性字形，会干扰文字识别。
# 实测图标区同样会随可用状态变亮变暗，但判据只用文字带（见 has_enabled_color）。
_TEXT_LEFT = 65
_TEXT_RIGHT = 8
_TEXT_TOP = 10
_TEXT_BOTTOM = 10

_MIN_CONFIDENCE = 0.85

# 繁简归一化。实测 RapidOCR 会把「跳题卡」读成「跳題卡」（繁体），
# 严格匹配会漏读 —— 归一化后再比对。
_VARIANTS = str.maketrans({"題": "题", "換": "换", "跳": "跳"})

# 可用态的亮度判据（实测标定，见 has_enabled_color）。
_BRIGHT_LEVEL = 140
_BRIGHT_MIN_PIXELS = 40

# 道具提示条（「跳题卡使用成功」）的位置和颜色。
#
# 实测（data/runs/daily/20261110-084219/015_prop_unconfirmed.png）：
# 提示条是一块近黑的圆角矩形，x=155..274、y=207..260，白字居中。
# 卡片背景是深蓝（RGB 约 7,40,79，最大通道 79），提示条近黑
# （约 1,8,16，最大通道 16）—— 同一块区域里近黑像素 4971 个，
# 没有提示条的题面帧是 0 个（转场淡出帧最多 328 个）。
#
# 它压在题号行和题干上：**出现和消失都会改变题干区像素**，所以
# 不能拿它当"题干变了"的证据，也不能在它还在的时候读题号。
_TOAST_BOX = Rect(140, 200, 150, 65)
_TOAST_DARK = 50
_TOAST_MIN_PIXELS = 1000
# 提示条是**卡面上的一块黑**，四周还是卡面的深蓝。整块画面都黑
# （转场淡出、窗口被遮）不算提示条 —— 所以还要看盒子四周。
_TOAST_RING = 12
_TOAST_MAX_RING_DARK = 0.7


@dataclass(frozen=True)
class PropState:
    """一种道具的读数。count 为 None 表示读数不可靠（不猜）。"""

    kind: str
    count: int | None
    enabled_color: bool
    # 文字带（点击落点）—— 点在读到的文字上，不点空白。
    text_box: Rect

    @property
    def available(self) -> bool:
        """数量大于 0 且按钮为亮色 —— 两个条件缺一不可。"""
        return self.count is not None and self.count > 0 and self.enabled_color


def _text_rect(box: Rect) -> Rect:
    return Rect(box.left + _TEXT_LEFT, box.top + _TEXT_TOP,
                box.width - _TEXT_LEFT - _TEXT_RIGHT,
                box.height - _TEXT_TOP - _TEXT_BOTTOM)


def has_enabled_color(image: Image.Image, kind: str) -> bool:
    """文字带里的亮像素够不够多 —— 亮色即"可用"。

    实测（data/runs/daily/*/ 与 tools/check_props.py）：跳题卡可用时文字带
    有 300+ 个亮像素（亮度上限 235~255），不可用时为 0（上限 ≤131）。
    阈值 140 落在 131 与 235 之间的空档，两态区分得很开。

    判据只取亮度上限，不绑定色相 —— 换题卡是紫色，且历史帧里换题卡
    **恒为 0 且暗色**，没有可用态样本可标定；用亮度能让两张卡共用一把尺子，
    但换题卡的亮色阈值仍属推断，等拿到真实样本再校准。
    """
    rect = _text_rect(_PROP_BOXES[kind])
    if image.width != 430 or image.height < rect.bottom or rect.width <= 0 or rect.height <= 0:
        return False
    pixels = np.asarray(image.convert("RGB"))[rect.top:rect.bottom, rect.left:rect.right]
    bright = pixels.max(axis=2)
    return int((bright >= _BRIGHT_LEVEL).sum()) >= _BRIGHT_MIN_PIXELS


def has_toast(image: Image.Image) -> bool:
    """画面上有没有道具提示条（「跳题卡使用成功」）。

    提示条盖住题号行和题干：它在的时候题号读数不稳、题干帧差
    也被污染。判据是**卡面上的一块近黑矩形**：盒子本身够黑，
    四周还得是卡面的深蓝 —— 整幅画面都黑的时候不算提示条。
    """
    rect = _TOAST_BOX
    if image.width != 430 or image.height < rect.bottom or rect.width <= 0 or rect.height <= 0:
        return False
    array = np.asarray(image.convert("RGB"))
    dark = array.max(axis=2) < _TOAST_DARK
    if int(dark[rect.top:rect.bottom, rect.left:rect.right].sum()) < _TOAST_MIN_PIXELS:
        return False
    top = max(0, rect.top - _TOAST_RING)
    left = max(0, rect.left - _TOAST_RING)
    bottom = min(array.shape[0], rect.bottom + _TOAST_RING)
    right = min(array.shape[1], rect.right + _TOAST_RING)
    ring = np.ones((bottom - top, right - left), dtype=bool)
    ring[rect.top - top:rect.bottom - top, rect.left - left:rect.right - left] = False
    count = int(ring.sum())
    if count == 0:
        return False
    return int(dark[top:bottom, left:right][ring].sum()) / count < _TOAST_MAX_RING_DARK


def read_props(image: Image.Image, ocr) -> list[PropState]:
    """复用已预热的 OCR 引擎，只识别两个已知文字带。

    不启动文本检测（那是给整图用的，慢一个数量级），因此拿不到框 ——
    点击位置用固定的文字带，识别结果只用来读数量。
    """
    runner = getattr(ocr, "_engine_run_text", None)
    if (image.width != 430 or image.height < _PROP_ROW_BOTTOM
            or getattr(ocr, "_engine", None) is None
            or getattr(ocr, "warm_error", None) or not callable(runner)):
        return []
    states = []
    for kind, box in _PROP_BOXES.items():
        rect = _text_rect(box)
        crop = image.crop((rect.left, rect.top, rect.right, rect.bottom))
        crop = crop.resize((crop.width * 2, crop.height * 2))
        try:
            lines = runner(crop) or []
        except Exception:  # noqa: BLE001 - 读不准不能消耗道具
            lines = []
        counts = []
        for text, confidence in lines:
            if confidence < _MIN_CONFIDENCE:
                continue
            normalized = str(text).strip().translate(_VARIANTS)
            match = re.fullmatch(_PROP_LABELS[kind] + r"\s*[·•.、:：]?\s*(\d{1,3})",
                                 normalized)
            if match is not None:
                counts.append(int(match[1]))
        # 多条读数互相冲突时不能挑一个 —— 读不出就是读不出。
        count = counts[0] if len(counts) == 1 else None
        states.append(PropState(kind, count, has_enabled_color(image, kind), rect))
    return states


def choose_prop(states: list[PropState], *, last_question: bool,
                allow_skip: bool = True) -> PropState | None:
    """跳题优先、换题其次 —— 优先级是**回退链**，不是"选一张就不管了"。

    * 末题只能换题（跳题卡在末题被界面禁用）。
    * allow_skip=False（题号读不出，跳题后无法确认题位）时直接跳过跳题卡，
      继续看换题卡 —— 有换题卡就用，不能因为跳题卡不可用就整条放弃。
    """
    order = ("swap",) if last_question else ("skip", "swap")
    for kind in order:
        if kind == "skip" and not allow_skip:
            continue
        for state in states:
            if state.kind == kind and state.available:
                return state
    return None
