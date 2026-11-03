"""每日一答选项定位：检测深蓝灰填充块，坐标与 OCR 按钮带保持一致。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .errors import WqError
from .models import Rect

_OPTION_TOP = 295
_OPTION_BOTTOM = 540

# 横向搜索范围：覆盖整张图可点区域。
_BUTTON_LEFT = 30
_BUTTON_RIGHT = 400

# 深蓝灰填充条件。R≈42, G≈59, B≈74 在 58 帧样本上稳定命中。
_GRAY_BRIGHT_MIN = 40
_GRAY_BRIGHT_MAX = 105
_GRAY_B_MIN = 50
_GRAY_B_OVER_R = 5

# 每行至少这么多深蓝灰像素才算"这一行横穿了按钮"。
# 按钮宽约 320px，列搜索范围 30..400 共 370px。
_MIN_ROW_WIDTH = 200

_MIN_BAND_HEIGHT = 35
_MAX_BAND_HEIGHT = 80

# 一局至少要有几个选项才算答题界面
_MIN_OPTIONS = 2
_MAX_OPTIONS = 4

_BUTTON_HEIGHT = 45


def find_blue_gray_bands(arr: np.ndarray, top: int, bottom: int) -> list[tuple[int, int]]:
    R = arr[:, :, 0].astype(int)
    G = arr[:, :, 1].astype(int)
    B = arr[:, :, 2].astype(int)
    bright = (R + G + B) / 3
    fill = (
        (bright > _GRAY_BRIGHT_MIN)
        & (bright < _GRAY_BRIGHT_MAX)
        & (B > _GRAY_B_MIN)
        & (B > R + _GRAY_B_OVER_R)
    )
    row_counts = fill[top:bottom, _BUTTON_LEFT:_BUTTON_RIGHT].sum(axis=1)
    bands: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for offset, count in enumerate(row_counts):
        y = top + offset
        if count >= _MIN_ROW_WIDTH:
            if start is None:
                start = y
            gap = 0
        else:
            if start is not None:
                gap += 1
                # 容忍 8px 内的断（按钮内花纹造成的几像素暗行）
                if gap > 8:
                    bands.append((start, y - gap))
                    start = None
    if start is not None:
        bands.append((start, bottom - 1))
    return [b for b in bands if _MIN_BAND_HEIGHT <= b[1] - b[0] + 1 <= _MAX_BAND_HEIGHT]


def find_option_buttons(image: Image.Image) -> list[Rect]:
    """定位选项按钮，返回按从上到下排序的矩形。"""
    arr = np.asarray(image.convert("RGB"))
    height = arr.shape[0]
    bands = find_blue_gray_bands(arr, _OPTION_TOP, min(_OPTION_BOTTOM, height))
    if not bands:
        raise WqError(
            "没检测到每日一答选项按钮。可能不在答题界面、"
            "倒计时已结束或界面改版 —— 对照 data/runs/ 里最新截图确认"
        )
    if len(bands) < _MIN_OPTIONS:
        raise WqError(
            f"只检测到 {len(bands)} 个选项按钮，至少要 {_MIN_OPTIONS} 个才像答题界面"
        )
    if len(bands) > _MAX_OPTIONS:
        raise WqError(
            f"检测到 {len(bands)} 个按钮，超过合理上限 {_MAX_OPTIONS} —— "
            "可能不是答题界面，或界面改版了"
        )
    boxes: list[Rect] = []
    for top, bottom in bands:
        center = (top + bottom) // 2
        center = min(center, height - _BUTTON_HEIGHT // 2)
        top_box = center - _BUTTON_HEIGHT // 2
        boxes.append(
            Rect(left=_BUTTON_LEFT, top=top_box, width=_BUTTON_RIGHT - _BUTTON_LEFT, height=_BUTTON_HEIGHT)
        )
    return boxes


def _detect_any_buttons(image: Image.Image) -> bool:
    """返回 True 表示"屏幕上还有按钮"（题面或反馈高亮）。"""
    arr = np.asarray(image.convert("RGB"))
    region = arr[_OPTION_TOP:_OPTION_BOTTOM, _BUTTON_LEFT:_BUTTON_RIGHT]
    R = region[:, :, 0].astype(int)
    G = region[:, :, 1].astype(int)
    B = region[:, :, 2].astype(int)
    bright = (R + G + B) / 3
    # 深蓝灰
    gray = (bright > 40) & (bright < 110) & (B > 50) & (B > R + 5)
    # 亮蓝紫高亮
    blue = (B > 150) & (B > R + 30) & (bright > 110) & (bright < 220)
    # 红色高亮
    red = (R > 130) & (R > G + 30) & (R > B + 30)
    # 任意一种按钮色 + 横向足够长 = 有按钮
    any_btn = gray | blue | red
    row = any_btn.sum(axis=1)
    big = (row > 100).sum()
    return big >= 30  # 至少 30 行有按钮像素 = 至少一个横向色块


def pick_button(boxes: list[Rect], index: int) -> Rect:
    if not (0 <= index < len(boxes)):
        raise WqError(f"选项下标 {index} 越界（共 {len(boxes)} 个选项）")
    return boxes[index]
