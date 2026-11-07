"""每日一答选项定位：检测深蓝灰外轮廓，坐标与 OCR 按钮带保持一致。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..button_contours import find_contour_bands
from ..errors import WqError
from ..models import Rect

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

# 按钮外轮廓的最小宽度，排除文字和小块装饰。
_MIN_BUTTON_WIDTH = 200

_MIN_BAND_HEIGHT = 35
_MAX_BAND_HEIGHT = 80

# 一局至少要有几个选项才算答题界面
_MIN_OPTIONS = 2
_MAX_OPTIONS = 4

_BUTTON_HEIGHT = 45


def find_blue_gray_bands(arr: np.ndarray, top: int, bottom: int) -> list[tuple[int, int]]:
    """从深蓝灰色块的外轮廓取按钮区间，不受内部白字孔洞影响。"""
    start = max(0, min(top, arr.shape[0]))
    end = max(start, min(bottom, arr.shape[0]))
    region = arr[start:end, _BUTTON_LEFT:_BUTTON_RIGHT].astype(np.int16)
    if region.size == 0:
        return []
    R, G, B = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    bright = (R + G + B) / 3
    fill = (
        (bright > _GRAY_BRIGHT_MIN)
        & (bright < _GRAY_BRIGHT_MAX)
        & (B > _GRAY_B_MIN)
        & (B > R + _GRAY_B_OVER_R)
    )
    return find_contour_bands(
        fill, min_width=_MIN_BUTTON_WIDTH,
        min_height=_MIN_BAND_HEIGHT, max_height=_MAX_BAND_HEIGHT,
        y_offset=start,
    )


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
