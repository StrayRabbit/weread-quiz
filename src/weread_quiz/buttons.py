"""PK 选项按钮定位：检测青色外轮廓，返回从上到下排列的原图矩形。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .button_contours import find_contour_bands
from .errors import WqError
from .models import Rect

# 青色判定：蓝色通道明显高于红色，且整体够亮
_CYAN_MIN_BLUE = 90
_CYAN_MIN_GREEN = 70
_CYAN_MIN_BLUE_OVER_RED = 20

# 按钮外轮廓的宽高范围；排除文字、装饰线和卡片边框。
_MIN_BUTTON_WIDTH = 200
_MIN_BAND_HEIGHT = 30
_MAX_BAND_HEIGHT = 80

_OPTION_TOP = 340
_OPTION_BOTTOM = 700

_BUTTON_LEFT = 60
_BUTTON_WIDTH = 310

# 一局至少要有几个选项才算答题界面
_MIN_OPTIONS = 2
# 上限：超过这个数多半是误检（卡片边框被当成按钮）
_MAX_OPTIONS = 8


def _cyan_mask(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.int16)
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (
        (blue - red > _CYAN_MIN_BLUE_OVER_RED)
        & (blue > _CYAN_MIN_BLUE)
        & (green > _CYAN_MIN_GREEN)
    )


def cyan_row_counts(image: Image.Image, top: int = 0,
                    bottom: int | None = None) -> np.ndarray:
    """每一行有多少青色像素（与 find_cyan_bands 用同一套判据）。

    行号从 top 开始算。别的模块要判断「某个横带上有没有一整行青色」
    时用它，免得各写一套青色阈值。
    """
    height = image.size[1]
    end = height if bottom is None else min(bottom, height)
    start = max(0, min(top, end))
    if end <= start:
        return np.zeros(0, dtype=int)
    # 只把搜索区间转成数组。convert+astype 整图是这一步的主要开销。
    crop = image.crop((0, start, image.size[0], end))
    return _cyan_mask(np.asarray(crop.convert("RGB"))).sum(axis=1)


def find_cyan_bands(image: Image.Image, top: int = 0, bottom: int | None = None) -> list[tuple[int, int]]:
    """找出青色按钮外轮廓，返回 [(y_top, y_bottom), ...]。"""
    height = image.size[1]
    end = height if bottom is None else max(0, min(bottom, height))
    start = max(0, min(top, end))
    right = min(_BUTTON_LEFT + _BUTTON_WIDTH, image.width)
    if end <= start or right <= _BUTTON_LEFT:
        return []
    # 卡片外围也是青色；排除两侧卡框，避免其外轮廓遮蔽内部按钮。
    crop = image.crop((_BUTTON_LEFT, start, right, end))
    return find_contour_bands(
        _cyan_mask(np.asarray(crop.convert("RGB"))), min_width=_MIN_BUTTON_WIDTH,
        min_height=_MIN_BAND_HEIGHT, max_height=_MAX_BAND_HEIGHT,
        y_offset=start,
    )


def option_bands(image: Image.Image) -> list[tuple[int, int]]:
    """选项区间内的青色边框带，只做过滤，不做数量或合法性判断。"""
    return find_cyan_bands(image, top=_OPTION_TOP, bottom=_OPTION_BOTTOM)


def find_option_buttons(image: Image.Image) -> list[Rect]:
    """定位选项按钮，返回按从上到下排序的矩形。"""
    height = image.size[1]
    bands = option_bands(image)
    centers = [(top + bottom) // 2 for top, bottom in bands]

    if not centers:
        raise WqError(
            "没检测到选项按钮边框。可能不在答题界面，或者界面改版了 —— "
            "对照 data/runs/ 里最新的截图确认"
        )
    if len(centers) < _MIN_OPTIONS:
        raise WqError(
            f"只检测到 {len(centers)} 个选项按钮，至少要 {_MIN_OPTIONS} 个才像答题界面"
        )
    if len(centers) > _MAX_OPTIONS:
        raise WqError(
            f"检测到 {len(centers)} 个按钮，超过合理上限 {_MAX_OPTIONS} —— "
            "多半是卡片边框被误当成按钮，界面可能改版了"
        )

    boxes = []
    for center in centers:
        center = min(center, height - _BUTTON_HEIGHT // 2)
        top = center - _BUTTON_HEIGHT // 2
        boxes.append(
            Rect(left=_BUTTON_LEFT, top=top, width=_BUTTON_WIDTH, height=_BUTTON_HEIGHT)
        )
    return boxes


_BUTTON_HEIGHT = 44


def pick_button(boxes: list[Rect], index: int) -> Rect:
    """按下标取按钮。下标越界抛异常，不返回 None。"""
    if not (0 <= index < len(boxes)):
        raise WqError(f"选项下标 {index} 越界（共 {len(boxes)} 个选项）")
    return boxes[index]
