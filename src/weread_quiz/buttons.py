"""PK 选项按钮定位：按青色边框分带，返回从上到下排列的原图矩形。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .errors import WqError
from .models import Rect

# 青色判定：蓝色通道明显高于红色，且整体够亮
_CYAN_MIN_BLUE = 90
_CYAN_MIN_GREEN = 70
_CYAN_MIN_BLUE_OVER_RED = 20

# 每行至少这么多青色像素才算"这一行横穿了按钮边框"。
# 按钮宽约 310px；卡片边框只有 2~3px 宽。
_MIN_ROW_WIDTH = 100

# 连续段最小厚度。按钮高 43~50px，卡片边框和装饰线只有 1~3px。
_MIN_BAND_HEIGHT = 30

_OPTION_TOP = 340
_OPTION_BOTTOM = 700

_BUTTON_LEFT = 60
_BUTTON_WIDTH = 310

# 一局至少要有几个选项才算答题界面
_MIN_OPTIONS = 2
# 上限：超过这个数多半是误检（卡片边框被当成按钮）
_MAX_OPTIONS = 8


def find_cyan_bands(image: Image.Image, top: int = 0, bottom: int | None = None) -> list[tuple[int, int]]:
    """找出横向的青色边框条，返回 [(y_top, y_bottom), ...]。"""
    height = image.size[1]
    end = height if bottom is None else min(bottom, height)
    start = max(0, min(top, end))
    if end <= start:
        return []
    # 只把搜索区间转成数组。convert+astype 整图是这一步的主要开销。
    crop = image.crop((0, start, image.size[0], end))
    arr = np.asarray(crop.convert("RGB")).astype(int)
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cyan = (
        (blue - red > _CYAN_MIN_BLUE_OVER_RED)
        & (blue > _CYAN_MIN_BLUE)
        & (green > _CYAN_MIN_GREEN)
    )

    row_counts = cyan.sum(axis=1)
    bands: list[tuple[int, int]] = []
    begin: int | None = None
    for offset, count in enumerate(row_counts):
        y = start + offset
        if count >= _MIN_ROW_WIDTH and begin is None:
            begin = y
        elif count < _MIN_ROW_WIDTH and begin is not None:
            bands.append((begin, y - 1))
            begin = None
    if begin is not None:
        bands.append((begin, end - 1))

    return [band for band in bands if band[1] - band[0] + 1 >= _MIN_BAND_HEIGHT]


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
