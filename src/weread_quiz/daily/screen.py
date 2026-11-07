"""每日一答像素判据：题号带、题干渲染、答后揭示态及同题校验。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .buttons import (
    _BUTTON_LEFT,
    _BUTTON_RIGHT,
    _OPTION_BOTTOM,
    _OPTION_TOP,
    find_option_buttons,
)
from ..errors import WqError

# 题干所在的纵向区间。Daily 的题干在 y200..300，区别于 PK 的 y250..345。
_STEM_BAND = (200, 300)

# 二值化阈值：只把足够亮的像素算作"文字"
_STEM_WHITE = 180

_STEM_CHANGE_THRESHOLD = 500

_BUTTON_HEIGHT = 45


_STEM_TEXT_BAND = (225, 290)

_STEM_TEXT_MIN_PIXELS = 100


def stem_text_pixels(image: Image.Image) -> int:
    """题干正文区（不含题号行）的白色像素数。"""
    arr = np.asarray(image.convert("L"))
    top, bottom = _STEM_TEXT_BAND
    band = arr[top : min(bottom, arr.shape[0]), _BUTTON_LEFT:_BUTTON_RIGHT]
    if band.size == 0:
        return 0
    return int((band >= _STEM_WHITE).sum())


def has_rendered_stem(image: Image.Image) -> bool:
    """检查题干正文区是否有足够文字像素，排除空白但帧差稳定的半渲染画面。"""
    return stem_text_pixels(image) >= _STEM_TEXT_MIN_PIXELS


def stem_mask(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("L"))
    top, bottom = _STEM_BAND
    band = arr[top : min(bottom, arr.shape[0]), _BUTTON_LEFT:_BUTTON_RIGHT]
    return band >= _STEM_WHITE


def stem_change_count(before: Image.Image, after: Image.Image) -> int:
    mask_a = stem_mask(before)
    mask_b = stem_mask(after)
    if mask_a.shape != mask_b.shape:
        return max(mask_a.size, mask_b.size)
    return int(np.count_nonzero(mask_a != mask_b))


def is_new_question(before: Image.Image, after: Image.Image) -> bool:
    return stem_change_count(before, after) > _STEM_CHANGE_THRESHOLD


def is_question_stable(before: Image.Image, after: Image.Image) -> bool:
    """相邻候选帧的题干、按钮布局和选项文字是否稳定。"""
    if before.size != after.size or is_new_question(before, after):
        return False
    try:
        previous = find_option_buttons(before)
        current = find_option_buttons(after)
    except WqError:
        return False
    return len(previous) == len(current) and all(
        abs(a.center_y - b.center_y) <= 2
        and _same_option_text(before, a, after, b)
        for a, b in zip(previous, current)
    )


def is_question_screen(image: Image.Image) -> bool:
    """是不是"正在答题"的界面。"""
    if not has_progress_line(image):
        return False
    try:
        return len(find_option_buttons(image)) >= 2
    except WqError:
        return False


# 答后高亮的颜色判据。daily 的高亮是**青色**（正确项，RGB 约 13,193,255）
# 或**红色**（错误项），不是 PK 的青绿色 —— PK 的绿色判据在 daily 上
# 会同时误报（入口页）和漏报（真实揭示帧），见设计文档方案 1。
_HIGHLIGHT_BLUE_MIN = 150      # B > 150
_HIGHLIGHT_BLUE_OVER_R = 30    # B > R + 30
_HIGHLIGHT_RED_MIN = 130       # R > 130
_HIGHLIGHT_RED_OVER_GB = 30    # R > G + 30 且 R > B + 30
_HIGHLIGHT_ROW_PIXELS = 100    # 一行里超过这么多彩色像素才算"有按钮被染色"
_HIGHLIGHT_MIN_ROWS = 30       # 至少这么多行


def has_answer_reveal(image: Image.Image) -> bool:
    """检查选项区的大面积青色或红色高亮；揭示态不能作为下一题返回。"""
    arr = np.asarray(image.convert("RGB"))
    height = arr.shape[0]
    top, bottom = _OPTION_TOP, min(_OPTION_BOTTOM, height)
    if bottom <= top:
        return False
    region = arr[top:bottom, _BUTTON_LEFT:_BUTTON_RIGHT]
    R = region[:, :, 0].astype(int)
    G = region[:, :, 1].astype(int)
    B = region[:, :, 2].astype(int)
    bright = (R + G + B) / 3
    blue = (B > _HIGHLIGHT_BLUE_MIN) & (B > R + _HIGHLIGHT_BLUE_OVER_R) & (bright > 110) & (bright < 220)
    red = (R > _HIGHLIGHT_RED_MIN) & (R > G + _HIGHLIGHT_RED_OVER_GB) & (R > B + _HIGHLIGHT_RED_OVER_GB)
    highlighted = blue | red
    return bool((highlighted.sum(axis=1) > _HIGHLIGHT_ROW_PIXELS).sum() >= _HIGHLIGHT_MIN_ROWS)


_PROGRESS_BAND = (200, 211)
_PROGRESS_WHITE = 180
_PROGRESS_MIN_PIXELS = 100


def has_progress_line(image: Image.Image) -> bool:
    """按题号带亮色像素判断是否有题号内容；这是几何预筛，不是 OCR 语义确认。"""
    arr = np.asarray(image.convert("L"))
    top, bottom = _PROGRESS_BAND
    band = arr[top : min(bottom, arr.shape[0]), _BUTTON_LEFT:_BUTTON_RIGHT]
    if band.size == 0:
        return False
    return int((band >= _PROGRESS_WHITE).sum()) >= _PROGRESS_MIN_PIXELS


# 点击前复核时，按钮内文字掩码允许的差异像素数。
#
_SAME_TEXT_TOLERANCE = 8
# 点击身份校验独立于动画稳定阈值，题干和题号的小幅改变也不能放行。
_SAME_STEM_TOLERANCE = 8


def _same_option_text(before, previous, after, current) -> bool:
    """稳定门控和点击复核共用按钮内文字掩码，动画中文字变化不能放行。"""
    masks = []
    for image, box in ((before, previous), (after, current)):
        interior = image.crop((box.left + 8, box.top + 6,
                               box.right - 8, box.bottom - 6))
        masks.append(np.asarray(interior.convert("L")) >= _STEM_WHITE)
    return (masks[0].shape == masks[1].shape
            and int(np.count_nonzero(masks[0] != masks[1])) <= _SAME_TEXT_TOLERANCE)


def is_same_question(before: Image.Image, after: Image.Image) -> bool:
    """点击前的身份校验：题干、按钮布局、按钮内文字都没变。"""
    if not has_progress_line(before) or not has_progress_line(after):
        return False
    if not is_question_stable(before, after):
        return False
    if stem_change_count(before, after) > _SAME_STEM_TOLERANCE:
        return False
    try:
        previous = find_option_buttons(before)
        current = find_option_buttons(after)
    except WqError:
        return False
    for a, b in zip(previous, current):
        if any(abs(getattr(a, name) - getattr(b, name)) > 2
               for name in ("left", "top", "width", "height")):
            return False
    return True
