"""填空横线：把画面里画出来的下划线补回题干。

小程序的填空题把横线**画成图形**，OCR 读不到它 —— 题干会变成没有
空位的「女红男」，模型只能靠猜「这里本来有个空」。这里用像素把横线
找出来，在题干里补成「____」，让模型知道这是个填空题。

**横线不是靠长度认出来的**：汉字「一」「二」「三」也有横笔，长度跟
填空线差不多。所以一条候选横线要同时满足三条：

1. 附近没有文字框压着它 —— 横笔永远在自己的字框里（规则 1）；
2. 上下若干行内没有别的亮像素 —— 独立的线上下都干净，字的横笔
   上下还有别的笔画（规则 2）；
3. 只有一两像素厚、长度够 —— 边框、图标比这厚。

三条都不满足就当没有横线：宁可题干少一个信息，也不能把字改错。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# 补进题干的占位符。中文填空题的通用写法。
PLACEHOLDER = "____"

# 横线至少这么长（像素）才算数。长度只是初筛，真正的判据是下面三条。
_MIN_LENGTH = 16
# 横线只有一两像素厚。更厚的白条是边框或图标，不是填空。
_MAX_HEIGHT = 3
# 横线与文字框的纵向距离小于它就算「挨着」，用来识别横笔。
_NEAR_Y = 12
# 横线上下各看这么多行，判断它是不是独立的一条线。
_GAP_ROWS = 8
# 横向多看几个像素，避开抗锯齿边缘。
_GAP_TOL = 2
# 上下合计超过这么多亮像素，就不是独立的线。
_MAX_INK = 2
# 横线与文字框横向重叠这么多**比例**，才算「压在字上」—— 汉字的横笔必然
# **整个**在自己的框里。
#
# **不能用绝对像素数**：OCR 的文字框比字形宽几个像素，紧跟在字后面的
# 填空线会蹭上那点余量被误伤（实测 2 倍识别下「襟江带 __」的横线就是
# 这样丢的：框宽出 5 像素，横线被当成了字的横笔）。
_OVERLAP_RATIO = 0.5
# 亮像素：接近白色，且不是高饱和的彩色（卡片边框是青色）。
_LIGHT_MIN = 170
_LIGHT_SPREAD = 70
# 分行的容差：文字中心相差不超过半个字高就算同一行。
_LINE_TOLERANCE_RATIO = 0.5
_MIN_LINE_TOLERANCE = 4.0


@dataclass(frozen=True)
class TextSpan:
    """图上的一段文字。"""

    x0: float
    x1: float
    y0: float
    y1: float
    text: str

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class BlankLine:
    """一条填空横线。"""

    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class _Line:
    """一行文字。"""

    spans: list[TextSpan]

    @property
    def top(self) -> float:
        return min(span.y0 for span in self.spans)

    @property
    def bottom(self) -> float:
        return max(span.y1 for span in self.spans)

    @property
    def center(self) -> float:
        return sum(span.center_y for span in self.spans) / len(self.spans)

    def distance_to(self, center_y: float) -> float:
        """到某一行的纵向距离；在行内是 0。"""
        if self.top <= center_y <= self.bottom:
            return 0.0
        return self.top - center_y if center_y < self.top else center_y - self.bottom


def find_blank_lines(
    image: Image.Image,
    region: tuple[int, int],
    spans=(),
    *,
    min_length: int = _MIN_LENGTH,
    max_height: int = _MAX_HEIGHT,
    near_y: int = _NEAR_Y,
    allow_embedded: bool = False,
) -> list[BlankLine]:
    """在 region=(上沿, 下沿) 里找独立的水平亮线。

    spans 是图上已知的文字段，用来排除「一」「二」这类横笔。
    allow_embedded 仅用于分段补读：允许整行框内、行底部且上下干净的填空线。
    """
    top = max(0, int(region[0]))
    bottom = min(image.size[1], int(region[1]))
    if bottom - top < 4:
        return []
    crop = image.crop((0, top, image.size[0], bottom))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    array = np.asarray(crop)
    if array.size == 0:
        return []
    if array.ndim != 3 or array.shape[2] < 3:
        return []
    # 不转成 int64 —— uint8 就够用，high-low 也不会下溢（high >= low）。
    # 逐通道取 min/max 比 axis=2 的归约快一个数量级（每题都要跑）。
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    low = np.minimum(np.minimum(red, green), blue)
    high = np.maximum(np.maximum(red, green), blue)
    light = (low > _LIGHT_MIN) & ((high - low) < _LIGHT_SPREAD)

    segments = _merge_runs(_row_runs(light, min_length))
    known = list(spans)
    result: list[BlankLine] = []
    for first_row, last_row, x0, x1 in segments:
        if last_row - first_row + 1 > max_height:
            continue
        y0 = top + first_row
        y1 = top + last_row
        center = (y0 + y1) / 2
        if _touches_text(known, center, x0, x1, near_y):
            if not allow_embedded or not _is_embedded_blank(
                light, top, known, first_row, last_row, x0, x1,
            ):
                continue
        if _has_ink_above_below(light, first_row, last_row, x0, x1):
            continue
        result.append(BlankLine(x0=x0, x1=x1, y0=y0, y1=y1))
    return result


def compose_stem(spans, blanks, placeholder: str = PLACEHOLDER) -> str:
    """把文字段和横线按位置合成题干。

    横线按它在图上的横向位置插到对应的文字之间：横线在文字左边就
    补在前面，在中间就补在中间。**没有文字的行不补** —— 那里多半
    是装饰而不是填空。
    """
    if not spans:
        return ""
    lines = _group_lines(spans)
    tolerance = _line_tolerance(spans)
    # 每条横线归到最近的一行；归不进去的（附近没有文字）就丢掉。
    assigned: dict[int, list[BlankLine]] = {}
    for blank in blanks:
        index = _line_index(blank, lines, tolerance)
        if index is not None:
            assigned.setdefault(index, []).append(blank)
    parts = []
    for index, line in enumerate(lines):
        items = [(span.x0, 0, span.text) for span in line.spans]
        for blank in assigned.get(index, []):
            items.append((blank.x0, 1, placeholder))
        items.sort(key=lambda item: (item[0], item[1]))
        parts.append("".join(text for _, _, text in items))
    return "".join(parts)


def _row_runs(light: np.ndarray, min_length: int) -> list[tuple[int, int, int, int]]:
    """逐行找出够长的连续亮段，返回 (行号, 行号, 左, 右)。

    只扫有亮像素的行 —— 题干带里有一半是空的，省下来的都是实打实的
    每题预算。
    """
    runs = []
    for index in np.flatnonzero(light.any(axis=1)):
        columns = np.flatnonzero(light[index])
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 4) + 1)
        for group in groups:
            if len(group) >= min_length:
                runs.append((int(index), int(index), int(group[0]), int(group[-1])))
    return runs


def _merge_runs(runs) -> list[tuple[int, int, int, int]]:
    """纵向合并：同一根线可能因抗锯齿占两行。"""
    merged: list[tuple[int, int, int, int]] = []
    for run in runs:
        if (merged
                and run[0] - merged[-1][1] <= 1
                and abs(run[2] - merged[-1][2]) <= 4
                and abs(run[3] - merged[-1][3]) <= 4):
            merged[-1] = (merged[-1][0], run[1], merged[-1][2], merged[-1][3])
        else:
            merged.append(run)
    return merged


def _touches_text(spans, center: float, x0: float, x1: float, near_y: int) -> bool:
    """附近有文字框且横向压着它 —— 这是字的横笔，不是填空线。"""
    length = max(1, x1 - x0)
    for span in spans:
        if abs(span.center_y - center) > near_y:
            continue
        overlap = min(x1, span.x1) - max(x0, span.x0)
        if overlap >= length * _OVERLAP_RATIO:
            return True
    return False


def _is_embedded_blank(light, top, spans, first_row, last_row, x0, x1) -> bool:
    """整行框内的填空线应在行底部，且整行高度内没有其他笔画穿过。"""
    center = top + (first_row + last_row) / 2
    for span in spans:
        if not (span.x0 + 4 <= x0 and x1 + 4 <= span.x1
                and span.y0 <= center <= span.y1):
            continue
        if center < span.y0 + (span.y1 - span.y0) * 0.65:
            continue
        left, right = x0 + _GAP_TOL, x1 - _GAP_TOL + 1
        above = light[max(0, int(span.y0) - top):first_row, left:right]
        below = light[last_row + 1:min(light.shape[0], int(np.ceil(span.y1)) - top),
                      left:right]
        if int(above.sum()) + int(below.sum()) <= _MAX_INK:
            return True
    return False


def _has_ink_above_below(light: np.ndarray, first_row: int, last_row: int,
                         x0: float, x1: float) -> bool:
    """横线上下还有亮像素 —— 它是更大图形的一部分，不是独立的线。"""
    left = max(0, int(x0) - _GAP_TOL)
    right = min(light.shape[1], int(x1) + _GAP_TOL + 1)
    count = 0
    for row in range(max(0, first_row - _GAP_ROWS), first_row):
        count += int(light[row, left:right].sum())
    for row in range(last_row + 1, min(light.shape[0], last_row + 1 + _GAP_ROWS)):
        count += int(light[row, left:right].sum())
    return count > _MAX_INK


def _group_lines(spans) -> list[_Line]:
    """按纵向把文字段分成若干行。

    判据是**文字中心**相差不超过半个字高 —— 比「上下不重叠」稳，
    同一行里字号略有差别时也不会拆成两行。
    """
    ordered = sorted(spans, key=lambda span: (span.center_y, span.x0))
    tolerance = _line_tolerance(spans)
    lines: list[_Line] = []
    for span in ordered:
        if lines and abs(span.center_y - lines[-1].center) <= tolerance:
            lines[-1].spans.append(span)
        else:
            lines.append(_Line([span]))
    for line in lines:
        line.spans.sort(key=lambda span: span.x0)
    return lines


def _line_tolerance(spans) -> float:
    """分行容差：半个字高，至少 4 像素。"""
    heights = sorted(span.y1 - span.y0 for span in spans)
    if not heights:
        return _MIN_LINE_TOLERANCE
    median = heights[len(heights) // 2]
    return max(_MIN_LINE_TOLERANCE, _LINE_TOLERANCE_RATIO * median)


def _line_index(blank: BlankLine, lines: list[_Line], tolerance: float) -> int | None:
    """横线属于哪一行（最近的一行）；太远就不属于任何行。"""
    best_index = None
    best_distance = None
    for index, line in enumerate(lines):
        distance = line.distance_to(blank.center_y)
        if best_distance is None or distance < best_distance:
            best_index = index
            best_distance = distance
    if best_index is None or best_distance > tolerance:
        return None
    return best_index
