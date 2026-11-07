"""按钮颜色掩码的外轮廓过滤，供 daily 和 PK 共用。"""
from __future__ import annotations

import cv2
import numpy as np


def find_contour_bands(mask: np.ndarray, *, min_width: int,
                       min_height: int, max_height: int,
                       y_offset: int = 0) -> list[tuple[int, int]]:
    """只看外轮廓的宽高，返回原图纵向区间（含下边界）。"""
    if mask.size == 0:
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    bands = []
    for contour in contours:
        _, top, width, height = cv2.boundingRect(contour)
        if width >= min_width and min_height <= height <= max_height:
            bands.append((top + y_offset, top + y_offset + height - 1))
    return sorted(bands)
