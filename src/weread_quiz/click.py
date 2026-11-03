"""点击封装。"""

from __future__ import annotations

import time

import pyautogui

from . import abort
from .dpi import box_center_to_screen
from .models import Rect

# 紧急中止：把鼠标猛地推到屏幕左上角时，pyautogui 抛 FailSafeException。
# pk.py / daily.py 捕获它并停下。
#
# Esc 是主入口（见 abort.py），这个角落 failsafe 保留作兜底 ——
# 万一键盘钩子装不上，至少还有一条中止路径。
pyautogui.FAILSAFE = True


def click_box(box: Rect, region: Rect, scale: float, pause: float = 0.3) -> None:
    """点击 region 内 box 的中心。坐标经 DPI 换算。"""
    # 按了 Esc 就不再点 —— 中止必须赶在这一次点击之前生效。
    abort.check()
    x, y = box_center_to_screen(region, box, scale)
    pyautogui.click(x, y)
    time.sleep(pause)
