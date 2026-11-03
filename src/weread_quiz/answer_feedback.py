"""复用已有帧，按所选按钮内部颜色记录答后反馈；不参与流程决策。"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .models import Rect
from .usage import safe_log

# 采样点共 18 个。
_POINTS = 18

# 提前取帧的时点（相对点击，单位秒）：答后反馈只亮很短一瞬，要赶在渐隐之前。
#
# 揭示色要等服务端回包才出现（实测 0.2~0.5 秒），只取一帧多半还是原色。
# 0.1 秒一帧铺满固定延迟那一段，总等待时间不变（抓屏本身约 20 毫秒，
# 落在等待里）。
SAMPLE_OFFSETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# 绝对判据：多数点已经是实心反馈色。
_ABSOLUTE_MIN = int(_POINTS * 0.7) + 1

# 相对判据：多数点相对点击前明显偏向某一侧（覆盖颜色渐显到一半的情况）。
_SHIFT_MIN = int(_POINTS * 0.6) + 1

# 色相位移阈值：(b - r) 至少变化这么多才算"往蓝/青走"或"往红走"。
_SHIFT_DELTA = 25

# 位移后当前颜色必须仍然明显偏向那一侧，否则暗下来的蓝底会被误当成变红。
_SHIFT_TARGET = 25

# 末题取帧窗口里保留的帧数（留证用）。
#
# 末题的颜色只亮一瞬，抓不到时得靠这一段判断它到底出现过没有、
# 什么时候消失 —— 只留一帧看不出这些。
_WINDOW_KEEP = 12


class AnswerFeedback:
    """每次实际点击一个实例；确认或换题后停止取色，不主动抓屏。"""

    def __init__(self, mode: str) -> None:
        self._stem_box = (40, 296, 390, 345) if mode == "pk" else (40, 225, 390, 290)
        self.status = "unknown"
        self.started = False
        self._done = False
        self._reported = False
        self._diagnostic_frame: Image.Image | None = None
        self._reason = "未观察到反馈帧"
        self._color_detail = ""
        self._before_detail = ""
        self._base_br: list[int] = []
        # 颜色证据最强的那一帧（留证用）。第一帧通常还没有颜色。
        self._best_score = -1
        # 末题的取帧窗口（留证用）；非末题不保留，省内存。
        self._window: list[Image.Image] = []
        # 末题才需要提前取帧 —— 其他题沿用原有等待，不额外抓屏。
        self.last_question = False

    def start(self, before: Image.Image, box: Rect) -> None:
        if before.mode != "RGB":
            return
        # 按钮矩形不完整就放弃取色 —— 反馈是可选信息，不能带崩答题。
        if not all(hasattr(box, name) for name in ("left", "top", "width", "height")):
            return
        self._size = before.size
        self._stem = self._stem_mask(before)
        # 均匀稀疏采样按钮内部，绕过边框、圆角和中心文字/鼠标。
        self._points = [
            (box.left + int(box.width * x), box.top + int(box.height * y))
            for x in (0.15, 0.25, 0.35, 0.65, 0.75, 0.85)
            for y in (0.25, 0.5, 0.75)
        ]
        if (box.width <= 0 or box.height <= 0
                or any(not (0 <= x < before.width and 0 <= y < before.height)
                       for x, y in self._points)):
            return
        # 点击前的底色是判"相对位移"的基准。
        pixels = before.load()
        self._base_br = [pixels[x, y][2] - pixels[x, y][0] for x, y in self._points]
        self._before_color, red, blue = self._color(before)
        self._before_detail = f"点击前红点{red}/{_POINTS}、蓝点{blue}/{_POINTS}（{self._before_color}）"
        self.started = True

    def _stem_mask(self, image: Image.Image) -> np.ndarray:
        # 只转换题干局部，不转换整图；RGB 三通道均亮才算白字。
        return np.all(np.asarray(image.crop(self._stem_box).convert("RGB")) >= 180, axis=2)

    def _color(self, image: Image.Image) -> tuple[str, int, int]:
        """判定按钮内部颜色。返回 (判定, 绝对红点数, 绝对蓝点数)。"""
        pixels = image.load()
        red = blue = 0
        to_red = to_blue = 0
        for index, (x, y) in enumerate(self._points):
            r, g, b = pixels[x, y][:3]
            blue_red = b - r
            if r >= 130 and r - g >= 40 and r - b >= 40:
                red += 1
            if b >= 150 and g >= 70 and blue_red >= 40:
                blue += 1
            # 相对位移：颜色渐显到一半时绝对亮度不够，但色相已经转过去。
            delta = blue_red - self._base_br[index]
            if delta >= _SHIFT_DELTA and blue_red >= _SHIFT_TARGET:
                to_blue += 1
            if -delta >= _SHIFT_DELTA and -blue_red >= _SHIFT_TARGET:
                to_red += 1
        # 绝对判据优先：它在前 9 题上已被实机验证过；色相位移只作补充。
        if red >= _ABSOLUTE_MIN:
            return "wrong", red, blue
        if blue >= _ABSOLUTE_MIN:
            return "correct", red, blue
        if to_red >= _SHIFT_MIN:
            return "wrong", red, blue
        if to_blue >= _SHIFT_MIN:
            return "correct", red, blue
        return "unknown", red, blue
    def observe(self, image: Image.Image) -> None:
        if not self.started or self._done:
            return
        if image.size != self._size or image.mode != "RGB":
            self._reason = "截图尺寸或格式变化，停止归属本题"
            self._done = True
            return
        color, red, blue = self._color(image)
        self._color_detail = f"当前红点{red}/{_POINTS}、蓝点{blue}/{_POINTS}（{color}）"
        # 留证留**颜色证据最强**的那一帧 —— 第一帧通常还没变色。
        if red + blue > self._best_score:
            self._best_score = red + blue
            self._diagnostic_frame = image
        if self.last_question and len(self._window) < _WINDOW_KEEP:
            self._window.append(image)
        changed = int(np.count_nonzero(self._stem_mask(image) != self._stem))
        if changed > 20:
            self._reason = f"题干变化{changed}像素（阈值20），停止归属本题"
            self._done = True
            return
        if color != "unknown" and color != self._before_color:
            self.status = color
            self._done = True
            self._diagnostic_frame = None
        else:
            self._reason = f"颜色未明确变化，题干变化{changed}像素"

    def observe_now(self, grab) -> None:
        """立刻取一帧。答后颜色只亮很短一瞬，晚了只剩渐隐的暗色。"""
        try:
            image = grab()
        except Exception:  # noqa: BLE001 - 反馈是尽力而为，不能影响答题
            return
        if image is not None:
            self.observe(image)

    def sample_window(self, grab, *, offsets=SAMPLE_OFFSETS, delay: float = 1.0) -> None:
        """在固定延迟内连取几帧送进观察器 —— 揭示色只亮一瞬，晚了只剩渐隐的暗色。

        总等待时间仍是 delay：取帧本身约 20 毫秒，落在等待里。
        """
        started = time.monotonic()
        for offset in offsets:
            gap = offset - (time.monotonic() - started)
            if gap > 0:
                time.sleep(gap)
            if time.monotonic() - started >= delay:
                break
            self.observe_now(grab)
        time.sleep(max(0.0, delay - (time.monotonic() - started)))

    def finish(self, result: dict, *, recorder=None, verbose: bool = True,
               label: str = "本题", terminal: bool = False) -> None:
        if not self.started or self._reported:
            return
        self._reported = True
        result["answer_feedback"] = self.status
        text = {"correct": "正确", "wrong": "错误",
                "unknown": "未确认（未捕获到明确的同题变色反馈）"}[self.status]
        message = f"{label} 答后反馈：{text}"
        safe_log(recorder, message)
        if self.status == "unknown":
            safe_log(recorder, f"{label} 答后反馈诊断：{self._reason}；"
                     f"{self._before_detail}；{self._color_detail}")
            if terminal and recorder is not None:
                # 末题：把整段取帧窗口落盘 —— 颜色只亮一瞬，一帧看不出它
                # 到底出现过没有、什么时候消失的。非末题只留一张代表帧。
                if self._window:
                    for index, image in enumerate(self._window, 1):
                        try:
                            recorder.shot(f"answer_feedback_window_{index}", image=image)
                        except OSError as exc:
                            safe_log(recorder, f"{label} 反馈留证失败（{type(exc).__name__}）")
                elif self._diagnostic_frame is not None:
                    try:
                        recorder.shot("answer_feedback_unknown_first",
                                      image=self._diagnostic_frame)
                    except OSError as exc:
                        safe_log(recorder, f"{label} 反馈留证失败（{type(exc).__name__}）")
        self._diagnostic_frame = None
        # 窗口帧是整张 430×788 的截图，留着就是十几兆内存 —— 报完就放掉。
        self._window.clear()
        if verbose and not getattr(recorder, "echo", False):
            print(f"  {message}")
