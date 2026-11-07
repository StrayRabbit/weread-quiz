"""显示缩放适配：抓屏归一化到标定宽度，点击坐标按物理窗口换算。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from weread_quiz import capture
from weread_quiz.pk.buttons import find_option_buttons
from weread_quiz.dpi import BASELINE_WIDTH, box_center_to_screen
from weread_quiz.models import Rect


class _FakeRaw:
    def __init__(self, size):
        self.size = size
        self.rgb = bytes(size[0] * size[1] * 3)


class _FakeMSS:
    """替身 mss：按请求的区域返回一块纯色。"""

    def __init__(self, screen=(1920, 1080)):
        self._screen = screen
        # mss 的约定：0 是所有屏幕的并集，1 是第一块屏幕。
        self.monitors = [
            {"left": 0, "top": 0, "width": screen[0], "height": screen[1]},
            {"left": 0, "top": 0, "width": screen[0], "height": screen[1]},
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def grab(self, monitor):
        if isinstance(monitor, dict):
            return _FakeRaw((monitor["width"], monitor["height"]))
        return _FakeRaw((monitor["width"], monitor["height"]))


def _install_fake_mss(monkeypatch, screen=(1920, 1080)):
    monkeypatch.setattr(capture.mss, "MSS", lambda: _FakeMSS(screen))


@pytest.mark.parametrize("factor", [1.0, 1.25, 1.5, 2.0])
def test_region_capture_is_normalized_to_baseline_width(monkeypatch, factor):
    """带区域的抓屏一律归一化到标定宽度 —— 下游像素判据只认这个坐标系。"""
    _install_fake_mss(monkeypatch)
    width = int(BASELINE_WIDTH * factor)
    region = Rect(left=100, top=50, width=width, height=int(788 * factor))

    image = capture.grab_screen(region)

    assert image.width == BASELINE_WIDTH
    assert image.height == pytest.approx(788, abs=6)


def test_full_screen_capture_is_not_normalized(monkeypatch):
    """整屏抓取是给人看的留证，不归一化。"""
    _install_fake_mss(monkeypatch, screen=(2560, 1440))

    image = capture.grab_screen(None)

    assert image.size == (2560, 1440)


@pytest.mark.parametrize("factor", [1.0, 1.25, 1.5, 2.0])
def test_click_point_stays_inside_the_window(factor):
    """点击点必须落在窗口内。

    早先的公式把坐标除以系统缩放比，150% 时整窗偏到窗口左侧 1/3 处，
    点到的根本不是按钮 —— 这条断言守着那个回归。
    """
    region = Rect(left=1878, top=125, width=int(642 * factor / 1.5),
                  height=int(1181 * factor / 1.5))
    box = Rect(left=60, top=340, width=310, height=44)

    x, y = box_center_to_screen(region, box)

    assert region.left <= x <= region.left + region.width
    assert region.top <= y <= region.top + region.height


def test_click_point_maps_back_to_the_same_relative_position():
    """标定坐标 → 物理坐标 → 回到标定坐标，位置不变。"""
    region = Rect(left=1878, top=125, width=642, height=1181)
    box = Rect(left=60, top=340, width=310, height=44)

    x, y = box_center_to_screen(region, box)
    factor = region.width / BASELINE_WIDTH

    assert (x - region.left) / factor == pytest.approx(box.left + box.width / 2)
    assert (y - region.top) / factor == pytest.approx(box.top + box.height / 2)


def _synthetic_question_page() -> Image.Image:
    """一张标定尺寸的答题页：两个青色边框的选项按钮。"""
    image = Image.new("RGB", (BASELINE_WIDTH, 788), "white")
    array = np.asarray(image).copy()
    for top in (377, 452):
        # 按钮：青色边框 + 浅青填充（真实界面就是这样被检出的）。
        array[top:top + 3, 60:370] = (0, 200, 215)
        array[top + 46:top + 49, 60:370] = (0, 200, 215)
        array[top:top + 49, 60:63] = (0, 200, 215)
        array[top:top + 49, 367:370] = (0, 200, 215)
        array[top + 3:top + 46, 60:370] = (225, 250, 252)
    return Image.fromarray(array)


@pytest.mark.parametrize("factor", [1.0, 1.25, 1.5, 2.0])
def test_buttons_are_found_after_normalizing_a_scaled_capture(factor):
    """150%/200% 的窗口抓下来归一化后，按钮位置与标定坐标一致。"""
    page = _synthetic_question_page()
    scaled = page.resize((int(BASELINE_WIDTH * factor),
                          int(788 * factor)), Image.LANCZOS)
    normalized = scaled.resize((BASELINE_WIDTH, 788), Image.LANCZOS)

    boxes = find_option_buttons(normalized)

    assert len(boxes) == 2
    assert boxes[0].left == 60
    assert boxes[0].width == 310
    assert boxes[0].center_y == pytest.approx(401, abs=6)
