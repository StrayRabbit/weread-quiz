"""测试按钮条检测。"""

import numpy as np

from tools.capture_watch import cyan_row_bands


def _synthetic(bands: list[tuple[int, int]], width: int = 430, height: int = 788):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (20, 35, 70)  # 深蓝底
    for y0, y1 in bands:
        img[y0 : y1 + 1, 45:290] = (60, 200, 220)  # 青色按钮
    return img


def test_detects_two_option_buttons():
    bands = cyan_row_bands(_synthetic([(377, 426), (453, 494)]))
    assert bands == [(377, 426), (453, 494)]


def test_returns_empty_when_no_buttons():
    assert cyan_row_bands(_synthetic([])) == []


def test_single_button_is_not_enough_to_trigger_capture():
    """只有一个按钮条说明不是答题界面（可能是别的页面）。"""
    assert len(cyan_row_bands(_synthetic([(377, 426)]))) == 1


def test_thin_border_lines_are_ignored():
    """卡片边框只有 1-2px 厚，不能被当成按钮。"""
    bands = cyan_row_bands(_synthetic([(200, 201), (377, 426), (453, 494)]))
    assert bands == [(377, 426), (453, 494)]


def test_four_option_buttons_all_detected():
    """万一以后变成四选一，同样要能全检出。"""
    bands = cyan_row_bands(_synthetic([(300, 340), (370, 410), (440, 480), (510, 550)]))
    assert len(bands) == 4


def test_dark_background_is_not_detected_as_button():
    """深蓝底本身不该被当成青色按钮。"""
    img = np.zeros((400, 430, 3), dtype=np.uint8)
    img[:, :] = (20, 35, 70)
    assert cyan_row_bands(img) == []
