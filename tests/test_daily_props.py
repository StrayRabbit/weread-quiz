"""daily 道具：数量与颜色双重门控，不依赖模型或真实鼠标。"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from weread_quiz.daily import props


FIXTURES = Path(__file__).parent / "fixtures"


def prop_frame(*, skip=True, swap=True, skip_count="2", swap_count="1"):
    """造一张带两块道具的底图；文字带按实测坐标画。"""
    image = Image.new("RGB", (430, 788), (8, 12, 20))
    draw = ImageDraw.Draw(image)
    # 跳题卡文字带：x109..199, y640..690。亮=可用，暗=禁用。
    draw.rectangle((109, 645, 199, 685), fill=(235, 90, 110) if skip else (70, 34, 46))
    draw.rectangle((289, 645, 379, 685), fill=(190, 110, 235) if swap else (60, 40, 84))
    return image


def ocr_for(skip="跳题卡·2", swap="换题卡·1", confidence=0.99):
    # OCR 走识别模式（没有框），按调用顺序返回两行。
    labels = iter([skip, swap])
    return SimpleNamespace(_engine=object(), warm_error=None,
                           _engine_run_text=lambda image: [(next(labels), confidence)])


def test_quantity_and_color_are_both_required():
    states = props.read_props(prop_frame(skip=False), ocr_for())
    assert states[0].count == 2 and not states[0].available
    assert states[1].count == 1 and states[1].available
    assert props.choose_prop(states, last_question=False).kind == "swap"


@pytest.mark.parametrize("label", ["跳题卡·0", "跳题卡", "跳题卡·O", "跳题卡·-1", "续命卡·2"])
def test_zero_or_unknown_count_never_available(label):
    states = props.read_props(prop_frame(), ocr_for(skip=label, swap="换题卡·0"))
    assert props.choose_prop(states, last_question=False) is None


def test_traditional_character_is_normalized():
    states = props.read_props(prop_frame(), ocr_for(skip="跳題卡·2"))
    assert states[0].count == 2 and states[0].available
    states = props.read_props(prop_frame(), ocr_for(swap="換題卡·1"))
    assert states[1].count == 1 and states[1].available


def test_conflicting_readings_are_treated_as_unknown():
    """同一条文字带读出两个不同数量 —— 不能挑一个猜，读数作废。"""
    lines = iter([[("跳题卡·2", 0.99), ("跳题卡·3", 0.99)], [("换题卡·0", 0.99)]])
    ocr = SimpleNamespace(_engine=object(), warm_error=None,
                          _engine_run_text=lambda image: next(lines))
    states = props.read_props(prop_frame(), ocr)
    assert states[0].count is None and not states[0].available


def test_skip_priority_and_last_question_prohibition():
    states = props.read_props(prop_frame(), ocr_for())
    assert props.choose_prop(states, last_question=False).kind == "skip"
    assert props.choose_prop(states, last_question=True).kind == "swap"


def test_last_question_without_swap_falls_back_to_nothing():
    states = props.read_props(prop_frame(swap=False), ocr_for(swap="换题卡·1"))
    assert props.choose_prop(states, last_question=True) is None


def test_low_confidence_and_unwarmed_ocr_are_rejected():
    assert not any(p.available for p in props.read_props(prop_frame(), ocr_for(confidence=0.5)))
    assert props.read_props(prop_frame(), None) == []
    ocr = ocr_for()
    ocr._engine = None
    assert props.read_props(prop_frame(), ocr) == []


def test_wrong_button_position_is_rejected():
    """读到的名字必须对应自己的卡片：换题卡读数不能算进跳题卡。"""
    ocr = ocr_for(skip="换题卡·2", swap="跳题卡·1")
    states = props.read_props(prop_frame(), ocr)
    assert states[0].count is None and states[1].count is None
    assert props.choose_prop(states, last_question=False) is None


def test_small_or_blank_image_is_rejected():
    assert props.read_props(Image.new("RGB", (200, 200)), ocr_for()) == []
    assert not any(p.available for p in props.read_props(Image.new("RGB", (430, 788)), ocr_for()))


def _fixture(name):
    with Image.open(FIXTURES / name) as source:
        return source.convert("RGB")


@pytest.mark.parametrize("name, expected_color", [
    # 真实截图：跳题卡亮色（库存 2）与末题禁用（库存 1，颜色变暗）。
    ("daily_props_skip_enabled.png", True),
    ("daily_props_skip_disabled.png", False),
])
def test_real_prop_bar_brightness_decides_availability(name, expected_color):
    """真实道具条上的亮度判据（夹具来自实机截图的道具区）。"""
    image = _fixture(name)
    assert props.has_enabled_color(image, "skip") is expected_color
    # 换题卡在这两张截图里都是 0 张（暗色）。
    assert props.has_enabled_color(image, "swap") is False


def test_real_prop_bar_reads_count_and_traditional_text():
    """真实截图上的文字带 OCR 结果（离线夹具，不加载模型）。

    真模型跑一遍真实帧要放在 tools/check_props.py 里 —— 离线测试不加载
    onnxruntime（项目约定，也免得拖慢整套回归）。
    """
    lines = [("跳題卡·2", 0.99), ("换题卡·0", 0.98)]
    ocr = SimpleNamespace(_engine=object(), warm_error=None,
                          _engine_run_text=lambda image: [lines.pop(0)])
    states = props.read_props(_fixture("daily_props_skip_enabled.png"), ocr)
    skip = next(s for s in states if s.kind == "skip")
    assert skip.count == 2 and skip.enabled_color
    assert props.choose_prop(states, last_question=False).kind == "skip"


def test_real_toast_is_detected():
    """真实道具提示条（「跳题卡使用成功」，实机帧 20261110-084219）。

    提示条盖住题号行和题干：它在的时候读到的题号不稳、题干帧差
    也是假的，所以判据必须能认出它。
    """
    assert props.has_toast(_fixture("daily_prop_toast.png")) is True
    # 没有提示条的题面帧不能误报。
    assert props.has_toast(_fixture("daily_props_skip_enabled.png")) is False
    # 坐标是标定过的：宽度不对的截图不认。
    assert props.has_toast(Image.new("RGB", (200, 200))) is False
