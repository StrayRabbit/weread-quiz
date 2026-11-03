"""PK 像素判据：题干帧差、渲染、揭示态、同题校验及三态卡片模板。"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from .buttons import find_option_buttons
from .errors import WqError

# 题干所在的纵向区间。选项区不能算进去 —— 按钮动画噪声太大。
_STEM_BAND = (250, 345)

_STEM_X_RANGE = (40, 390)

# 二值化阈值：只把足够亮的像素算作"文字"
_STEM_WHITE = 180

_STEM_CHANGE_THRESHOLD = 20

_PROGRESS_CACHE_BAND = (248, 296)

_PROGRESS_CHANGE_THRESHOLD = 40


def _stem_band(image: Image.Image) -> np.ndarray:
    """题干判定带：纵向 y250-345，横向 x40-390 的灰度矩阵。"""
    arr = np.asarray(image.convert("L"))
    top, bottom = _STEM_BAND
    band = arr[top : min(bottom, arr.shape[0])]
    left, right = _STEM_X_RANGE
    if band.shape[1] > right:
        band = band[:, left:right]
    return band


def stem_mask(image: Image.Image) -> np.ndarray:
    """题干区二值化掩码。True 表示该像素属于白色文字。"""
    return _stem_band(image) >= _STEM_WHITE


def stem_change_count(before: Image.Image, after: Image.Image) -> int:
    """两张图的题干区有多少个像素的明暗状态不同。"""
    mask_a = stem_mask(before)
    mask_b = stem_mask(after)
    if mask_a.shape != mask_b.shape:
        # 尺寸变了就是画面重绘，按大幅变化处理
        return max(mask_a.size, mask_b.size)
    return int(np.count_nonzero(mask_a != mask_b))


def progress_cache_mask(image: Image.Image) -> np.ndarray:
    """等待阶段题号 OCR 缓存的像素基准，仅包含题号行区块，不作题目身份校验。"""
    gray = np.asarray(image.convert("L"))
    left, right = _STEM_X_RANGE
    top, bottom = _PROGRESS_CACHE_BAND
    return gray[top:bottom, left:right] >= _STEM_WHITE


def progress_cache_changed(baseline: np.ndarray, size: tuple[int, int],
                           image: Image.Image) -> bool:
    """始终与执行 OCR 的帧比较，避免渐变被滚动基准吞掉。"""
    if image.size != size:
        return True
    current = progress_cache_mask(image)
    return (baseline.shape != current.shape
            or int(np.count_nonzero(baseline != current)) > _PROGRESS_CHANGE_THRESHOLD)


def is_new_question(before: Image.Image, after: Image.Image) -> bool:
    """after 是不是换到了下一题。"""
    return stem_change_count(before, after) > _STEM_CHANGE_THRESHOLD


def is_question_stable(before: Image.Image, after: Image.Image) -> bool:
    """相邻候选帧的题干和按钮布局是否稳定，忽略倒计时和颜色动画。"""
    if before.size != after.size or is_new_question(before, after):
        return False
    try:
        previous = find_option_buttons(before)
        current = find_option_buttons(after)
    except WqError:
        return False
    return len(previous) == len(current) and all(
        abs(a.center_y - b.center_y) <= 2
        for a, b in zip(previous, current)
    )


def is_same_question(before: Image.Image, after: Image.Image) -> bool:
    """点击专用身份校验：题干、布局及逐按钮白色文字不变。"""
    if not is_question_stable(before, after):
        return False
    previous = find_option_buttons(before)
    current = find_option_buttons(after)
    for a, b in zip(previous, current):
        if any(abs(getattr(a, name) - getattr(b, name)) > 2
               for name in ("left", "top", "width", "height")):
            return False
        masks = []
        for image, box in ((before, a), (after, b)):
            interior = image.crop((box.left + 8, box.top + 6,
                                   box.right - 8, box.bottom - 6))
            masks.append(np.asarray(interior.convert("L")) >= _STEM_WHITE)
        if masks[0].shape != masks[1].shape:
            return False
        if np.count_nonzero(masks[0] != masks[1]) > _STEM_CHANGE_THRESHOLD:
            return False
    return True


def is_question_screen(image: Image.Image) -> bool:
    """这张图是不是正在答题（而不是匹配页、结算页、聊天页）。"""
    try:
        return len(find_option_buttons(image)) >= 2
    except WqError:
        return False


_STEM_TEXT_BAND = (296, 360)

_STEM_TEXT_MIN_PIXELS = 100


def stem_text_pixels(image: Image.Image) -> int:
    """题干正文区（不含题号行）的白色像素数。"""
    arr = np.asarray(image.convert("L"))
    top, bottom = _STEM_TEXT_BAND
    band = arr[top : min(bottom, arr.shape[0])]
    if band.size == 0:
        return 0
    left, right = _STEM_X_RANGE
    if band.shape[1] > right:
        band = band[:, left:right]
    return int((band >= _STEM_WHITE).sum())


def has_rendered_stem(image: Image.Image) -> bool:
    """题干正文区存在文字像素时返回真；空白半渲染帧不放行。"""
    return stem_text_pixels(image) >= _STEM_TEXT_MIN_PIXELS


_OPTION_AREA = (340, 700)

_REVEAL_MIN_GREEN = 120
_REVEAL_MIN_GREEN_OVER_RED = 60

_REVEAL_FILL_THRESHOLD = 5000


def reveal_fill_count(image: Image.Image) -> int:
    """选项区里"答对高亮"的绿色像素数。越大越像揭示态。"""
    arr = np.asarray(image.convert("RGB")).astype(int)
    top, bottom = _OPTION_AREA
    band = arr[top : min(bottom, arr.shape[0])]
    if band.size == 0:
        return 0
    red, green = band[:, :, 0], band[:, :, 1]
    mask = (green > _REVEAL_MIN_GREEN) & ((green - red) > _REVEAL_MIN_GREEN_OVER_RED)
    return int(np.count_nonzero(mask))


def has_answer_reveal(image: Image.Image) -> bool:
    """这张图是不是**答后揭示态**：刚答完，正确项还没被点亮动画收走。"""
    return reveal_fill_count(image) >= _REVEAL_FILL_THRESHOLD


# 模板图的来源分辨率：卡片转角就是从这张 414×780 的真实答题页裁的。
_TEMPLATE_REF_SIZE = (414, 780)

# 搜索区域：截图里卡片左上角所在的那一块，比例坐标 (x0, y0, x1, y1)。
#
# 模板本身是 assets/pk_question_card.png —— 答题卡片左上角那一小段
# **框线转角**（参考分辨率下 x26-58 / y196-238，共 32×42 像素）。
# 它不含任何文字、不含倒计时、不含按钮，两题之间一模一样。
#
# x 从 2% 到 34%、y 从 21% 到 36% —— 覆盖卡片转角，上下左右各留了
# 余量，让转角在区域里挪动几个像素也照样能匹配上。
_TEMPLATE_REGION = (0.02, 0.21, 0.34, 0.36)

# 归一化后的区域尺寸（参考分辨率下的像素）。
_TEMPLATE_REGION_SIZE = (
    int(round((_TEMPLATE_REGION[2] - _TEMPLATE_REGION[0]) * _TEMPLATE_REF_SIZE[0])),
    int(round((_TEMPLATE_REGION[3] - _TEMPLATE_REGION[1]) * _TEMPLATE_REF_SIZE[1])),
)

# 匹配分数阈值。
#
_TEMPLATE_SCORE_THRESHOLD = 0.75

# 宽高比容差。相对参考分辨率的宽高比偏差超过这个值就判"尺寸不可信"，
# 返回 None —— 宁可不用这条判据，也别拿一个明显不同的画面硬比。
_TEMPLATE_ASPECT_TOLERANCE = 0.15

# Canny 边缘检测阈值。框线转角是细线，用边缘比用灰度更能凸显形状。
_TEMPLATE_CANNY = (80, 200)

# 解码后的模板图缓存，以及"资源不可用已提示过"的标记。
# 等待阶段每帧都会调，重复读盘既慢又会在资源缺失时刷屏。
_template_cache: Image.Image | None = None
_template_warned = False


def _resolve_template_resource() -> Path:
    """定位打包进来的卡片模板图。"""
    try:
        candidate = resources.files(__package__).joinpath("assets/pk_question_card.png")
        path = Path(str(candidate))
        if path.is_file():
            return path
    except (ModuleNotFoundError, TypeError, ValueError, OSError):
        pass
    return Path(__file__).resolve().parent / "assets" / "pk_question_card.png"


_TEMPLATE_RESOURCE = _resolve_template_resource()


def _warn_template_unavailable(reason: str) -> None:
    """资源不可用只提示一次 —— 等待阶段每帧都会调，提示多了是刷屏。"""
    global _template_warned
    if _template_warned:
        return
    _template_warned = True
    print(f"  ⚠ PK 卡片模板不可用（{reason}），本轮跳过结构预筛")


def _load_template_image() -> Image.Image | None:
    """从磁盘读模板图。读不出来返回 None（不抛异常）。"""
    try:
        with Image.open(_TEMPLATE_RESOURCE) as image:
            return image.convert("RGB")
    except (OSError, ValueError, SyntaxError) as exc:
        _warn_template_unavailable(f"{type(exc).__name__}")
        return None


def _template_image() -> Image.Image | None:
    """拿到模板图，进程内只解码一次。"""
    global _template_cache
    if _template_cache is None:
        _template_cache = _load_template_image()
    return _template_cache


def _edges(image: Image.Image):
    """灰度 + Canny 边缘。"""
    import cv2  # noqa: PLC0415

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, _TEMPLATE_CANNY[0], _TEMPLATE_CANNY[1])


def _template_score(image: Image.Image, template: Image.Image) -> float | None:
    """算出这张图与卡片模板的相似度。尺寸不可信时返回 None。"""
    import cv2  # noqa: PLC0415

    width, height = image.size
    ref_w, ref_h = _TEMPLATE_REF_SIZE
    if width <= 0 or height <= 0:
        return None
    aspect = width / height
    ref_aspect = ref_w / ref_h
    if abs(aspect - ref_aspect) / ref_aspect > _TEMPLATE_ASPECT_TOLERANCE:
        # 画面比例差太多，缩放也救不回来 —— 交给别的门控判断。
        return None

    x0, y0, x1, y1 = _TEMPLATE_REGION
    box = (
        max(0, int(round(x0 * width))),
        max(0, int(round(y0 * height))),
        min(width, int(round(x1 * width))),
        min(height, int(round(y1 * height))),
    )
    if box[2] - box[0] < 10 or box[3] - box[1] < 10:
        return None
    region = image.crop(box).resize(_TEMPLATE_REGION_SIZE, Image.LANCZOS)

    haystack = _edges(region)
    needle = _edges(template)
    if (haystack.shape[0] < needle.shape[0]
            or haystack.shape[1] < needle.shape[1]):
        return None
    result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def looks_like_question_template(image: Image.Image) -> bool | None:
    """卡片结构模板三态判定：匹配为 True，不匹配为 False，资源或比例不可用为 None。"""
    template = _template_image()
    if template is None:
        return None
    try:
        score = _template_score(image, template)
    except Exception:  # noqa: BLE001 - 预筛算不出来不能打断答题流程
        # KeyboardInterrupt / SystemExit 不是 Exception 的子类，
        # 所以紧急停止不会被吞掉。
        return None
    if score is None:
        return None
    return score >= _TEMPLATE_SCORE_THRESHOLD

