"""按路线文字白名单推进到答题页；精确匹配目标，点击前复核文字和位置，不使用道具。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import abort
from .capture import grab_screen
from .click import click_box
from .models import Rect

DEFAULT_MIN_CONFIDENCE = 0.85

RECHECK_TOLERANCE = 12

_RECHECK_OK = "ok"
_RECHECK_GONE = "gone"
_RECHECK_MOVED = "moved"


@dataclass(frozen=True)
class TextHit:
    """一次文字命中：文字本身 + 置信度 + 检测框。"""

    text: str
    confidence: float
    box: tuple  # 四角坐标 ((x0,y0),(x1,y1),(x2,y2),(x3,y3))


def read_text_lines(raw) -> list[tuple[str, float, tuple]]:
    """把 RapidOCR 的原始返回整理成 (文字, 置信度, 四角框)。"""
    lines: list[tuple[str, float, tuple]] = []
    for entry in raw or []:
        try:
            box, text, confidence = entry[0], entry[1], float(entry[2])
        except (TypeError, IndexError, ValueError):
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        lines.append((cleaned, confidence, tuple(tuple(p) for p in box)))
    return lines


def find_target(lines, targets, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
    """在文字行里找目标按钮。"""
    if isinstance(targets, str):
        targets = (targets,)
    for target in targets:
        for text, confidence, box in lines:
            if confidence < min_confidence:
                continue
            if text == target:
                return TextHit(text=text, confidence=confidence, box=box)
    return None


def box_center(box) -> tuple[float, float]:
    """检测框的中心点。"""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def boxes_match(first, second, tolerance: float = RECHECK_TOLERANCE) -> bool:
    """两个检测框是不是同一个按钮？中心点距离在 tolerance 内就算。"""
    if not first or not second:
        return False
    x1, y1 = box_center(first)
    x2, y2 = box_center(second)
    return abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance


def box_to_rect(box) -> Rect:
    """检测框 → Rect（图像坐标系）。"""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    left = int(min(xs))
    top = int(min(ys))
    return Rect(
        left=left,
        top=top,
        width=int(max(xs) - left),
        height=int(max(ys) - top),
    )


# ---------------------------------------------------------------
# 推进循环
# ---------------------------------------------------------------

def _read_all_lines(ocr, image) -> list[tuple[str, float, tuple]]:
    """让 OCR 读整张图的全部文字。"""
    if ocr is None:
        return []
    runner = getattr(ocr, "_engine_run", None)
    if runner is None:
        return []
    try:
        raw = runner(image)
    except Exception:  # noqa: BLE001 - 读不出当"这一页没读到"
        return []
    return read_text_lines(raw)


def _recheck(region, ocr, hit: TextHit) -> str:
    """点击前复核：目标文字还在吗？位置变了吗？"""
    fresh = grab_screen(region)
    lines = _read_all_lines(ocr, fresh)
    current = find_target(lines, hit.text)
    if current is None:
        return _RECHECK_GONE
    if not boxes_match(hit.box, current.box):
        return _RECHECK_MOVED
    return _RECHECK_OK


def advance_to_question(region, scale, ocr, recorder, targets,
                         is_question_page, timeout: float = 90.0,
                         poll_interval: float = 0.3, *, before_click=None) -> bool:
    """把界面推进到答题页；可在点击前准备，准备完成后仍须重新复核按钮。"""
    targets = tuple(targets)
    last_target = targets[-1] if targets else None
    deadline = time.monotonic() + timeout
    last_seen: list[str] = []
    # 本轮推进中作废的目标（复核时位置对不上）。
    banned: set[str] = set()

    while time.monotonic() < deadline:
        abort.check()
        image = grab_screen(region)
        if is_question_page is not None and is_question_page(image):
            return True

        lines = _read_all_lines(ocr, image)
        last_seen = [text for text, _, _ in lines]

        # 作废的目标从屏幕上消失了 → 页面已经换掉了，它重新变得可信。
        for name in list(banned):
            if name not in last_seen:
                banned.discard(name)

        usable = [t for t in targets if t not in banned]
        hit = find_target(lines, usable)
        if hit is None:
            time.sleep(poll_interval)
            continue

        if before_click is not None:
            before_click(hit.text)
        status = _recheck(region, ocr, hit)
        if status == _RECHECK_GONE:
            _log(recorder, f"推进：复核时「{hit.text}」已消失，本轮放弃")
            time.sleep(poll_interval)
            continue
        if status == _RECHECK_MOVED:
            banned.add(hit.text)
            _log(recorder, f"推进：复核时「{hit.text}」位置已变，本轮不再点击它")
            time.sleep(poll_interval)
            continue

        rect = box_to_rect(hit.box)
        _log(recorder, f"推进：点击「{hit.text}」(置信度 {hit.confidence:.3f})")
        click_box(rect, region, scale)
        if hit.text == last_target:
            # 白名单的最后一个目标 —— 点完就是进入答题。
            return True
        time.sleep(poll_interval)

    # 超时：把当前画面存下来 —— 那是排错的唯一依据。
    if recorder:
        try:
            path = recorder.shot("advance_stuck")
            _log(recorder, f"推进失败：读到的文字 {last_seen}")
            _log(recorder, f"推进失败时的截图：{path}")
            # 打印到终端 —— 用户在控制台前看不到 run.log。
            print(f"  推进失败时的截图：{path}")
        except Exception:  # noqa: BLE001 - 落盘失败不该掩盖超时本身
            pass
    return False


def _log(recorder, message: str) -> None:
    """写一行推进日志。日志失败不能影响推进。"""
    if recorder is None:
        return
    try:
        recorder.log(message)
    except Exception:  # noqa: BLE001
        pass
