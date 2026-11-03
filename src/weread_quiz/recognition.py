"""两种答题模式共用的本地识别、文本求答和单次图片级联机制。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from PIL import Image

from .errors import RecognitionTimeout
from .ocr_core import OcrDecision, OcrRead
from .vision import VisionQuestion, VisionService, _check_deadline


@dataclass
class LocalRecognition:
    read: OcrRead | None
    decision: OcrDecision
    strip_ran: bool = False
    strip_options: int = 0
    deadline: float | None = None


def read_local(ocr, image: Image.Image, *, deadline: float | None = None,
               observe: Callable[[OcrRead | None], float | None] | None = None) -> LocalRecognition:
    """按需补读一次；判否仍保留题号证据，不调用模型。

    observe 在整图 OCR 后、deadline 检查前运行，让倒计时预算锚定原截图。
    """
    _check_deadline(deadline)
    whole = ocr.read(image)
    if observe is not None:
        deadline = observe(whole)
    _check_deadline(deadline)
    decision = ocr.decide_merged(whole, None)
    strip = None
    strip_ran = False
    if not decision.accepted and (decision.needs_strip or whole is None):
        strip = ocr.read_strip(image, whole)
        strip_ran = True
        _check_deadline(deadline)
        decision = ocr.decide_merged(whole, strip)
    evidence = decision.read if decision.accepted else whole
    if ((evidence is None or not evidence.has_progress)
            and strip is not None and strip.has_progress):
        evidence = strip
    return LocalRecognition(
        evidence, decision, strip_ran,
        sum(bool(text.strip()) for text in strip.options) if strip is not None else 0,
        deadline,
    )


def answer_from_read(vision: VisionService, read: OcrRead, prompt: str, *,
                     timeout: float | None = None,
                     deadline: float | None = None,
                     log: Callable[[str], None] | None = None) -> VisionQuestion | None:
    """完整选项才请求一次文本模型；下标始终对应原按钮槽位。"""
    if (not read.options or len(read.options) != read.button_count
            or any(not text.strip() for text in read.options)):
        if log is not None:
            log("本地 OCR 选项不完整，不请求文本答案")
        return None
    options = "\n".join(f"{index}. {text}" for index, text in enumerate(read.options))
    _check_deadline(deadline)
    request_options = {"timeout": timeout}
    if deadline is not None:
        request_options["deadline"] = deadline
    index = vision.answer_from_text(prompt % (read.stem, options),
                                    len(read.options), **request_options)
    _check_deadline(deadline)
    if index is None or not (0 <= index < len(read.options)):
        return None
    return VisionQuestion(
        stem=read.stem, options=list(read.options), answer_index=index,
        screen="question", local_read=True,
        progress_no=getattr(read, "progress_no", None),
    )


def text_then_image(answer: Callable[[], VisionQuestion | None],
                    visual: Callable[[], VisionQuestion], *,
                    deadline: float | None = None,
                    minimum_image_budget: float,
                    log: Callable[[str], None] | None = None) -> VisionQuestion:
    """文本只试一次；图片只消费原 deadline 的余量，不续期、不重试 OCR。

    answer 的异常是否可回退由模式 adapter 决定，这里不混淆本地判否和模型失败。
    """
    question = answer()
    if question is not None:
        return question
    _check_deadline(deadline)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining < minimum_image_budget:
            message = (f"图片路径 跳过：剩余预算 {remaining:.2f}s "
                       f"< {minimum_image_budget:.1f}s，走不了图片路径")
            if log is not None:
                log(message)
            raise RecognitionTimeout(message)
    return visual()
