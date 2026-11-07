"""每日一答混合识别：本地确认、预读文本答题和图片回退。"""
from __future__ import annotations

import time

from PIL import Image

from .ocr import DailyOcrRead, DailyOcrReader, TEXT_PROMPT
from .vision import FALLBACK_RESERVE_SECONDS, text_path_deadline
from ..errors import AnswerUnavailable, OcrRejected
from ..recognition import answer_from_read, read_local, text_then_image
from ..vision import VisionQuestion, VisionService, _check_deadline

# 与 daily.py 的同名常量保持一致（数字对齐 PK 链路）。
_IDENTIFY_BUDGET_SECONDS = 9.0
_TEXT_TIMEOUT = 5.0


def _crop_card(image: Image.Image) -> Image.Image:
    """只裁模型输入；OCR 和按钮定位始终使用原图坐标。"""
    from .vision import _DAILY_CROP

    return _DAILY_CROP(image) if image.size[1] > 600 else image


class DailyHybridReader:
    """代理视觉服务，使用 daily 独立的 OCR 几何及置信度门控。"""

    expects_full_image = True

    def __init__(self, vision: VisionService, ocr: DailyOcrReader | None = None,
                 *, verbose: bool = True, log_sink=None) -> None:
        # vision 是模型服务门面 —— 文本路径和整图路径都走它。
        self._vision = vision
        self._ocr = ocr or DailyOcrReader()
        self._verbose = verbose
        self._log_sink = log_sink
        self.local_ocr_available: bool | None = None
        self.ocr_accepted = 0
        self.ocr_fallback = 0
        self._last_progress_no: int | None = None
        self._last_progress_image = None

    def warmup(self, sample=None) -> bool:
        """开局前加载 OCR；传入真实尺寸样本可同时预热推理。"""
        ok = self._ocr.warmup(sample)
        self.local_ocr_available = ok
        if not ok:
            reason = getattr(self._ocr, "load_error", "") or "未知原因"
            self._log(f"OCR 快路径不可用：{reason}")
            if self._verbose:
                print(f"⚠ RapidOCR 不可用，本次全程走图片路径：{reason}")
            return False
        warm_error = getattr(self._ocr, "warm_error", "")
        if warm_error:
            self._log(f"OCR 预热失败（不影响识别）：{warm_error}")
        return True

    def gate_summary(self) -> str:
        total = self.ocr_accepted + self.ocr_fallback
        if not total:
            return "OCR 快路径：本局未触发"
        return (f"OCR 快路径：命中 {self.ocr_accepted}/{total}，"
                f"回退图片 {self.ocr_fallback} 次")

    def read_question_local(self, image: Image.Image, *, deadline=None) -> DailyOcrRead:
        """纯本地读题。判否携带题号证据，不把残缺选项当作可答读数。"""
        self._last_progress_image = image
        self._last_progress_no = None
        local = read_local(self._ocr, image, deadline=deadline)
        evidence = local.read
        self._last_progress_no = getattr(evidence, "progress_no", None)
        if not local.decision.accepted or evidence is None:
            raise OcrRejected(local.decision.reason or "OCR 门控未通过", read=evidence)
        return evidence

    def answer_from_ocr_read(self, read: DailyOcrRead, *, timeout=None,
                             deadline=None) -> VisionQuestion | None:
        """复用通过门控的预读；模型失败返回 None，不再跑 OCR。"""
        question = answer_from_read(self._vision, read, TEXT_PROMPT,
                                    timeout=timeout, deadline=deadline, log=self._log)
        if question is None:
            return None
        self.ocr_accepted += 1
        self._log(f"OCR 快路径 置信度 {[round(c, 3) for c in read.confidences]}"
                  f" 最低 {read.min_confidence:.2f} → {read.stem} → {read.options[question.answer_index]}")
        return question

    def read_question_visual(self, image: Image.Image, prompt=None, *,
                             timeout=None, deadline=None) -> VisionQuestion:
        """只走图片路径，避免本地判否后重复 OCR。"""
        from .vision import DAILY_PROMPT

        _check_deadline(deadline)
        self.ocr_fallback += 1
        options = {"timeout": timeout}
        if deadline is not None:
            options["deadline"] = deadline
        question = self._vision.read_question(
            _crop_card(image), prompt=prompt or DAILY_PROMPT, **options,
        )
        _check_deadline(deadline)
        if self._last_progress_image is image and question.progress_no is None:
            question.progress_no = self._last_progress_no
        return question

    def read_question(self, image: Image.Image, prompt=None, *,
                      timeout=None, deadline=None) -> VisionQuestion:
        """识别级联共用一个绝对预算，文本失败后只花剩余预算回退图片。"""
        if deadline is None:
            deadline = time.monotonic() + _IDENTIFY_BUDGET_SECONDS
        if self.local_ocr_available is False:
            return self.read_question_visual(image, prompt=prompt, timeout=timeout, deadline=deadline)
        def answer():
            try:
                local = self.read_question_local(image, deadline=deadline)
            except OcrRejected as exc:
                self._log(f"图片路径 回退原因：{exc}")
                return None
            _check_deadline(deadline)
            text_deadline = text_path_deadline(deadline, time.monotonic(), _TEXT_TIMEOUT)
            try:
                question = self.answer_from_ocr_read(
                    local, timeout=min(timeout or _TEXT_TIMEOUT, _TEXT_TIMEOUT),
                    deadline=text_deadline,
                )
            except AnswerUnavailable:
                question = None
            if question is None:
                self._log("图片路径 回退原因：OCR 文本路径未取到合法答案")
            return question

        return text_then_image(
            answer,
            lambda: self.read_question_visual(image, prompt=prompt, timeout=timeout, deadline=deadline),
            deadline=deadline, minimum_image_budget=FALLBACK_RESERVE_SECONDS,
            log=self._log,
        )

    def read_progress_number(self, image: Image.Image) -> int | None:
        """复用同一张图的题号；未读过该图时才单独 OCR。"""
        if self._last_progress_image is image:
            return self._last_progress_no
        from .number import read_question_number

        return read_question_number(image, ocr=self._ocr)

    def _log(self, message: str) -> None:
        if self._log_sink is not None:
            try:
                self._log_sink(message)
            except Exception:
                pass  # 日志失败不能影响答题。

    def __getattr__(self, name: str):
        return getattr(self._vision, name)

    def __enter__(self) -> "DailyHybridReader":
        self._vision.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._vision.__exit__(*exc_info)
