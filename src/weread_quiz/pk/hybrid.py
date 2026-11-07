"""PK 混合识别：本地门控、预读文本答题、限时图片回退和用量代理。"""

from __future__ import annotations

import time
from collections import Counter

from PIL import Image

from ..errors import AnswerUnavailable, OcrRejected, RecognitionTimeout
from .budget import PKQuestionBudget, _IMAGE_PATH_MIN_BUDGET, _TEXT_PATH_TIMEOUT, text_path_timeout
from .ocr import OcrRead, RapidOcrReader, TEXT_PROMPT
from ..recognition import answer_from_read, read_local, text_then_image
from ..vision import VisionQuestion, VisionService, _check_deadline

_QUESTION_WINDOW_SECONDS = 12.0


def _text_timeout(timeout: float | None, cap: float) -> float:
    """文本路径的单次上限：再慢也不能超过 cap。

    留出来的预算给图片路径 —— 两条路都试过还答不出来的，才是真没救。
    """
    if timeout is None:
        return cap
    return min(timeout, cap)


def _short_reason(reason: str) -> str:
    """给局末汇总用的一句话摘要。

    完整原因（置信度、逐项读数）留在 run.log —— 摘要只是说明
    "本局为什么走了图片路径"。
    """
    if "未读出" in reason:
        # "没读出"比"置信度低"更严重，放在前面判 —— 整图路漏了按钮文字时，
        # 补读路再被低置信度拒掉，原因是「没读出来」，不是「阈值定高了」。
        # 归错类会让调阈值的人以为该降阈值。
        return "本地 OCR 没读出选项"
    if "置信度" in reason:
        return "本地 OCR 置信度不足"
    if "选项数不符" in reason or "按钮数" in reason:
        return "选项数与按钮数不一致"
    if "题号行" in reason:
        return "没读到题号行"
    if "两路结果不一致" in reason:
        return "两路读数不一致"
    if "合法下标" in reason:
        # 门控其实**通过了** —— 是模型没给出可用的下标。
        # 归到"本地门控未通过"是错的：那是两种完全不同的事故。
        return "模型没给出合法下标"
    return "本地门控未通过"


class HybridQuestionReader:
    """先试 OCR 文本路径，门控不通过就回退原图片路径。"""

    def __init__(
        self,
        vision: VisionService,
        ocr: RapidOcrReader | None = None,
        *,
        verbose: bool = True,
        log_sink=None,
        text_timeout: float | None = None,
    ) -> None:
        # vision 是模型服务门面 —— 文本路径和整图路径都走它，
        # 用哪种协议由门面自己认（见 model_service 模块）。
        self._vision = vision
        self._ocr = ocr or RapidOcrReader()
        self._verbose = verbose
        self._log_sink = log_sink
        # 文本路径的单次上限（秒）。默认 pk_budget._TEXT_PATH_TIMEOUT；
        # 慢模型可以在配置里调高（PK_TEXT_TIMEOUT）。
        self.text_timeout = (float(text_timeout) if text_timeout is not None
                             else _TEXT_PATH_TIMEOUT)
        self.local_ocr_available: bool | None = None
        self.last_question_progress_text = ""
        self.last_question_local_read: OcrRead | None = None
        self.last_question_local_finished_at = None
        self.last_question_model_started_at = None
        # 本局的门控统计，跑完打印，用来判断阈值定得合不合适。
        self.ocr_accepted = 0
        self.ocr_fallback = 0
        # 回退原因按短文案计数 —— 局末汇总用，说明"本局为什么回退"。
        self._fallback_reasons: Counter[str] = Counter()

    def warmup(self, sample=None) -> bool:
        """预加载 OCR 模型。返回是否可用（不可用时全程走图片路径）。"""
        ok = self._ocr.warmup(sample)
        self.local_ocr_available = ok
        if not ok:
            reason = getattr(self._ocr, "load_error", "") or "未知原因"
            self._log(f"OCR 快路径不可用：{reason}")
            if self._verbose:
                print(f"⚠ RapidOCR 不可用，本次全程走图片路径：{reason}")
            return False
        # 预热失败不影响识别，但说明输入有问题 —— 记一笔，别让它静悄悄消失。
        warm_error = getattr(self._ocr, "warm_error", "")
        if warm_error:
            self._log(f"OCR 预热失败（不影响识别）：{warm_error}")
        return True

    def gate_summary(self) -> str:
        """本局门控统计。跑完打印一次，用来判断阈值定得合不合适。"""
        total = self.ocr_accepted + self.ocr_fallback
        if not total:
            return "OCR 快路径：本局未触发"
        summary = (f"OCR 快路径：命中 {self.ocr_accepted}/{total}，"
                   f"回退图片 {self.ocr_fallback} 次")
        if self._fallback_reasons:
            # 回退原因按出现次数排，最多列两条 —— 再多终端也读不完。
            parts = [
                name if count == 1 else f"{name} ×{count}"
                for name, count in self._fallback_reasons.most_common(2)
            ]
            summary += f"（原因：{'、'.join(parts)}）"
        return summary

    def read_question_visual(self, image: Image.Image, prompt: str | None = None, *,
                             timeout: float | None = None, deadline: float | None = None) -> VisionQuestion:
        """只走图片路径，不重复本地 OCR；转发单次超时和绝对 deadline。"""
        return self._vision.read_question(image, prompt=prompt, timeout=timeout,
                                              **({"deadline": deadline} if deadline is not None else {}))

    def read_question_with_budget(self, image: Image.Image, prompt: str | None = None, *,
                                  budget: PKQuestionBudget) -> VisionQuestion:
        """答题循环专用：共享截图预算，使本地倒计时也约束模型及点击校验。"""
        return self.read_question(image, prompt=prompt, deadline=budget.deadline, budget=budget)

    def read_question(self, image: Image.Image, prompt: str | None = None, *,
                      timeout: float | None = None, deadline: float | None = None,
                      budget: PKQuestionBudget | None = None) -> VisionQuestion:
        """识别一道题并判答案。"""
        self.last_question_local_read = None
        self.last_question_progress_text = ""
        self.last_question_local_finished_at = None
        self.last_question_model_started_at = None
        def observe(read):
            self.last_question_progress_text = getattr(read, "progress_text", "")
            if budget is not None:
                budget.observe(read)
                if budget.countdown_seconds is not None:
                    self._log(f"倒计时预算：截图剩余 {budget.countdown_seconds}s，"
                              f"当前识别余量 {max(0.0, budget.deadline - time.monotonic()):.2f}s（已预留复核及点击时间）")
                return budget.deadline
            return deadline

        local = read_local(self._ocr, image, deadline=deadline, observe=observe)
        deadline = local.deadline
        read, decision = local.read, local.decision
        if decision.accepted:
            self.last_question_local_read = read
        strip_ran, strip_options = local.strip_ran, local.strip_options
        if read is not None and read.progress_text:
            self.last_question_progress_text = read.progress_text
        if self.last_question_progress_text:
            self.last_question_local_finished_at = time.monotonic()
        # 文本路径实际耗了多少秒。0.0 表示压根没走文本路径（门控直接拒绝），
        # 那时预算是满的，照常回退图片路径。
        text_elapsed = 0.0

        def answer():
            nonlocal decision, text_elapsed
            if not decision.accepted or read is None:
                return None
            started = time.monotonic()
            self.last_question_model_started_at = started
            question = self._answer_from_text(read, timeout=_text_timeout(timeout, self.text_timeout),
                                              deadline=deadline)
            text_elapsed = time.monotonic() - started
            if question is not None:
                chosen = read.options[question.answer_index]
                self.ocr_accepted += 1
                reason = self._merge_reason(decision, strip_ran=strip_ran,
                                            strip_options=strip_options)
                self._log(
                    f"OCR 快路径 置信度 {[round(c, 3) for c in read.confidences]}"
                    f" 最低 {read.min_confidence:.2f} → {read.stem} → {chosen}"
                    f"（{reason or '整图路'}，模型 {text_elapsed * 1000:.0f}ms）"
                )
            else:
                decision = decision.__class__(False, "OCR 路径模型没给出合法下标", read)
            return question

        def visual():
            if not getattr(self._vision, "supports_image", True):
                raise AnswerUnavailable("no_image_capability")
            # 只有真正发出图片请求才计入回退次数。
            self.ocr_fallback += 1
            reason = self._merge_reason(decision, strip_ran=strip_ran,
                                        strip_options=strip_options)
            if text_elapsed:
                reason = f"{reason}（文本路径已耗时 {text_elapsed * 1000:.0f}ms）"
            self._log(f"图片路径 回退原因：{reason}")
            self._fallback_reasons[_short_reason(reason)] += 1
            if self.last_question_model_started_at is None:
                self.last_question_model_started_at = time.monotonic()
            _check_deadline(deadline)
            question = self._vision.read_question(
                image, prompt=prompt, timeout=timeout,
                **({"deadline": deadline} if deadline is not None else {}),
            )
            _check_deadline(deadline)
            return question

        return text_then_image(answer, visual, deadline=deadline,
                               minimum_image_budget=_IMAGE_PATH_MIN_BUDGET,
                               log=self._log)

    def read_question_local(self, image: Image.Image) -> OcrRead:
        """纯本地 OCR 与门控；判否抛 OcrRejected 并携带题号证据，不请求模型。"""
        read, decision, strip_ran, strip_options = self._read_local(image)
        if not decision.accepted or read is None:
            raise OcrRejected(
                self._merge_reason(decision, strip_ran=strip_ran,
                                   strip_options=strip_options)
                or "OCR 门控未通过",
                read=read,
            )
        return read

    def _read_local(self, image: Image.Image):
        """编排整图与逐按钮补读；判否保留题号证据、缺失下标及有效选项数。"""
        local = read_local(self._ocr, image)
        return local.read, local.decision, local.strip_ran, local.strip_options

    def answer_from_ocr_read(self, read: OcrRead, *,
                             timeout: float | None = None, deadline: float | None = None) -> VisionQuestion | None:
        """用**已经读好的** OcrRead 走文本路径，只要下标。"""
        return self._answer_from_text(read, timeout=timeout, deadline=deadline)

    def try_read_question(self, image: Image.Image, prompt: str | None = None, *,
                          timeout: float | None = None) -> VisionQuestion:
        """只走 OCR 文本快路径；本地判否抛 OcrRejected，模型未给答案抛 AnswerUnavailable。"""
        read, decision, strip_ran, strip_options = self._read_local(image)
        if not decision.accepted or read is None:
            # **本地门控判否** —— 这一帧本地就读不出。
            #
            # 注意这与下面的「模型没给出合法下标」是**两回事**：
            # 这里是**一次网络请求都没发**，纯本地判定。
            raise OcrRejected(
                self._merge_reason(decision, strip_ran=strip_ran,
                                   strip_options=strip_options)
                or "OCR 门控未通过"
            )
        question = self._answer_from_text(read, timeout=timeout)
        if question is None:
            raise AnswerUnavailable(
                "OCR 路径模型没给出合法下标（文本路径已发出请求但未取到答案）"
            )
        chosen = read.options[question.answer_index]
        self.ocr_accepted += 1
        self._log(
            f"OCR 快路径 置信度 {[round(c, 3) for c in read.confidences]}"
            f" 最低 {read.min_confidence:.2f} → {read.stem} → {chosen}"
            f"（{decision.reason or '整图路'}）"
        )
        return question

    def _log(self, message: str) -> None:
        """把门控结论写进 run.log —— 这是校准阈值的唯一数据来源。"""
        if self._log_sink is not None:
            try:
                self._log_sink(message)
            except Exception:
                pass  # 日志失败不能影响答题。

    @staticmethod
    def _merge_reason(decision, *, strip_ran: bool,
                      strip_options: int = 0) -> str:
        """补读执行但无结果时，不能仍将其记录为未运行。"""
        reason = decision.reason or ""
        if not strip_ran:
            return reason
        if "逐按钮补读未运行" in reason:
            return reason.replace(
                "逐按钮补读未运行",
                f"逐按钮补读未通过（读出 {strip_options} 个非空选项）",
            )
        if "拼接路未运行" not in reason:
            return reason
        if strip_options:
            return reason.replace(
                "拼接路未运行",
                f"逐按钮补读读出了 {strip_options} 个选项，但题干没读出",
            )
        return reason.replace("拼接路未运行", "逐按钮补读也未读出")

    def _answer_from_text(self, read: OcrRead, *,
                          timeout: float | None = None, deadline: float | None = None) -> VisionQuestion | None:
        """把 OCR 文本发给模型，文本阶段（含复核）共用限时，超时后允许图片接手。"""
        supports_image = getattr(self._vision, "supports_image", True)
        started = time.monotonic()
        limit = text_path_timeout(_text_timeout(timeout, self.text_timeout), deadline,
                                  supports_image=supports_image)
        if limit <= 0:
            self._log("文本路径 跳过：剩余预算仅够图片回退")
            return None
        text_deadline = deadline
        if supports_image and deadline is not None:
            text_deadline = min(deadline, started + limit)
        try:
            return answer_from_read(self._vision, read, TEXT_PROMPT,
                                    timeout=limit, deadline=text_deadline, log=self._log)
        except RecognitionTimeout:
            # 文本阶段到时不等于整题到时；图片仍使用原截止时间。
            _check_deadline(deadline)
            return None

    def __getattr__(self, name: str):
        """其余属性全部转给 VisionService（usage、read_result、close…）。"""
        return getattr(self._vision, name)

    def __enter__(self) -> "HybridQuestionReader":
        self._vision.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._vision.__exit__(*exc_info)
