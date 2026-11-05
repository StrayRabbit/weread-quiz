"""统一的模型入口 —— 主模型（文本路径）+ 可选的视觉模型（整图路径）。

**两种模型都支持，谁也不会把谁顶掉。**

* **主模型**（MODEL_*）—— 文本路径（判下标）用它。协议（OpenAI 兼容 /
  System One）由代码自己认，用户在配置里看不见。
* **视觉模型**（VISION_*，可选）—— 整图路径（读截图）用它。没配就用主模型，
  前提是主模型能读图 —— 决策模型读不了，那种配置下整图路径明确报不可用。

**为什么值得配两个** —— 实测 69 道带标准答案的真实题（同一份 OCR 文本）：

===========================  ==========  ==========
模型                          答对         时延中位
===========================  ==========  ==========
DeepSeek-flash（视觉）        69/69       1094ms
阿里 decision-model-preview    59/69        78ms
===========================  ==========  ==========

决策模型快 14 倍，但会「抓最近邻」（问「上一句是」答下一句）。
**好在它拿不准的时候自己知道** —— 答错的 10 道里 9 道置信度 ≤ 0.76。

所以两个都配时：文本路径先问决策模型（快），置信度低于复核门槛
（默认 0.8）就问视觉模型复核（准）。这样答对率 85.5% → 98.6%，平均
时延约 390ms；只配视觉模型是 100% / 1094ms，只配决策模型是 85.5% / 78ms。
"""

from __future__ import annotations

import time
from typing import Callable

from .decision import DecisionService
from .errors import AnswerUnavailable
from .usage import UsageTracker
from .vision import VisionService

PROTOCOL_CHAT = "chat"
PROTOCOL_SYSTEMONE = "systemone"

# 模型名里出现这些片段，就按 System One 走。
_SYSTEMONE_NAME_HINTS = ("decision-model", "jev-", "systemone")

# 复核门槛的默认值 —— 只在"主模型是决策模型且配了视觉模型"时生效。
#
# 实测 69 道真实题：决策模型答错的 10 道里 9 道置信度 ≤ 0.76。
# 门槛 0.8 → 30% 的题转给视觉模型复核，答对率 85.5% → 98.6%。
_DEFAULT_REVIEW_THRESHOLD = 0.8


def detect_protocol(base_url: str, model: str) -> str:
    """判断这个端点该用哪个协议。

    两条判据：地址本身就是 System One 的（阿里百炼的决策模型要走
    ``/compatible-mode/v1/systemone``，Jev 直接给的就是），或者模型名像决策模型。
    """
    url = (base_url or "").rstrip("/").lower()
    name = (model or "").strip().lower()
    if "systemone" in url:
        return PROTOCOL_SYSTEMONE
    if any(hint in name for hint in _SYSTEMONE_NAME_HINTS):
        return PROTOCOL_SYSTEMONE
    return PROTOCOL_CHAT


class ModelService:
    """模型服务门面 —— 文本路径和整图路径都走它，协议在构造时就定好了。

    **主模型**负责文本路径（判下标），**视觉模型**负责整图路径（读截图）。
    只配一个模型时主模型兼两份工作 —— 前提是它能读图。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float | None = None,
        retries: int = 1,
        thinking: bool = True,
        recover_truncated_response: bool = True,
        review_threshold: float | None = None,
        timing_sink: Callable[[str], None] | None = None,
        image_base_url: str | None = None,
        image_api_key: str | None = None,
        image_model: str | None = None,
    ) -> None:
        self.protocol = detect_protocol(base_url, model)
        self.usage = UsageTracker()
        self.last_question_usage = self.usage.snapshot()
        self._timing_sink = timing_sink
        self._text: object = None
        self._vision: VisionService | None = None
        self._image: VisionService | None = None

        if self.protocol == PROTOCOL_SYSTEMONE:
            self._text = DecisionService(
                base_url, api_key, model,
                timeout=timeout if timeout is not None else 11.0,
                # 门槛由门面统一处理（复核 or 直接用）—— 这里不丢答案，
                # 否则"没有视觉模型可复核"时会把好答案扔掉去蒙。
                min_confidence=0.0,
                usage=self.usage,
                timing_sink=timing_sink,
            )
        else:
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._vision = VisionService(
                base_url, api_key, model,
                retries=retries,
                thinking=thinking,
                recover_truncated_response=recover_truncated_response,
                usage=self.usage,
                timing_sink=timing_sink,
            )
            self._text = self._vision

        # 整图路径：配了视觉模型就用它；没配就看主模型能不能读图。
        if image_base_url and image_api_key and image_model:
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._image = VisionService(
                image_base_url, image_api_key, image_model,
                retries=retries,
                thinking=thinking,
                recover_truncated_response=recover_truncated_response,
                usage=self.usage,
                timing_sink=timing_sink,
            )
        elif self._vision is not None:
            self._image = self._vision

        if (self._image is not None
                and detect_protocol(image_base_url or "", image_model or "")
                == PROTOCOL_SYSTEMONE):
            # 视觉模型填成了决策模型 —— 它读不了图，整图路径必然失败。
            # 说一声，别让用户对着"读不了"猜配置。
            if timing_sink is not None:
                timing_sink(
                    f"⚠ 视觉模型 {image_model} 看起来是决策模型 —— "
                    f"它读不了图，整图路径会失败"
                )

        self._review_threshold = self._resolve_review_threshold(review_threshold)

    # ------------------------------------------------------------- 能力

    @property
    def supports_structured_choice(self) -> bool:
        """模型要的是结构化的题干+选项（criteria），还是拼好的 prompt。

        决策模型要结构化的；普通视觉模型要 prompt。
        """
        return self.protocol == PROTOCOL_SYSTEMONE

    @property
    def supports_image(self) -> bool:
        """能不能读图 —— 整图兜底靠它。"""
        return self._image is not None

    @property
    def review_threshold(self) -> float:
        """复核门槛 —— 0 表示不复核。"""
        return self._review_threshold

    # ------------------------------------------------------------- 文本路径

    def choose_index(self, stem: str, options: list[str], *,
                     prompt: str | None = None,
                     timeout: float | None = None,
                     deadline: float | None = None) -> int | None:
        """文本路径判下标 —— 主模型先答，拿不准就问视觉模型复核。

        主模型是决策模型时，置信度低于复核门槛就转给视觉模型（实测能救回
        13% 的题）。**没有视觉模型时，决策模型的答案再不确定也比蒙一个好**，
        所以那种配置下直接用它的答案。
        """
        request: dict = {}
        if timeout is not None:
            request["timeout"] = timeout
        if deadline is not None:
            request["deadline"] = deadline

        if not self.supports_structured_choice:
            return self._measure(
                self._text.answer_from_text,
                self._format(prompt, stem, options), len(options), **request,
            )

        index = self._measure(
            self._text.answer_options, stem, list(options), **request,
        )
        confidence = getattr(self._text, "last_confidence", None)
        if index is not None and not self._need_review(confidence):
            return index
        if self._image is None:
            # 没有视觉模型可复核 —— 拿不准也用决策模型的答案，别蒙。
            return index
        if self._timing_sink is not None:
            shown = "未知" if confidence is None else f"{confidence:.2f}"
            self._timing_sink(
                f"决策模型置信度 {shown} 低于复核门槛 "
                f"{self._review_threshold:.2f} → 转给视觉模型复核"
            )
        reviewed = self._measure(
            self._image.answer_from_text,
            self._format(prompt, stem, options), len(options), **request,
        )
        if reviewed is not None:
            return reviewed
        # 复核没答出来（超时、解析失败…）—— 退回决策模型的答案。
        # 丢掉它去蒙一个，期望只有 25% 对；留着它至少有 85%。
        return index

    def answer_options(self, stem: str, options: list[str], **kwargs):
        """结构化判下标（System One 协议）。"""
        if not self.supports_structured_choice:
            raise AnswerUnavailable("模型不支持结构化判下标")
        return self._text.answer_options(stem, options, **kwargs)

    def answer_from_text(self, prompt: str, option_count: int, **kwargs):
        """拼好的 prompt 判下标（chat 协议）。"""
        return self._text.answer_from_text(prompt, option_count, **kwargs)

    def probe(self, deadline: float):
        """发一个最小的判下标请求 —— 确认端点通、模型名对、连接也建好了。"""
        if self.supports_structured_choice:
            return self.answer_options("计算 1+1", ["2", "3"], timeout=8.0,
                                       deadline=deadline)
        return self.answer_from_text(
            '计算 1+1。选项：0. 2；1. 3。只输出 {"answer_index":0}。',
            2, max_tokens=512, timeout=8.0, deadline=deadline,
        )

    # ------------------------------------------------------------- 整图路径

    def read_question(self, image, prompt=None, **kwargs):
        """读图并判答案。没有读图能力的模型明确报不可用。"""
        if self._image is None:
            raise AnswerUnavailable("no_image_capability")
        if prompt is not None:
            kwargs["prompt"] = prompt
        return self._measure(self._image.read_question, image, **kwargs)

    def recognize(self, image, prompt=None, **kwargs):
        if self._image is None:
            raise AnswerUnavailable("no_image_capability")
        if prompt is not None:
            kwargs["prompt"] = prompt
        return self._measure(self._image.recognize, image, **kwargs)

    def read_result_detail(self, image):
        """结算页判读。读不了图就没有这一步。"""
        if self._image is None:
            raise AnswerUnavailable("no_image_capability")
        return self._measure(self._image.read_result_detail, image)

    # ------------------------------------------------------------- 杂项

    def warmup_client(self) -> float:
        """预热两个客户端（只建连接，不请求模型）。"""
        started = time.perf_counter()
        for service in self._services():
            try:
                service.warmup_client()
            except Exception:  # noqa: BLE001 - 预热失败不该影响答题
                pass
        return time.perf_counter() - started

    def close(self) -> None:
        for service in self._services():
            try:
                service.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "ModelService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __getattr__(self, name: str):
        """其余属性转给底层服务（warmup_client、retries…）。"""
        text = self.__dict__.get("_text")
        if text is None:
            raise AttributeError(name)
        return getattr(text, name)

    # ------------------------------------------------------------- 内部

    def _services(self) -> list:
        """要去重的服务列表（主模型和视觉模型可能是同一个）。"""
        services = []
        for service in (self._text, self._image):
            if service is None:
                continue
            if any(service is existing for existing in services):
                continue
            services.append(service)
        return services

    def _measure(self, call, *args, **kwargs):
        """跑一次请求，并把这一次的 token 用量记到门面上。"""
        start = self.usage.snapshot()
        try:
            return call(*args, **kwargs)
        finally:
            self.last_question_usage = self.usage.since(start)

    def _need_review(self, confidence) -> bool:
        """这次判定要不要转给视觉模型复核。"""
        if self._review_threshold <= 0:
            return False
        if confidence is None:
            # 没有置信度 —— 拿不准。
            return True
        return confidence < self._review_threshold

    def _resolve_review_threshold(self, value: float | None) -> float:
        """复核门槛。

        * ``None``（没配）= 自动：配了视觉模型就用 0.8，没配就不复核。
        * ``0`` = 明确不复核，决策模型的答案直接用。
        * ``>0`` = 用这个值。
        """
        if value is not None:
            return float(value)
        if self._image is not None and self.protocol == PROTOCOL_SYSTEMONE:
            return _DEFAULT_REVIEW_THRESHOLD
        return 0.0

    @staticmethod
    def _format(prompt: str | None, stem: str, options: list[str]) -> str:
        """视觉模型要的是拼好的题干加选项。"""
        listing = "\n".join(f"{index}. {text}" for index, text in enumerate(options))
        if not prompt:
            return f"题目：{stem}\n\n选项：\n{listing}"
        return prompt % (stem, listing)


def build_model_service(cfg, *, timeout: float | None = None,
                        retries: int | None = None,
                        recover_truncated_response: bool = True,
                        timing_sink: Callable[[str], None] | None = None) -> ModelService:
    """按配置构造模型服务。

    主模型只有一组（协议由 :func:`detect_protocol` 自己认）；视觉模型是
    可选的第二组，配了就用，没配就由主模型兼任（前提是它能读图）。
    """
    cfg.validate_model()
    return ModelService(
        cfg.model_base_url,
        cfg.model_api_key,
        cfg.model_name,
        timeout=timeout,
        retries=cfg.vision_retries if retries is None else retries,
        thinking=cfg.model_thinking,
        recover_truncated_response=recover_truncated_response,
        review_threshold=getattr(cfg, "model_min_confidence", None),
        timing_sink=timing_sink,
        image_base_url=getattr(cfg, "vision_base_url", "") or None,
        image_api_key=getattr(cfg, "vision_api_key", "") or None,
        image_model=getattr(cfg, "vision_name", "") or None,
    )
