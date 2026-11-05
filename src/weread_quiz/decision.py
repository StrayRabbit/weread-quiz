"""决策模型（System One）适配层。

决策模型不是聊天模型 —— 没有 messages、没有 max_tokens、不生成文本。
请求是「业务状态 + 结构化问题」，返回是「选中的选项 + 概率分布 + 置信度」。

两个供应商的协议几乎一样，差别只有两处：

* 端点：阿里百炼是 ``{workspace}.<region>.maas.aliyuncs.com/compatible-mode/v1``，
  Jev 是自己的 ``/v1`` —— 两者都在 ``/systemone`` 收请求。
* 响应外壳：阿里把答案放在顶层，Jev 放在 ``answers`` 下。

**它只能判下标，不能读题。** 本地 OCR 门控不通过时那张整图仍然要交给
视觉模型 —— 那条兜底路径（约 5%）不能省，所以决策模型是"文本路径的替代品"，
不是"整个模型层的替代品"。
"""

from __future__ import annotations

import time
from typing import Callable

import httpx

from .errors import AnswerUnavailable, RecognitionTimeout
from .usage import UsageTracker
from .vision import _check_deadline, _shared_ssl_context

# 判下标的问题名 —— 只用于在响应里定位答案，不传给模型。
_QUESTION = "answer"

# instructions 是"问什么"，criteria 是"有哪些选项"。
_INSTRUCTIONS = "选出这道题唯一正确的选项。"

# 单次调用超时上限。决策模型实测在百毫秒量级，这里给的是上限而不是预期值。
_DEFAULT_TIMEOUT = 11.0


class DecisionService:
    """调用决策模型判下标。"""

    # 它要的是结构化的题干+选项（criteria），不是拼好的 prompt ——
    # recognition 的接缝靠这个标记决定走哪条路。
    supports_structured_choice = True

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT,
        *,
        min_confidence: float = 0.0,
        usage: UsageTracker | None = None,
        timing_sink: Callable[[str], None] | None = None,
    ) -> None:
        # 端点可以给到 /v1（适配层自己拼 /systemone），也可以直接给完整的
        # /systemone —— 用户照着文档抄容易抄全，拼成 /systemone/systemone。
        base = base_url.rstrip("/")
        self._url = base if base.endswith("/systemone") else f"{base}/systemone"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._timeout = timeout
        # 置信度门槛 —— 低于它当作没答出来，交给图片路径。
        # 默认 0（不设门槛）：供应商文档给的示例阈值偏严，照抄会让通过率接近 0，
        # 阈值得用自己的样本校准。
        self._min_confidence = min_confidence
        # 用量计数器默认独立，但可以和视觉模型共用同一个 ——
        # 「第 N 题 token」「本局」那几行都从视觉模型读，共用就不用改统计口径。
        self.usage = usage if usage is not None else UsageTracker()
        self.last_question_usage = self.usage.snapshot()
        self._client: httpx.Client | None = None
        self._timing_sink = timing_sink
        # 最近一次判定的置信度 —— 调阈值时要看它。
        self.last_confidence: float | None = None

    # ------------------------------------------------------------------ 连接

    def _get_client(self) -> httpx.Client:
        """惰性创建并复用 httpx 客户端；SSL context 与视觉模型共用。"""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout, verify=_shared_ssl_context()
            )
        return self._client

    def warmup_client(self) -> float:
        """预先构造客户端，返回本地耗时；不发送网络请求。"""
        started = time.perf_counter()
        try:
            self._get_client()
        except Exception:  # noqa: BLE001 - 预热失败不该影响答题
            return 0.0
        return time.perf_counter() - started

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "DecisionService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------ 请求

    def answer_options(
        self,
        stem: str,
        options: list[str],
        *,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> int | None:
        """给题干和候选选项，只要下标。

        失败返回 None（交给图片路径）；deadline 超时向外传播。
        """
        if not options:
            return None
        started = time.monotonic()
        start = self.usage.snapshot()
        try:
            payload = self._build_payload(stem, options)
            data = self._post(payload, timeout=timeout, deadline=deadline)
        except RecognitionTimeout:
            raise
        except Exception:
            return None
        finally:
            self.last_question_usage = self.usage.since(start)

        index, confidence = self._parse(data, len(options))
        self.last_confidence = confidence
        if index is None:
            return None
        if confidence is not None and confidence < self._min_confidence:
            if self._timing_sink is not None:
                self._timing_sink(
                    f"决策模型置信度 {confidence:.2f} < 门槛 {self._min_confidence:.2f}，"
                    f"按没答出来处理"
                )
            return None
        elapsed_ms = (time.monotonic() - started) * 1000
        if self._timing_sink is not None:
            self._timing_sink(
                f"决策模型 {elapsed_ms:.0f}ms 置信度 "
                f"{'未知' if confidence is None else f'{confidence:.2f}'} → 选项 {index}"
            )
        return index

    def _build_payload(self, stem: str, options: list[str]) -> dict:
        """组装请求体。

        criteria 的键是选项下标 —— 响应里 ``choice`` 回的就是这个键，
        数字键还能避开选项文字重复时的歧义。
        """
        listing = "\n".join(f"{index}. {text}" for index, text in enumerate(options))
        return {
            "model": self._model,
            "state": f"题目：{stem}\n\n选项：\n{listing}",
            "questions": {
                _QUESTION: {
                    "type": "choice",
                    "instructions": _INSTRUCTIONS,
                    "criteria": {str(i): text for i, text in enumerate(options)},
                }
            },
        }

    def _post(self, payload: dict, *, timeout: float | None = None,
              deadline: float | None = None) -> dict:
        """发出一次请求并返回解析后的 JSON。"""
        _check_deadline(deadline)
        client = self._get_client()
        options = self._timeout_option(timeout, deadline=deadline)
        resp = client.post(self._url, json=payload, headers=self._headers, **options)
        try:
            data = resp.json()
        except Exception:
            data = None
        finally:
            self.usage.record(_normalize_usage(data))
        _check_deadline(deadline)
        if resp.status_code >= 400:
            raise AnswerUnavailable(f"决策模型 API 返回 {resp.status_code}")
        if not isinstance(data, dict):
            raise AnswerUnavailable("决策模型响应不是 JSON 对象")
        return data

    def _timeout_option(self, timeout: float | None, *,
                        deadline: float | None = None) -> dict:
        """把超时压到 deadline 剩余时间里。"""
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RecognitionTimeout("deadline_exceeded")
            if timeout is None or remaining < timeout:
                return {"timeout": httpx.Timeout(remaining)}
        if timeout is not None:
            return {"timeout": httpx.Timeout(timeout)}
        return {}

    def _parse(self, data: dict, option_count: int) -> tuple[int | None, float | None]:
        """取出下标和置信度。

        阿里把答案放在响应顶层，Jev 放在 ``answers`` 下 —— 两种都认。
        """
        answers = data.get("answers")
        if not isinstance(answers, dict):
            answers = data
        answer = answers.get(_QUESTION)
        if not isinstance(answer, dict):
            return None, None

        confidence = answer.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None

        choice = answer.get("choice")
        index: int | None = None
        if isinstance(choice, bool):
            index = None
        elif isinstance(choice, int):
            index = choice
        elif isinstance(choice, str):
            stripped = choice.strip()
            index = int(stripped) if stripped.lstrip("-").isdigit() else None
        if index is None or not (0 <= index < option_count):
            return None, confidence
        return index, confidence


def _normalize_usage(data) -> dict | None:
    """把决策模型的用量字段名归一成视觉模型那套。

    Jev 用 ``input_tokens`` / ``output_tokens``，阿里和 OpenAI 兼容层用
    ``prompt_tokens`` / ``completion_tokens`` —— 用量统计只认后者。
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    normalized = dict(raw)
    if "prompt_tokens" not in normalized and "input_tokens" in normalized:
        normalized["prompt_tokens"] = normalized["input_tokens"]
    if "completion_tokens" not in normalized and "output_tokens" in normalized:
        normalized["completion_tokens"] = normalized["output_tokens"]
    return normalized
