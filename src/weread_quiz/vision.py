"""模型服务：图片识题、文本判下标、响应解析、请求预算和连接复用。"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from typing import Callable

import httpx
from PIL import Image

from .errors import AnswerUnavailable, RecognitionTimeout
from .request_timing import RequestTiming
from .usage import UsageTracker

# 进程内共享的 SSL context（见 _shared_ssl_context 的说明）。
#
# 建它要 ~365ms（读 CA 证书包），而 httpx 建 Client 时会建 3 个
# transport、各建一次 —— 也就是 3 × 365ms ≈ 1.1 秒白花。
_ssl_context = None

# 单次模型调用的超时上限。
#
# 取 11 秒而不是 12：超时之后还要留点击链的时间（点击前 fresh 校验
# ~50ms + 点击 ~265ms + pyautogui pause 150ms ≈ 0.5 秒），11.5 秒的
# 边界会让"刚好在 11.4 秒返回"这种情况点击时倒计时已经走完。
#
# daily 路径不受影响 —— 它有自己更宽的 DAILY_VISION_TIMEOUT=11.5。
_TIMEOUT_SECONDS = 11.0

# 重试前的等待。模型单次调用 2.9~5.8 秒，等 0.5 秒几乎不占预算。
_RETRY_BACKOFF_SECONDS = 0.5

# 读图的 token 预算。
#
# **思考模型会把预算吃光** —— 实测 deepseek-flash 读图时推理部分就用了
# 800~2700 token，预算给 1200 会有 17% 的图返回空内容（finish_reason=length）。
# 给到 2000 起步，返回空时翻倍重试（同一个请求原样重发还是空）。
_IMAGE_MAX_TOKENS = 2000
_MAX_TOKENS_CEILING = 8000

_QUESTION_NO_BAND = (250, 300)

PROMPT = """这是微信读书每日一答小程序的一个界面截图。

先判断这是哪种界面：
- question：正在答题。有倒计时（如 00:08）、"第 N 题 · 共 M 题"、题目卡片、若干选项按钮。
- matching：匹配等待。有"邀请好友""自动匹配""取消匹配"这类按钮，没有倒计时和题目卡片。
- other：其他（结算页、排行榜、首页等）。

如果是 question，只输出 JSON，不要解释：
{"screen":"question","stem":"题干文字","options":["选项1","选项2"],"answer":正确选项在 options 数组中的下标}

如果不是 question，只输出：
{"screen":"matching"}

规则：
- options 按画面里从上到下的顺序排列
- 题干里的填空横线、下划线照原样保留
- answer 从 0 开始计数
- 不要输出 Markdown 代码块标记"""

# PK 用的精简 prompt。
#
# 与 PROMPT 的区别只有两处：删掉冗长的规则说明、压缩 JSON 示例。
# **必须保留 screen 分类** —— 这不是可选项。
#
# 速度收益有限：thinking 关掉后完整 PROMPT 本就只要 1.19s，此版本
# 1.46s，差异在噪声范围内。真正的提速来自 MODEL_THINKING=0。
PROMPT_PK = """这是微信读书每日一答/问答 PK 的界面截图。先判断界面类型：
- question：正在答题，有题目卡片和若干选项按钮
- matching：匹配等待，有"邀请好友""自动匹配""取消匹配"这类按钮，没有题目卡片
- other：其他（结算页、首页等）

如果是 question，只输出 JSON：
{"screen":"question","stem":"题干文字","options":["选项1","选项2"],"answer":正确选项在options中的下标}
options 按画面从上到下排列，answer 从 0 开始计数。
如果不是 question，只输出 {"screen":"matching"} 或 {"screen":"other"}。
只输出 JSON，不要解释，不要 Markdown 代码块标记。"""


@dataclass
class VisionQuestion:
    """模型从截图里读出的一道题。"""

    stem: str
    options: list[str]
    answer_index: int | None = None
    raw_text: str = ""
    # 界面类型：question / matching / other
    screen: str = "question"
    # 模型没按 JSON 格式、直接回了一个序号。
    # 这时 options 是空的，只有 answer_index 可信。
    bare_index: bool = False
    # 题干和选项是**本地 OCR 读出来的**，模型只回了一个下标。
    #
    local_read: bool = False
    # 界面自己写的题号（「第N题·共12题」里的 N）。读不出为 None。
    #
    # **读不出就是 None，绝不猜** —— 猜错比读不出更糟。
    progress_no: int | None = None


def make_non_question(screen: str, raw_text: str = "") -> VisionQuestion:
    return VisionQuestion(
        stem="", options=[], answer_index=None, raw_text=raw_text, screen=screen
    )


def is_question(vision: "VisionQuestion") -> bool:
    return vision.screen == "question" and (bool(vision.options) or vision.bare_index)


def image_to_jpeg_data_url(image: Image.Image, quality: int = 80) -> str:
    """PIL 图片 → base64 data URL。"""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def strip_think(text: str) -> str:
    """剥掉 <think>...</think> 思考块。"""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def _tail_after_think(text: str) -> str:
    """内容里没有 </think> 时，取末尾像答案的那一段。"""
    text = text.strip()
    if not text:
        return ""
    if "{" in text:
        start = text.rfind("{")
        end = text.rfind("}")
        if end > start:
            return text[start : end + 1]
    # 没有 JSON，退而求其次找最后一个独立数字
    matches = re.findall(r"(?<!\d)([1-9])(?!\d)", text)
    return matches[-1] if matches else ""


def _repair_json(payload: str) -> dict | None:
    """修补模型输出的非法 JSON。"""
    options: list[str] = []
    bracket = payload.find("[")
    closing = payload.rfind("]")
    if bracket >= 0 and closing > bracket:
        for match in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', payload[bracket:closing]):
            options.append(match.group(1))

    answer = None
    answer_match = re.search(r'"answer"\s*:\s*(-?\d+)', payload)
    if answer_match:
        answer = int(answer_match.group(1))

    screen = ""
    screen_match = re.search(r'"screen"\s*:\s*"(\w+)"', payload)
    if screen_match:
        screen = screen_match.group(1)

    if not options and answer is None and not re.search(r'"screen"', payload):
        return None

    stem = ""
    stem_key = payload.find('"stem"')
    if stem_key >= 0:
        colon = payload.find(":", stem_key)
        end_of_stem = bracket if bracket > colon else len(payload)
        raw = payload[colon + 1 : end_of_stem].strip().rstrip(",").strip()
        # 截到下一个 JSON 键为止（未转义引号把后续内容吞进来了）
        next_key = re.search(r'"\s*(?:options|answer|index|total|no)\s*"', raw)
        if next_key:
            raw = raw[: next_key.start()]
        stem = raw.strip().strip('"').replace('\\"', '"').strip()
        # 尾部残留的逗号和杂散引号
        stem = stem.rstrip('",').strip().strip('"').strip()

    return {"stem": stem, "options": options, "answer": answer, "screen": screen}


class _InvalidVisionResponse(AnswerUnavailable):
    """保留兼容的异常正文，另带不含回复原文的结构错误码。"""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def parse_vision_response(text: str) -> VisionQuestion:
    """从模型返回里解析出 VisionQuestion。解析不出就抛 AnswerUnavailable。"""
    if not text or not text.strip():
        raise _InvalidVisionResponse("empty_response", "模型返回为空")

    cleaned = strip_think(text)
    # 去掉 Markdown 代码块围栏
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    bare = cleaned.strip()
    if bare.isdigit() and 1 <= int(bare) <= 8:
        index = int(bare) - 1
        return VisionQuestion(
            stem="",
            options=[],
            answer_index=index,
            raw_text=text,
            screen="question",
            bare_index=True,
        )

    if cleaned.startswith("[") and cleaned.endswith("]"):
        raise _InvalidVisionResponse("invalid_shape", "模型返回了 JSON 数组而不是对象")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise _InvalidVisionResponse("missing_json", f"模型返回里找不到 JSON：{text[:120]!r}")

    payload = cleaned[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        repaired = _repair_json(payload)
        if repaired is None:
            raise _InvalidVisionResponse(
                "invalid_json", f"模型返回的 JSON 解析失败：{exc}；原文：{text[:160]!r}"
            ) from exc
        data = repaired

    screen = str(data.get("screen") or "question").strip().lower()
    if screen != "question":
        return make_non_question(screen or "other", text)

    stem = str(data.get("stem", "")).strip()
    if not stem:
        reason = "empty_stem" if "stem" in data else "missing_stem"
        raise _InvalidVisionResponse(reason, f"模型没返回题干：{text[:160]!r}")

    raw_options = data.get("options") or []
    if not isinstance(raw_options, list) or not raw_options:
        reason = ("missing_options" if "options" not in data else
                  "empty_options" if isinstance(raw_options, list) else "invalid_options")
        raise _InvalidVisionResponse(reason, f"模型没返回选项：{text[:160]!r}")
    options = [str(o).strip() for o in raw_options]
    if any(not o for o in options):
        raise _InvalidVisionResponse("empty_option", f"模型返回了空选项：{options!r}")

    answer_index = data.get("answer")
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        answer_index = None
    elif not (0 <= answer_index < len(options)):
        answer_index = None

    return VisionQuestion(
        stem=stem,
        options=options,
        answer_index=answer_index,
        raw_text=text,
        screen="question",
    )


def _shared_ssl_context():
    """进程内复用系统证书 SSL context，所有 HTTP 客户端共享。"""
    global _ssl_context
    if _ssl_context is None:
        _ssl_context = httpx.create_ssl_context()
    return _ssl_context


def _is_minimax_endpoint(base_url: str, model: str) -> bool:
    """按模型名及端点识别 MiniMax，仅该端点发送其私有 thinking 扩展。"""
    return "minimax" in base_url.lower() or model.lower().startswith("minimax")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise RecognitionTimeout("deadline_exceeded")


class VisionService:
    """调用视觉模型识别题目并判答案。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = _TIMEOUT_SECONDS,
        retries: int = 1,
        *,
        recover_truncated_response: bool = True,
        thinking: bool = True,
        enabled: bool = True,
        usage: UsageTracker | None = None,
        timing_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        # 没配视觉模型时这是个"哑"服务 —— 整图路径一调用就明确报错，
        # 而不是发一个注定失败的请求（决策模型独占模式）。
        self._enabled = enabled
        # 是否是 MiniMax 端点 —— 决定要不要发 thinking 私有扩展。
        self._is_minimax = _is_minimax_endpoint(base_url, model)
        self._timeout = timeout
        self.retries = max(0, retries)
        self.usage = usage if usage is not None else UsageTracker()
        self.last_question_usage = self.usage.snapshot()
        # 长连接，惰性创建。见 _get_client 的说明。
        self._client: httpx.Client | None = None
        # 严格模式（每日一答用）：模型思考块未闭合时直接抛错，不取
        # 尾部兜底，避免错误"以为了尾巴里是答案"。默认 True 保持 PK
        # 行为不变。
        self._recover_truncated_response = recover_truncated_response
        self._thinking = thinking
        self._timing_sink = timing_sink

    def _get_client(self) -> httpx.Client:
        """惰性创建并复用 httpx 客户端；使用共享 SSL context，关闭连接由 close 负责。"""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout, verify=_shared_ssl_context()
            )
        return self._client

    def warmup_client(self) -> float:
        """预先构造客户端和 SSL context，返回本地耗时；不发送网络请求或预热模型。"""
        started = time.perf_counter()
        try:
            self._get_client()
        except Exception:  # noqa: BLE001 - 预热失败不该影响答题
            return 0.0
        return time.perf_counter() - started

    def close(self) -> None:
        """关掉长连接。不用也能跑完（进程退出即回收），但显式释放更干净。"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "VisionService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def recognize(self, image: Image.Image, prompt: str = PROMPT, *,
                  timeout: float | None = None, deadline: float | None = None,
                  max_tokens: int | None = None) -> str:
        """发一张图给模型，返回原始文本（已剥离 think 块）。"""
        if not self._enabled:
            # 没配视觉模型 —— 立刻说清楚，不要发一个注定失败的请求。
            raise AnswerUnavailable("vision_model_not_configured")
        retries = self.retries if deadline is None else 0
        last_error: Exception | None = None
        budget = max_tokens or _IMAGE_MAX_TOKENS
        for attempt in range(retries + 1):
            try:
                _check_deadline(deadline)
                options = {"deadline": deadline} if deadline is not None else {}
                text = self._call_once(image, prompt, timeout=timeout,
                                       max_tokens=budget, **options)
            except RecognitionTimeout:
                # deadline_exceeded：如果 deadline 还没到，重试一次。
                #
                # 失败日志经常显示 remaining=0.000s —— httpx 的 5 秒 HTTP
                # 超时和 deadline 几乎同时触发。但只要 deadline 还没真正
                # 过期，就多打一次不亏（卡死的概率极低）。
                if deadline is not None:
                    _check_deadline(deadline)  # 已过期则直接抛出
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    try:
                        options = {"deadline": deadline} if deadline is not None else {}
                        text = self._call_once(image, prompt, timeout=timeout,
                                               max_tokens=budget, **options)
                        if text.strip():
                            return text
                    except Exception:  # noqa: BLE001
                        raise
                raise
            except Exception as exc:  # noqa: BLE001 - 网络/格式错误统一重试
                last_error = exc
            else:
                if text.strip():
                    return text
                last_error = AnswerUnavailable("模型返回为空")
                # 返回空 = 思考块把预算吃光了（finish_reason=length）。
                # 原样重发一次还是空 —— 把预算翻倍才有意义。
                budget = min(budget * 2, _MAX_TOKENS_CEILING)

            if attempt < retries:
                # 快速失败说明是服务端抖动，歇一下比重连更有用。
                # 慢失败（约等于一次完整调用）说明是模型本身的问题，
                # 再试也是白等，直接放弃。
                time.sleep(_RETRY_BACKOFF_SECONDS)

        if deadline is not None:
            raise AnswerUnavailable("image_request_failed") from None
        raise AnswerUnavailable(
            f"模型 {self.retries + 1} 次调用都没返回内容：{last_error}"
        ) from last_error

    def _new_timing(self) -> RequestTiming:
        return RequestTiming(self._timing_sink, self.usage.snapshot().requests + 1)

    def _call_once(self, image: Image.Image, prompt: str, *,
                   timeout: float | None = None, deadline: float | None = None,
                   max_tokens: int = _IMAGE_MAX_TOKENS) -> str:
        timing = self._new_timing()
        try:
            with timing.stage("编码"):
                image_url = image_to_jpeg_data_url(image)
        except Exception:
            timing.finish(success=False)
            raise
        return self._post(
            timing,
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
            **({"deadline": deadline} if deadline is not None else {}),
        )

    def _post(self, timing: RequestTiming, content: list[dict],
              max_tokens: int = 1200, *, timeout: float | None = None,
              deadline: float | None = None) -> str:
        """发出一次请求并返回剥掉思考块的正文。图片与纯文本共用。"""
        success = False
        started = time.monotonic()
        path = "image" if any(item.get("type") == "image_url" for item in content) else "text"
        reason, finish_reason, content_type = "request_failed", "unknown", "missing"
        content_empty, thinking = True, "unknown"
        try:
            _check_deadline(deadline)
            # max_tokens 必须给够：推理模型的 <think> 思考块也吃 token。
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            # MiniMax 省略 thinking 时默认开启；仅在明确关闭时发私有扩展。
            #
            if not self._thinking and self._is_minimax:
                payload["thinking"] = {"type": "disabled"}
            # 连接生命周期仍交给 _get_client / close()，不能 with client。
            with timing.stage("客户端准备"):
                client = self._get_client()
            raw_usage = None
            try:
                trace_options = ({"extensions": {"trace": timing.trace}}
                                 if self._timing_sink is not None else {})
                options = self._timeout_option(timeout, deadline=deadline)
                with timing.stage("HTTP总计"):
                    resp = client.post(self._url, json=payload, headers=self._headers,
                                       **trace_options, **options)
                if resp.status_code >= 400:
                    reason = "http_status"
                    if deadline is None:
                        raise AnswerUnavailable(f"模型 API 返回 {resp.status_code}: {resp.text[:200]}")
                    try:
                        data = resp.json()
                        raw_usage = data.get("usage") if isinstance(data, dict) else None
                    except Exception:
                        pass
                else:
                    reason = "response_shape"
                    with timing.stage("JSON解析"):
                        data = resp.json()
                    raw_usage = data.get("usage") if isinstance(data, dict) else None
            finally:
                self.usage.record(raw_usage)
            _check_deadline(deadline)
            if resp.status_code >= 400:
                raise AnswerUnavailable(reason)
            choice = data["choices"][0]
            finish = choice.get("finish_reason")
            finish_reason = finish if finish in ("stop", "length", "content_filter", "tool_calls", "function_call") else "unknown"
            content_text = choice["message"]["content"] or ""
            original = choice["message"]["content"]
            content_type = (type(original).__name__ if type(original) in
                            (str, list, dict, int, float, bool, type(None)) else "other")
            content_empty = not original
            if deadline is not None and not isinstance(original, str):
                reason = "content_type"
                raise AnswerUnavailable(reason)
            thinking = ("unclosed" if "<think>" in content_text and "</think>" not in content_text
                        else "closed" if "</think>" in content_text else "absent")
            if deadline is not None and thinking == "unclosed":
                reason = "unclosed_thinking"
                raise AnswerUnavailable(reason)
            # 严格模式：思考块未闭合直接抛错，不取尾部兜底。
            if not self._recover_truncated_response and "<think>" in content_text and "</think>" not in content_text:
                raise AnswerUnavailable("模型思考块未闭合")
            body = strip_think(content_text)
            if not body:
                body = _tail_after_think(content_text)
            success = bool(body.strip())
            reason = "ok" if success else "empty_content" if content_empty else "empty_body"
            _check_deadline(deadline)
            if deadline is not None and path == "text" and body and not re.search(r'"answer_index"\s*:\s*"?(\d+)', body):
                self._failure_summary(path, "invalid_answer", started, deadline, "none",
                                      finish_reason, content_type, content_empty, thinking)
            return body
        except Exception as exc:
            success = False
            if deadline is not None:
                if isinstance(exc, RecognitionTimeout) or time.monotonic() >= deadline:
                    reason = "deadline_exceeded"
                self._failure_summary(path, reason, started, deadline, type(exc).__name__,
                                      finish_reason, content_type, content_empty, thinking)
                if reason == "deadline_exceeded":
                    raise RecognitionTimeout(reason) from None
                raise AnswerUnavailable(reason) from None
            raise
        finally:
            if not success and reason in ("empty_content", "empty_body"):
                self._failure_summary(path, reason, started, deadline, "none",
                                      finish_reason, content_type, content_empty, thinking)
            timing.finish(success=success)

    def _failure_summary(self, path, reason, started, deadline, exception="none",
                         finish_reason="unknown", content_type="unknown",
                         content_empty=True, thinking="unknown") -> None:
        if self._timing_sink is None:
            return
        try:
            remaining = f"{max(0., deadline - time.monotonic()):.3f}s" if deadline is not None else "default"
            self._timing_sink(
                f"模型失败 path={path} reason={reason} exception={exception} "
                f"elapsed={time.monotonic() - started:.3f}s remaining={remaining} "
                f"finish_reason={finish_reason} content_type={content_type} "
                f"content_empty={content_empty} thinking={thinking}")
        except Exception:
            pass

    def _image_summary(self, kind: str, status: str, response, **fields) -> None:
        """只记录受控结构字段，不记录回复原文、图片或请求凭据。"""
        if self._timing_sink is None:
            return
        try:
            size = len(response) if isinstance(response, str) else 0
            detail = " ".join(f"{key}={value}" for key, value in fields.items())
            self._timing_sink(f"图片识别 kind={kind} status={status} response_chars={size} {detail}")
        except Exception:
            pass

    def _timeout_option(self, timeout: float | None, *, deadline: float | None = None) -> dict:
        """生成请求超时参数；按 deadline 限制剩余预算，不传 None 以免启用无限等待。"""
        if deadline is not None:
            _check_deadline(deadline)
            remaining = min(deadline - time.monotonic(), self._timeout,
                            timeout if timeout is not None else self._timeout)
            return {"timeout": httpx.Timeout(remaining)}
        if timeout is None:
            return {}
        return {"timeout": httpx.Timeout(timeout)}

    def read_question(
        self, image: Image.Image, prompt: str = PROMPT, *,
        timeout: float | None = None, deadline: float | None = None,
    ) -> VisionQuestion:
        """识别一道题并判答案。识别失败抛 AnswerUnavailable，绝不猜。"""
        started = time.monotonic()
        start = self.usage.snapshot()
        try:
            options = {"deadline": deadline} if deadline is not None else {}
            body = self.recognize(image, prompt=prompt, timeout=timeout, **options)
            try:
                question = parse_vision_response(body)
            except Exception as exc:
                fields = {
                    "parse_reason": exc.reason if isinstance(exc, _InvalidVisionResponse) else "unexpected_parse_error",
                    "exception": type(exc).__name__,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                }
                if deadline is not None:
                    fields["remaining_ms"] = round(max(0.0, deadline - time.monotonic()) * 1000)
                self._image_summary("question", "invalid_response", body, **fields)
                if deadline is None:
                    raise
                raise AnswerUnavailable("invalid_response") from None
            screen = question.screen if question.screen in ("question", "matching", "other") else "unknown"
            status = ("non_question" if question.screen != "question"
                      else "answer_missing" if question.answer_index is None else "ok")
            self._image_summary("question", status, body, screen=screen,
                                options=len(question.options),
                                answer="missing" if question.answer_index is None else "present")
            _check_deadline(deadline)
            return question
        finally:
            self.last_question_usage = self.usage.since(start)

    def answer_from_text(
        self,
        prompt: str,
        option_count: int,
        *,
        max_tokens: int = 2000,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> int | None:
        """提交题干与编号选项，只接受范围内的答案下标；失败返回 None，deadline 超时向外传播。"""
        timing = self._new_timing()
        start = self.usage.snapshot()
        try:
            body = self._post(
                timing,
                [{"type": "text", "text": prompt}],
                max_tokens=max_tokens,
                timeout=timeout,
                **({"deadline": deadline} if deadline is not None else {}),
            )
        except RecognitionTimeout:
            raise
        except Exception:
            return None
        finally:
            self.last_question_usage = self.usage.since(start)
        _check_deadline(deadline)
        match = re.search(r'"answer_index"\s*:\s*"?(\d+)', body)
        if match is None:
            return None
        index = int(match.group(1))
        _check_deadline(deadline)
        if not (0 <= index < option_count):
            return None
        return index

    def read_question_number(self, image: Image.Image) -> tuple[int, int] | None:
        """读出当前题号和总题数，读不到返回 None。"""
        crop = image.crop((0, _QUESTION_NO_BAND[0], image.size[0], _QUESTION_NO_BAND[1]))
        text = self.recognize(
            crop,
            prompt=(
                "这是答题界面的题号。只输出 JSON："
                '{"no":当前题号,"total":总题数}\n'
                "例如第 3 题共 10 题 → {\"no\":3,\"total\":10}"
            ),
        )
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        no, total = data.get("no"), data.get("total")
        if not isinstance(no, int) or not isinstance(total, int):
            return None
        if no <= 0 or total <= 0:
            return None
        return no, total

    def read_result_detail(self, image: Image.Image) -> dict:
        """读取结算动画的明确胜负，或积分面板的奖励、称号。"""
        text = self.recognize(
            image,
            prompt=(
                "这是微信读书 PK 的赛后截图，可能是胜负动画，也可能是奖励面板。"
                "只输出 JSON："
                '{"result":"","title":"","score":""}\n'
                "result：仅抄录画面明确出现的 VICTORY、DEFEAT、胜利或失败文字；"
                "没有就填空字符串，不要从星级、称号、比分推测输赢。"
                "title：仅抄录奖励面板顶部的称号，没有就填空字符串。"
                "score：只读标有‘获得积分’的奖励增量（如 +5、+1）；"
                "不要把 125 VS 80 之类对战比分填进去，没显示奖励就填空字符串。"
            ),
        )
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            self._image_summary("result", "missing_json", text)
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            self._image_summary("result", "invalid_json", text)
            return {}
        if not isinstance(data, dict):
            self._image_summary("result", "invalid_shape", text)
            return {}
        score = data.get("score")
        reward = (isinstance(score, str)
                  and re.fullmatch(r"(?:获得积分\s*)?\+\s*(?:1|3|4|5)", score.strip()))
        self._image_summary(
            "result", "parsed", text,
            result_type=type(data.get("result")).__name__,
            title_type=type(data.get("title")).__name__,
            score_type=type(score).__name__,
            score_format="reward" if reward else "empty" if not score else "invalid",
        )
        return {
            key: data.get(key, "").strip() if isinstance(data.get(key), str) else ""
            for key in ("result", "title", "score")
        }
