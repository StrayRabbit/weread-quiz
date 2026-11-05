"""读图时思考块把 token 预算吃光 —— 重试要加预算，不能原样重发。

实测（deepseek-flash 读真实截图）：推理部分就用了 800~2700 token。
预算给 1200 时有 17% 的图返回空内容（finish_reason=length）；同一个请求
原样重发还是空 —— 必须把预算翻倍。
"""
from __future__ import annotations

import json

import httpx
import pytest
from PIL import Image

from weread_quiz.errors import AnswerUnavailable
from weread_quiz.vision import VisionService


def _chat(content: str) -> dict:
    return {
        "choices": [{
            "message": {"content": content},
            "finish_reason": "stop",
        }],
    }


def _service(handler) -> VisionService:
    service = VisionService("https://example.invalid/v1", "key", "model", retries=1)
    service._client = httpx.Client(transport=httpx.MockTransport(handler))
    return service


def _image() -> Image.Image:
    return Image.new("RGB", (430, 788), (20, 35, 70))


def test_an_empty_answer_is_retried_with_a_bigger_budget():
    """返回空 = 预算被思考块吃光 —— 第二次要带更大的 max_tokens。"""
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload["max_tokens"])
        if payload["max_tokens"] < 4000:
            # 预算不够，模型什么都没说。
            return httpx.Response(200, json=_chat(""))
        return httpx.Response(200, json=_chat("读出来了"))

    with _service(handler) as service:
        assert service.recognize(_image()) == "读出来了"
    assert len(seen) == 2
    assert seen[1] > seen[0], seen


def test_the_first_attempt_uses_a_generous_budget():
    """思考模型读图要 800~2700 token —— 起步就得给够。"""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json=_chat("读出来了"))

    with _service(handler) as service:
        service.recognize(_image())
    assert seen[0] >= 2000, seen


def test_an_empty_answer_without_retries_fails_loudly():
    """没有重试机会时（PK 的单次路径）要明确失败，不能返回空串。"""
    def handler(request):
        return httpx.Response(200, json=_chat(""))

    service = VisionService("https://example.invalid/v1", "key", "model", retries=0)
    service._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(AnswerUnavailable):
        service.recognize(_image())
