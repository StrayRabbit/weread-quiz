"""模型服务门面：协议自动识别、两种协议的请求与响应、配置迁移。

配置里只有一个模型 —— 用哪种协议（OpenAI 兼容的 chat/completions，还是
决策模型的 System One）由代码从地址和模型名自己认。
"""
from __future__ import annotations

import httpx
import pytest

from weread_quiz.config import AppConfig
from weread_quiz.decision import DecisionService
from weread_quiz.errors import AnswerUnavailable
from weread_quiz.model_service import (
    PROTOCOL_CHAT,
    PROTOCOL_SYSTEMONE,
    ModelService,
    detect_protocol,
)
from weread_quiz.ocr_core import OcrRead
from weread_quiz.recognition import answer_from_read

# 阿里：答案直接在响应顶层。
ALI_RESPONSE = {
    "answer": {
        "type": "choice",
        "choice": "1",
        "confidence": 0.82,
        "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
    },
    "usage": {"input_tokens": 120, "output_tokens": 0},
}

# Jev：答案放在 answers 下，字段名也不同。
JEV_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "answer": {
            "type": "choice",
            "choice": "1",
            "confidence": 0.82,
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        }
    },
    "usage": {"input_tokens": 120, "output_tokens": 0},
}


def _service(response, **kwargs) -> DecisionService:
    def handle(request):
        return httpx.Response(200, json=response)

    service = DecisionService("https://example.invalid/v1", "key", "model", **kwargs)
    service._client = httpx.Client(transport=httpx.MockTransport(handle))
    return service


# ────────────────────────────── 协议识别 ──────────────────────────────


@pytest.mark.parametrize("base_url,model,expected", [
    # 地址本身是 System One 的（Jev 就是这么给的）
    ("https://jevmodel.org/v1/systemone", "jev-latest", PROTOCOL_SYSTEMONE),
    # 模型名像决策模型 —— 地址还是 chat 兼容的那套，要自己拼 /systemone
    ("https://w.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
     "decision-model-preview", PROTOCOL_SYSTEMONE),
    ("https://jevmodel.org/v1", "jev-1.13.0", PROTOCOL_SYSTEMONE),
    # 普通 chat 模型
    ("https://api.deepseek.com/v1", "deepseek-chat", PROTOCOL_CHAT),
    ("https://api.minimaxi.com/v1", "MiniMax-Text-01", PROTOCOL_CHAT),
])
def test_protocol_is_detected_from_url_and_model_name(base_url, model, expected):
    assert detect_protocol(base_url, model) == expected


def test_chat_service_can_read_images():
    service = ModelService("https://api.deepseek.com/v1", "k", "deepseek-chat")
    assert service.protocol == PROTOCOL_CHAT
    assert service.supports_image is True
    assert service.supports_structured_choice is False


def test_systemone_service_cannot_read_images():
    service = ModelService("https://jevmodel.org/v1/systemone", "k", "jev-latest")
    assert service.protocol == PROTOCOL_SYSTEMONE
    assert service.supports_image is False
    assert service.supports_structured_choice is True


def test_systemone_image_path_fails_loudly():
    """读不了图就明确报不可用 —— 不是发一个注定失败的请求。"""
    service = ModelService("https://jevmodel.org/v1/systemone", "k", "jev-latest")
    with pytest.raises(AnswerUnavailable) as exc:
        service.read_question(object(), "读这张图")
    assert "no_image_capability" in str(exc.value)


# ────────────────────────────── 决策模型协议 ──────────────────────────────


def test_request_body_carries_structured_criteria():
    """criteria 的键是选项下标 —— 响应里 choice 回的就是它。"""
    captured = {}

    def handle(request):
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=ALI_RESPONSE)

    with DecisionService("https://example.invalid/v1", "key", "decision-model-preview") as service:
        service._client = httpx.Client(transport=httpx.MockTransport(handle))
        service.answer_options("题目", ["甲", "乙", "丙"])

    assert captured["model"] == "decision-model-preview"
    question = captured["questions"]["answer"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == ["0", "1", "2"]
    assert question["criteria"]["1"] == "乙"
    assert "题目" in captured["state"]


def test_endpoint_is_systemone_not_chat_completions():
    with DecisionService("https://example.invalid/v1/", "key", "m") as service:
        assert service._url == "https://example.invalid/v1/systemone"


def test_endpoint_does_not_double_the_systemone_segment():
    """用户照着文档抄，地址里可能已经带了 /systemone。"""
    with DecisionService("https://jevmodel.org/v1/systemone", "key", "m") as service:
        assert service._url == "https://jevmodel.org/v1/systemone"


@pytest.mark.parametrize("response", [ALI_RESPONSE, JEV_RESPONSE])
def test_both_response_envelopes_give_the_same_index(response):
    """阿里放顶层、Jev 放 answers 下 —— 两种都要认。"""
    with _service(response) as service:
        assert service.answer_options("题目", ["甲", "乙", "丙"]) == 1
        assert service.last_confidence == pytest.approx(0.82)


@pytest.mark.parametrize("choice", ["9", "-1", "甲", None])
def test_out_of_range_choice_is_not_an_answer(choice):
    """越界下标不能当成答案 —— 那会点到别的按钮上。"""
    response = {"answer": {"type": "choice", "choice": choice, "confidence": 0.9}}
    with _service(response) as service:
        assert service.answer_options("题目", ["甲", "乙", "丙"]) is None


def test_confidence_below_threshold_counts_as_no_answer():
    """置信度不够就当没答出来，交给图片路径。"""
    response = {"answer": {"type": "choice", "choice": "1", "confidence": 0.2}}
    with _service(response, min_confidence=0.5) as service:
        assert service.answer_options("题目", ["甲", "乙", "丙"]) is None


def test_confidence_above_threshold_passes():
    response = {"answer": {"type": "choice", "choice": "1", "confidence": 0.9}}
    with _service(response, min_confidence=0.5) as service:
        assert service.answer_options("题目", ["甲", "乙", "丙"]) == 1


def test_usage_field_names_are_normalized():
    """Jev 用 input_tokens/output_tokens，用量统计只认 prompt/completion。"""
    with _service(JEV_RESPONSE) as service:
        service.answer_options("题目", ["甲", "乙", "丙"])
        assert service.usage.snapshot().prompt_tokens == 120


def test_http_error_is_not_an_answer():
    def handle(request):
        return httpx.Response(500, json={"error": "boom"})

    with DecisionService("https://example.invalid/v1", "key", "m") as service:
        service._client = httpx.Client(transport=httpx.MockTransport(handle))
        assert service.answer_options("题目", ["甲", "乙", "丙"]) is None


# ────────────────────────────── 文本路径的接缝 ──────────────────────────────


def test_seam_uses_structured_options_for_systemone(monkeypatch):
    """系统单协议：走结构化的题干+选项。"""
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    calls = []

    class Service:
        supports_structured_choice = True

        def answer_options(self, stem, options, **kwargs):
            calls.append(("structured", stem, tuple(options), kwargs))
            return 1

        def answer_from_text(self, prompt, count, **kwargs):
            calls.append(("prompt", prompt, count))
            return 1

    read = OcrRead(stem="题目", options=["甲", "乙", "丙"],
                   confidences=[0.9] * 3, button_count=3, has_progress=True)
    question = answer_from_read(Service(), read, "题目：%s 选项：%s",
                                timeout=3.0, deadline=103.0)
    assert question.answer_index == 1
    assert calls[0][0] == "structured"
    assert calls[0][1] == "题目"


def test_seam_uses_prompt_for_chat(monkeypatch):
    """普通协议：走拼好的 prompt。"""
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    calls = []

    class Service:
        supports_structured_choice = False

        def answer_from_text(self, prompt, count, **kwargs):
            calls.append(("prompt", prompt, count))
            return 1

    read = OcrRead(stem="题目", options=["甲", "乙", "丙"],
                   confidences=[0.9] * 3, button_count=3, has_progress=True)
    answer_from_read(Service(), read, "题目：%s 选项：%s")
    assert calls[0][0] == "prompt"
    assert "0. 甲\n1. 乙\n2. 丙" in calls[0][1]


# ────────────────────────────── 配置 ──────────────────────────────


def test_placeholder_api_key_is_rejected_with_a_clear_message():
    """密钥还是模板里的占位符就报错 —— 总比跑起来每题都蒙强。"""
    cfg = AppConfig(model_base_url="https://api.minimaxi.com/v1",
                    model_api_key="sk-xxxxxxxx",
                    model_name="MiniMax-Text-01")
    with pytest.raises(ValueError) as exc:
        cfg.validate_model()
    assert "占位符" in str(exc.value)


def test_a_real_api_key_passes():
    cfg = AppConfig(model_base_url="https://api.deepseek.com/v1",
                    model_api_key="sk-abc123", model_name="deepseek-chat")
    cfg.validate_model()  # 不抛异常


def test_missing_model_is_rejected():
    cfg = AppConfig(model_base_url="", model_api_key="", model_name="")
    with pytest.raises(ValueError):
        cfg.validate_model()
