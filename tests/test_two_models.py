"""两个模型各管一段：主模型判下标，视觉模型复核 + 整图兜底。

**两种模型都得能用 —— 配上第二个模型不能把第一个顶掉，配一个也不能让
另一个消失。** 这里守住的是这几条：

* 主模型是决策模型 + 配了视觉模型 → 整图路径有人接（之前是"没配视觉
  模型就整图不可用"）。
* 决策模型拿不准 → 视觉模型复核（实测把答对率从 85.5% 拉到 98.6%）。
* 没有视觉模型 → 决策模型的答案再不确定也留着，别扔掉去蒙。
* 整图路径的 prompt 不能是 null —— 那是"视觉模型又不行了"的元凶。
"""
from __future__ import annotations

import httpx
import pytest

from weread_quiz.config import AppConfig
from weread_quiz.errors import AnswerUnavailable
from weread_quiz.model_service import ModelService, build_model_service
from weread_quiz.ocr_core import OcrRead

DECISION_URL = "https://example.invalid/v1"
VISION_URL = "https://vision.invalid/v1"


def _decision_response(choice: str, confidence: float) -> dict:
    return {
        "answer": {"type": "choice", "choice": choice, "confidence": confidence},
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }


def _chat_response(index: int) -> dict:
    return {
        "choices": [{
            "message": {"content": f'{{"answer_index": {index}}}'},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5},
    }


def _chat_question(question: dict) -> dict:
    """整图路径的响应形状 —— 要有题干和选项。"""
    import json

    return {
        "choices": [{
            "message": {"content": json.dumps(question, ensure_ascii=False)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5},
    }


class _Recorder:
    """记下每个服务各自收到了几次请求。"""

    def __init__(self):
        self.decision_calls = 0
        self.vision_calls = 0
        self.vision_bodies: list[dict] = []


def _facade(decision_response, vision_response, recorder=None,
            **kwargs) -> ModelService:
    recorder = recorder or _Recorder()
    service = ModelService(
        DECISION_URL, "key", "decision-model-preview",
        image_base_url=VISION_URL,
        image_api_key="vkey",
        image_model="deepseek-chat",
        **kwargs,
    )

    def decision_handle(request):
        recorder.decision_calls += 1
        return httpx.Response(200, json=decision_response)

    def vision_handle(request):
        recorder.vision_calls += 1
        import json

        recorder.vision_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=vision_response)

    service._text._client = httpx.Client(
        transport=httpx.MockTransport(decision_handle))
    service._image._client = httpx.Client(
        transport=httpx.MockTransport(vision_handle))
    service._recorder = recorder
    return service

# ────────────────────────────── 能力 ──────────────────────────────


def test_a_vision_model_makes_the_image_path_available():
    """主模型是决策模型（读不了图）时，视觉模型把整图路径接回来。"""
    service = _facade(_decision_response("1", 0.9), _chat_response(0))
    assert service.protocol == "systemone"
    assert service.supports_image is True
    assert service.supports_structured_choice is True


def test_without_a_vision_model_the_image_path_is_unavailable():
    service = ModelService(DECISION_URL, "key", "decision-model-preview")
    assert service.supports_image is False
    with pytest.raises(AnswerUnavailable):
        service.read_question(object(), "读这张图")


def test_a_chat_model_alone_still_reads_images():
    """只配一个视觉模型 —— 老样子，文本和整图都归它。"""
    service = ModelService("https://api.deepseek.com/v1", "k", "deepseek-chat")
    assert service.supports_image is True
    assert service.supports_structured_choice is False


# ────────────────────────────── 复核 ──────────────────────────────


def test_a_confident_decision_answer_is_used_as_is():
    """拿不准才复核 —— 置信度够高就别多花一次请求。"""
    recorder = _Recorder()
    service = _facade(_decision_response("1", 0.95), _chat_response(2), recorder)
    assert service.choose_index("题目", ["甲", "乙", "丙"]) == 1
    assert recorder.decision_calls == 1
    assert recorder.vision_calls == 0


def test_an_unsure_decision_answer_is_reviewed_by_the_vision_model():
    """置信度低于门槛 → 视觉模型复核，答案以复核为准。"""
    recorder = _Recorder()
    service = _facade(_decision_response("1", 0.3), _chat_response(2), recorder)
    assert service.choose_index("题目", ["甲", "乙", "丙"]) == 2
    assert recorder.decision_calls == 1
    assert recorder.vision_calls == 1


def test_the_review_sends_the_same_question_text():
    """复核问的是同一道题 —— 题干和选项都得带上。"""
    recorder = _Recorder()
    service = _facade(_decision_response("1", 0.3), _chat_response(2), recorder)
    service.choose_index("题目", ["甲", "乙", "丙"], prompt="题干：%s 选项：%s")
    body = recorder.vision_bodies[0]
    text = body["messages"][0]["content"][0]["text"]
    assert "题目" in text and "1. 乙" in text


def test_without_a_vision_model_the_decision_answer_is_kept():
    """没有视觉模型可复核 —— 决策模型的答案再不确定也比蒙一个好。"""
    service = ModelService(DECISION_URL, "key", "decision-model-preview")

    def handle(request):
        return httpx.Response(200, json=_decision_response("1", 0.1))

    service._text._client = httpx.Client(transport=httpx.MockTransport(handle))
    assert service.choose_index("题目", ["甲", "乙", "丙"]) == 1


def test_the_review_can_be_switched_off():
    """门槛设成 0 —— 决策模型的答案直接用，不复核。"""
    recorder = _Recorder()
    service = _facade(_decision_response("1", 0.1), _chat_response(2), recorder,
                      review_threshold=0)
    assert service.choose_index("题目", ["甲", "乙", "丙"]) == 1
    assert recorder.vision_calls == 0


def test_the_review_threshold_is_configurable():
    recorder = _Recorder()
    service = _facade(_decision_response("1", 0.5), _chat_response(2), recorder,
                      review_threshold=0.9)
    assert service.choose_index("题目", ["甲", "乙", "丙"]) == 2
    assert recorder.vision_calls == 1


# ────────────────────────────── 整图路径 ──────────────────────────────


def test_the_image_path_sends_a_real_prompt():
    """prompt 传 None 时要用模型自带的默认提示 —— 不能发 null 过去。

    发 null 的话，一部分供应商会直接报错，表现为"视觉模型又不行了"。
    """
    recorder = _Recorder()
    question = {"stem": "题目", "options": ["甲", "乙"], "answer_index": 0}
    service = _facade(_decision_response("1", 0.9), recorder=recorder,
                      vision_response=_chat_question(question))
    from PIL import Image

    image = Image.new("RGB", (430, 788), (20, 35, 70))
    service.read_question(image)
    content = recorder.vision_bodies[0]["messages"][0]["content"]
    assert content[0]["text"]
    assert content[0]["type"] == "text"


# ────────────────────────────── 配置 ──────────────────────────────


def test_the_vision_model_group_is_optional():
    cfg = AppConfig(model_base_url="https://api.deepseek.com/v1",
                    model_api_key="sk-abc", model_name="deepseek-chat")
    assert cfg.vision_model_configured is False
    service = build_model_service(cfg)
    assert service.supports_image is True


def test_a_partial_vision_model_group_is_rejected(monkeypatch):
    """三项要么都填要么都不填 —— 填一半说清楚，别等答题时才炸。"""
    monkeypatch.setenv("VISION_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("VISION_API_KEY", "")
    monkeypatch.setenv("VISION_NAME", "deepseek-chat")
    with pytest.raises(ValueError) as exc:
        AppConfig.from_env()
    assert "VISION_API_KEY" in str(exc.value)


def test_a_placeholder_vision_model_counts_as_not_configured(monkeypatch):
    """模板里的示例行没改 —— 当作没配。"""
    monkeypatch.setenv("VISION_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("VISION_API_KEY", "sk-xxxxxxxx")
    monkeypatch.setenv("VISION_NAME", "deepseek-chat")
    cfg = AppConfig.from_env()
    assert cfg.vision_model_configured is False


def test_a_decision_model_in_the_vision_slot_is_warned_about():
    """视觉模型填成决策模型 —— 它读不了图，启动时要说一声。"""
    notes = []
    ModelService(
        DECISION_URL, "key", "decision-model-preview",
        image_base_url="https://jevmodel.org/v1/systemone",
        image_api_key="vkey",
        image_model="jev-latest",
        timing_sink=notes.append,
    )
    assert any("决策模型" in note and "读不了图" in note for note in notes), notes
