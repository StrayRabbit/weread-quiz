"""PK 文本路径上限可配置（PK_TEXT_TIMEOUT）回归。

慢模型（实测 MiniMax-M3.1-Flash 文本路径 3.2~4.6 秒、偶尔超过 5 秒）
会被写死的 5 秒上限切掉，切掉后整图路径只剩 4 秒 —— 两条路都超时
就只剩蒙答。这个上限必须能在 .env 里调。
"""
import importlib
from types import SimpleNamespace

import pytest
from PIL import Image

from weread_quiz import config, pk
from weread_quiz.ocr_core import OcrRead
from weread_quiz.pk_budget import _TEXT_PATH_TIMEOUT
from weread_quiz.pk_hybrid import HybridQuestionReader


def _read() -> OcrRead:
    return OcrRead("题目", ["甲", "乙"], [0.99, 0.99], 2,
                   has_progress=True, progress_text="第1题·共10题")


# ---- 配置 ---------------------------------------------------------------


@pytest.fixture
def env_root(monkeypatch, tmp_path):
    """把配置根目录指到临时目录 —— 不受仓库里真实 .env 的影响。"""
    monkeypatch.setenv("WEREQ_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(
        "MODEL_BASE_URL=https://api.deepseek.com/v1\n"
        "MODEL_API_KEY=sk-test\n"
        "MODEL_NAME=deepseek-chat\n",
        encoding="utf-8",
    )
    importlib.reload(config)
    yield config
    monkeypatch.delenv("WEREQ_HOME", raising=False)
    importlib.reload(config)


def test_default_text_timeout(monkeypatch, env_root):
    """没配 = 默认 5 秒。"""
    monkeypatch.delenv("PK_TEXT_TIMEOUT", raising=False)
    assert env_root.AppConfig.from_env().pk_text_timeout == _TEXT_PATH_TIMEOUT == 5.0


def test_text_timeout_is_read_from_env(monkeypatch, env_root):
    """配了就是配的值。"""
    monkeypatch.setenv("PK_TEXT_TIMEOUT", "6.5")
    assert env_root.AppConfig.from_env().pk_text_timeout == 6.5


@pytest.mark.parametrize("raw", ["abc", "0", "-1"])
def test_bad_text_timeout_is_rejected(monkeypatch, env_root, raw):
    """写错了要当场报错 —— 不能默默按默认值跑。"""
    monkeypatch.setenv("PK_TEXT_TIMEOUT", raw)
    with pytest.raises(ValueError):
        env_root.AppConfig.from_env()


# ---- 混合读取器 ---------------------------------------------------------


class _Ocr:
    """只实现本地门控用到的两个接口。"""

    def read(self, image):
        return _read()

    def decide_merged(self, whole, strip):
        return SimpleNamespace(accepted=True, read=whole, reason="")


class _Model:
    """记录文本路径那次请求的单次上限。"""

    def __init__(self):
        self.seen = {}

    def answer_from_text(self, prompt, count, **kwargs):
        self.seen.update(kwargs)
        return 1


def test_hybrid_text_path_uses_configured_cap():
    """配了 6.5 秒 → 文本请求的单次上限就是 6.5 秒。"""
    model = _Model()
    hybrid = HybridQuestionReader(model, _Ocr(), verbose=False, text_timeout=6.5)
    hybrid.read_question(Image.new("RGB", (430, 788)))
    assert model.seen["timeout"] == 6.5


def test_hybrid_text_path_falls_back_to_default_cap():
    """没传就用写死的默认值。"""
    model = _Model()
    hybrid = HybridQuestionReader(model, _Ocr(), verbose=False)
    assert hybrid.text_timeout == _TEXT_PATH_TIMEOUT
    hybrid.read_question(Image.new("RGB", (430, 788)))
    assert model.seen["timeout"] == _TEXT_PATH_TIMEOUT


# ---- 预读复用路径 -------------------------------------------------------


def test_pre_read_uses_configured_cap():
    """等待阶段预读过的题，答题阶段复用它时也用配置的上限。"""
    seen = {}

    class Reader:
        text_timeout = 6.5

        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            seen["timeout"] = timeout
            return None

    pk._answer_from_pre_read(Reader(), _read())
    assert seen["timeout"] == 6.5


def test_pre_read_falls_back_to_default_cap():
    """替身没有 text_timeout（老调用方）时退回默认值。"""
    seen = {}

    class Reader:
        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            seen["timeout"] = timeout
            return None

    pk._answer_from_pre_read(Reader(), _read())
    assert seen["timeout"] == _TEXT_PATH_TIMEOUT


def test_pre_read_never_exceeds_remaining_budget(monkeypatch):
    """上限是「单次上限」，不是「无视预算」—— 剩余不够时按剩余算。"""
    monkeypatch.setattr(pk.time, "monotonic", lambda: 1000.0)
    seen = {}

    class Reader:
        text_timeout = 6.5

        def answer_from_ocr_read(self, read, *, timeout=None, deadline=None):
            seen["timeout"] = timeout
            return None

    pk._answer_from_pre_read(Reader(), _read(), deadline=1002.0)
    assert seen["timeout"] == pytest.approx(2.0)
