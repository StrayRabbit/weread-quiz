"""`.env` 用记事本存成 ANSI（GBK）也要能读。

中文 Windows 的记事本默认存 GBK。用 UTF-8 读会抛 UnicodeDecodeError，
用户看到的是一串看不懂的报错 —— 这是第一次配置时最容易撞上的坑。
"""
from __future__ import annotations

import importlib

from weread_quiz import config

ENV_TEXT = (
    "# 中文注释\n"
    "MODEL_BASE_URL=https://api.deepseek.com/v1\n"
    "MODEL_API_KEY=sk-abc123\n"
    "MODEL_NAME=deepseek-chat\n"
)


def _read_env(monkeypatch, tmp_path, encoding: str):
    """在临时根目录里放一份 .env，重新加载配置后读出来。"""
    monkeypatch.setenv("WEREQ_HOME", str(tmp_path))
    (tmp_path / ".env").write_bytes(ENV_TEXT.encode(encoding))
    for name in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    reloaded = importlib.reload(config)
    return reloaded.AppConfig.from_env()


def _restore(monkeypatch):
    monkeypatch.delenv("WEREQ_HOME", raising=False)
    importlib.reload(config)


def test_a_gbk_env_file_is_read(monkeypatch, tmp_path):
    """记事本存成 ANSI（GBK）也能读 —— 不能抛看不懂的解码错误。"""
    try:
        cfg = _read_env(monkeypatch, tmp_path, "gbk")
        assert cfg.model_name == "deepseek-chat"
        assert cfg.model_api_key == "sk-abc123"
    finally:
        _restore(monkeypatch)


def test_a_utf8_env_file_is_read(monkeypatch, tmp_path):
    try:
        cfg = _read_env(monkeypatch, tmp_path, "utf-8")
        assert cfg.model_name == "deepseek-chat"
    finally:
        _restore(monkeypatch)


def test_the_env_file_is_optional(monkeypatch, tmp_path):
    """没有 .env 也能加载 —— 只是模型没配。"""
    try:
        monkeypatch.setenv("WEREQ_HOME", str(tmp_path))
        for name in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME"):
            monkeypatch.delenv(name, raising=False)
        reloaded = importlib.reload(config)
        cfg = reloaded.AppConfig.from_env()
        assert cfg.model_configured is False
    finally:
        _restore(monkeypatch)
