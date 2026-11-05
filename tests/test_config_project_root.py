"""项目根目录的定位，以及启动器和它之间的契约。

免安装版把程序装进 site-packages，源码布局推导出的根目录不再成立 ——
启动器用 WEREQ_HOME 把解压目录告诉配置模块。这条契约两边都要有测试兜着。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from weread_quiz import config

REPO_ROOT = Path(config.__file__).resolve().parents[2]
LAUNCHERS = [
    "启动PK.bat",
    "启动每日一答-瓜分天数.bat",
    "启动每日一答-瓜分书币.bat",
]


def test_source_layout_falls_back_to_repo_root(monkeypatch):
    """没设 WEREQ_HOME —— 源码布局下根目录是仓库根。"""
    monkeypatch.delenv("WEREQ_HOME", raising=False)
    assert config._project_root() == REPO_ROOT


def test_wereq_home_wins(monkeypatch, tmp_path):
    """免安装版：WEREQ_HOME 指向解压目录。"""
    monkeypatch.setenv("WEREQ_HOME", str(tmp_path))
    assert config._project_root() == tmp_path


def test_reload_moves_data_dir(monkeypatch, tmp_path):
    """data/ 跟着根目录走 —— 免安装版的截图和日志落在解压目录里。"""
    monkeypatch.setenv("WEREQ_HOME", str(tmp_path))
    try:
        reloaded = importlib.reload(config)
        assert reloaded.PROJECT_ROOT == tmp_path
        assert reloaded.DATA_DIR == tmp_path / "data"
        assert reloaded.RUNS_DIR == tmp_path / "data" / "runs"
    finally:
        monkeypatch.delenv("WEREQ_HOME", raising=False)
        importlib.reload(config)


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_is_pure_ascii(name):
    """启动器必须纯 ASCII —— cmd 按系统代码页解码，非 ASCII 会错位。"""
    raw = (REPO_ROOT / name).read_bytes()
    raw.decode("ascii")  # 非 ASCII 会在这里炸出来
    assert b"\r\n" in raw, "批处理必须是 CRLF —— LF 会让 cmd 把整行当成一条命令"


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_exports_project_root(name):
    """启动器必须把 WEREQ_HOME 指向自己所在的目录。"""
    text = (REPO_ROOT / name).read_text(encoding="ascii")
    assert 'set "WEREQ_HOME=%~dp0"' in text


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_finds_standalone_python(name):
    """免安装版用随包的 python/，源码版用 .venv/ —— 两条路都得在。"""
    text = (REPO_ROOT / name).read_text(encoding="ascii")
    assert "python\\python.exe" in text
    assert ".venv\\Scripts\\python.exe" in text
