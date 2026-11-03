"""生成纯 ASCII、CRLF 启动器；中文由 Python 输出，避免 cmd 代码页解码差异。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 启动器的 ASCII 骨架。{title} / {command} 由下方填入。
# 提示文字全部是 ASCII —— 中文在 Python 侧输出（见 WEREQ_BANNER）。
TEMPLATE = """@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set WEREQ_BANNER={banner}
cd /d "%~dp0"

if not exist ".venv\\Scripts\\python.exe" (
  echo [ERROR] Virtualenv .venv not found.
  echo Run this in the project folder: uv sync
  pause
  exit /b 1
)

if not exist ".env" (
  echo [ERROR] Config file .env not found.
  echo Copy .env.example to .env and fill in the model settings.
  pause
  exit /b 1
)

echo.
echo ==== {title} ====
echo.
.venv\\Scripts\\python.exe -m weread_quiz {command}
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
"""

LAUNCHERS = {
    "启动PK.bat": {
        "title": "WeRead Quiz - PK",
        "command": "pk",
        "banner": "pk",
    },
    "启动每日一答-瓜分天数.bat": {
        "title": "WeRead Quiz - Daily (days)",
        "command": "daily --route daily-days",
        "banner": "daily-days",
    },
    "启动每日一答-瓜分书币.bat": {
        "title": "WeRead Quiz - Daily (coins)",
        "command": "daily --route daily-coins",
        "banner": "daily-coins",
    },
}


def main() -> None:
    for name, spec in LAUNCHERS.items():
        text = TEMPLATE.format(**spec)
        path = ROOT / name
        # 纯 ASCII —— cmd 按系统代码页怎么解码都不会错位。
        data = text.replace("\n", "\r\n").encode("ascii")
        path.write_bytes(data)
        print(f"写入 {name}（{len(data)} 字节，纯 ASCII）")


if __name__ == "__main__":
    main()
