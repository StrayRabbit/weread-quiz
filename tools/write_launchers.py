"""生成纯 ASCII、CRLF 启动器；中文由 Python 输出，避免 cmd 代码页解码差异。

同一套启动器要同时支持两种部署：

* 源码模式 —— 依赖项目里的 ``.venv``（先跑 ``uv sync``）。
* 免安装模式 —— 依赖随包携带的便携 Python（``python\\python.exe``）。

免安装模式下程序装在 site-packages 里，源码布局推导出的项目根目录不再成立，
所以启动器额外导出 ``WEREQ_HOME``，指向解压后的目录（.env 和 data/ 都在那儿）。
"""
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
set "WEREQ_HOME=%~dp0"

rem Standalone build ships its own Python; source build uses the project .venv.
set "PYEXE="
if exist "python\\python.exe" set "PYEXE=%~dp0python\\python.exe"
if not defined PYEXE if exist ".venv\\Scripts\\python.exe" set "PYEXE=%~dp0.venv\\Scripts\\python.exe"

if not defined PYEXE (
  echo [ERROR] Python runtime not found.
  echo Standalone build: python\\python.exe is missing, re-download the zip.
  echo Source build: run "uv sync" in this folder first.
  pause
  exit /b 1
)

if not exist ".env" (
  if not exist ".env.example" (
    echo [ERROR] .env.example is missing - re-download the package.
    pause
    exit /b 1
  )
  copy /y ".env.example" ".env" >nul
  echo [ERROR] Config file .env was missing - created one from .env.example.
  echo Fill in these three values and save:
  echo   MODEL_BASE_URL
  echo   MODEL_API_KEY
  echo   MODEL_NAME
  start notepad ".env"
  echo.
  echo Save the file and run this again.
  pause
  exit /b 1
)

echo.
echo ==== {title} ====
echo.
"%PYEXE%" -m weread_quiz {command}
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
"""

# 首次配置：复制 .env.example 到 .env 并用记事本打开。
CONFIG_TEMPLATE = """@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".env" (
  echo [.env] already exists, opening it.
) else (
  copy /y ".env.example" ".env" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to create .env from .env.example.
    pause
    exit /b 1
  )
  echo Created .env from .env.example.
)

start notepad ".env"
exit /b 0
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


def render(text: str) -> bytes:
    """渲染成纯 ASCII、CRLF 的批处理字节。

    纯 ASCII 是硬要求 —— cmd 按系统代码页解码批处理，非 ASCII 字符会错位，
    所以这里让 .encode("ascii") 直接炸出来，而不是悄悄写出一堆乱码。
    """
    try:
        return text.replace("\n", "\r\n").encode("ascii")
    except UnicodeEncodeError as exc:
        bad = text[exc.start:exc.start + 24]
        raise SystemExit(
            f"启动器必须纯 ASCII，第 {exc.start} 个字符是非 ASCII：{bad!r}"
        ) from exc


def write_all(dest_dir: Path) -> list[Path]:
    """把全部启动器写进 dest_dir，返回写出的文件列表。"""
    written: list[Path] = []
    for name, spec in LAUNCHERS.items():
        path = dest_dir / name
        data = render(TEMPLATE.format(**spec))
        path.write_bytes(data)
        written.append(path)
        print(f"写入 {path}（{len(data)} 字节，纯 ASCII）")

    path = dest_dir / "首次配置.bat"
    data = render(CONFIG_TEMPLATE)
    path.write_bytes(data)
    written.append(path)
    print(f"写入 {path}（{len(data)} 字节，纯 ASCII）")
    return written


def main() -> None:
    write_all(ROOT)


if __name__ == "__main__":
    main()
