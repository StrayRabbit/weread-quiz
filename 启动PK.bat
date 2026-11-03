@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set WEREQ_BANNER=pk
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
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
echo ==== WeRead Quiz - PK ====
echo.
.venv\Scripts\python.exe -m weread_quiz pk
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
