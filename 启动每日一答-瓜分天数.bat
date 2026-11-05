@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set WEREQ_BANNER=daily-days
cd /d "%~dp0"
set "WEREQ_HOME=%~dp0"

rem Standalone build ships its own Python; source build uses the project .venv.
set "PYEXE="
if exist "python\python.exe" set "PYEXE=%~dp0python\python.exe"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"

if not defined PYEXE (
  echo [ERROR] Python runtime not found.
  echo Standalone build: python\python.exe is missing, re-download the zip.
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
echo ==== WeRead Quiz - Daily (days) ====
echo.
"%PYEXE%" -m weread_quiz daily --route daily-days
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
