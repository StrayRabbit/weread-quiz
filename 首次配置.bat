@echo off
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
