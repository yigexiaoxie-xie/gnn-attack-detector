@echo off
setlocal
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\start_platform.ps1"
if errorlevel 1 (
  echo Platform start failed. Review the error above.
  exit /b 1
)
