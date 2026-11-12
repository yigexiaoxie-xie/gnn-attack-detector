@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_detector.ps1"
if errorlevel 1 exit /b %errorlevel%
endlocal
