@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows_cpu.ps1"
if errorlevel 1 exit /b %errorlevel%
endlocal
