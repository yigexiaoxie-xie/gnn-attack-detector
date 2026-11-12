@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_selected_attacks.ps1"
if errorlevel 1 exit /b %errorlevel%
endlocal
