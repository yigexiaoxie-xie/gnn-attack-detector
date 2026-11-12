@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_official_metattack.ps1"
if errorlevel 1 (
    echo Official Metattack graph preparation failed.
    exit /b 1
)
echo Official Metattack graphs are ready.
