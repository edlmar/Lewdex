@echo off
rem Lewdex installer: double-click. Needs internet the first time (~8 GB of downloads).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
echo.
pause
