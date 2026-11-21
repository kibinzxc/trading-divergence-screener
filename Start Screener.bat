@echo off
rem Double-click to start the screener and open it in the browser. The window shows progress and status.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_screener.ps1"
if errorlevel 1 pause
