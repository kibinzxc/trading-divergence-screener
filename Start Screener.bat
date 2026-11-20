@echo off
rem Double-click to start the screener and open it in the browser. Close this window to stop it.
cd /d "%~dp0"
title Divergence Screener
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5099"
python macd_divergence_screener_v5.py
echo.
echo The screener stopped. If it says the address is already in use, it is already running in another window.
pause
