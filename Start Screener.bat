@echo off
rem Double-click to start the screener and open it in the browser. Close this window to stop it.
cd /d "%~dp0"
title Divergence Screener
rem Open the browser only once the app answers (gives up after 60 seconds).
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 200;$i++){try{$null=Invoke-WebRequest http://127.0.0.1:5099/ -UseBasicParsing -TimeoutSec 2; Start-Process http://127.0.0.1:5099; break}catch{Start-Sleep -Milliseconds 300}}"
python macd_divergence_screener_v5.py
echo.
echo The screener stopped. If it says the address is already in use, it is already running in another window.
pause
