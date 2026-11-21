@echo off
rem Double-click to start the screener and open it in the browser. Close this window to stop it.
cd /d "%~dp0"
title Divergence Screener
set "LOG=%TEMP%\divergence-screener.log"
rem Open the browser only once the app answers (gives up after 60 seconds). It runs in its own
rem hidden window: sharing this console with the app froze the app.
start "" powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 200;$i++){try{$null=Invoke-WebRequest http://127.0.0.1:5099/ -UseBasicParsing -TimeoutSec 2; Start-Process http://127.0.0.1:5099; break}catch{Start-Sleep -Milliseconds 300}}"
echo Screener is starting. The browser opens when it is ready.
echo Address: http://127.0.0.1:5099
echo Close this window to stop it.
rem Output goes to a file, so nothing the console does (a click selecting text) can stall the app.
python -u macd_divergence_screener_v5.py > "%LOG%" 2>&1
echo.
echo The screener stopped. Last lines of %LOG%:
powershell -NoProfile -Command "Get-Content -Tail 15 $env:LOG"
pause
