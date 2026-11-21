# Console front end for the screener: start the app, show progress until it answers,
# open the browser, then show status and a few keys. Started by "Start Screener.bat".
param([int]$Port = 5099, [string]$Script = 'macd_divergence_screener_v5.py')   # overridable so the launcher can be tested

$url = "http://127.0.0.1:$Port"
$log = "$env:TEMP\divergence-screener-$Port.log"          # request log and errors
$out = "$env:TEMP\divergence-screener-$Port.out.log"      # the app's own messages
$Host.UI.RawUI.WindowTitle = 'Divergence Screener'

function Test-Port { $c = New-Object Net.Sockets.TcpClient; try { $c.ConnectAsync('127.0.0.1', $Port).Wait(200) } catch { $false } finally { $c.Close() } }
function Test-App { try { $null = Invoke-WebRequest "$url/" -UseBasicParsing -TimeoutSec 5; $true } catch { $false } }
function Show-Log { Write-Host; Get-Content $log, $out -Tail 12 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }; Write-Host }
function Say($text, $color = 'Gray') { Write-Host ("`r  " + $text.PadRight(70)) -ForegroundColor $color }

Write-Host
Write-Host '  DIVERGENCE SCREENER' -ForegroundColor Cyan
Write-Host '  -------------------' -ForegroundColor DarkGray

if ((Test-Port) -and (Test-App)) {
    Say 'Already running. Opening it in your browser.' Green
    Start-Process $url; Start-Sleep 2; exit 0
}

# The app writes to files, never to this window: a stalled console must not be able to stall the app.
$app = Start-Process python -ArgumentList '-u', $Script -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $log
$spin = '|/-\'; $i = 0; $sw = [Diagnostics.Stopwatch]::StartNew(); $ready = $false
while (-not $ready) {
    if ($app.HasExited) { Say 'The screener stopped while starting. Last lines of its log:' Red; Show-Log; exit 1 }
    if ($sw.Elapsed.TotalSeconds -gt 90) { Say 'No answer after 90 seconds. Last lines of its log:' Red; Show-Log; Stop-Process -Id $app.Id -Force; exit 1 }
    Write-Host ("`r  {0}  Starting the screener ... {1:0.0}s   " -f $spin[$i++ % 4], $sw.Elapsed.TotalSeconds) -NoNewline -ForegroundColor Yellow
    Start-Sleep -Milliseconds 120
    if (Test-Port) { $ready = Test-App }
}
Say ("Ready in {0:0.0}s" -f $sw.Elapsed.TotalSeconds) Green
Start-Process $url
Write-Host "  Open at $url"
Write-Host
Write-Host '  [O] open in browser   [L] show the log   [Q] quit' -ForegroundColor DarkGray
Write-Host '  Closing this window also stops the screener.' -ForegroundColor DarkGray
Write-Host

$up = [Diagnostics.Stopwatch]::StartNew()
while (-not $app.HasExited) {
    Write-Host ("`r  Running  {0:hh\:mm\:ss}   " -f $up.Elapsed) -NoNewline -ForegroundColor Green
    Start-Sleep -Milliseconds 250
    if ([Console]::KeyAvailable) {
        switch ([Console]::ReadKey($true).Key) {
            'O' { Start-Process $url }
            'L' { Show-Log; Write-Host '  [O] open in browser   [L] show the log   [Q] quit' -ForegroundColor DarkGray; Write-Host }
            'Q' { Stop-Process -Id $app.Id -Force; Say 'Stopped.'; Start-Sleep 1; exit 0 }
        }
    }
}
Say 'The screener stopped. Last lines of its log:' Red; Show-Log; exit 1
