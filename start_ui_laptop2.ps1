# PathSnap - LAPTOP 2 (evaluator UI)
# Run:  powershell -ExecutionPolicy Bypass -File .\start_ui_laptop2.ps1 -ApiIp 192.168.1.23
param([string]$ApiIp = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "envs\ui\Scripts\python.exe"
if (-not (Test-Path $py)) {
    py -3.11 -m venv envs\ui
    & $py -m pip install -r requirements\ui.txt
}
if (-not $ApiIp) { $ApiIp = Read-Host "IP address of Laptop 1 (shown in its API window)" }
$url = "http://$($ApiIp):8000"
"PATHSNAP_API_URL=$url" | Set-Content -Encoding ascii .env
Write-Host "Checking $url/health ..."
try { Invoke-RestMethod "$url/health" -TimeoutSec 5 | Out-Null; Write-Host "API reachable." -ForegroundColor Green }
catch {
    Write-Host "Cannot reach the API at $url." -ForegroundColor Red
    Write-Host "Check: API window open on Laptop 1, same Wi-Fi/hotspot, firewall rule for port 8000, correct IP." -ForegroundColor Red
    exit 1
}
$ErrorActionPreference = "Continue"
& $py -m streamlit run ui\app.py --server.port 8501 --browser.gatherUsageStats false
