# PathSnap - LAPTOP 1 (model + API server)
# Run:  powershell -ExecutionPolicy Bypass -File .\start_api_laptop1.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "envs\api\Scripts\python.exe"
if (-not (Test-Path $py)) {
    py -3.11 -m venv envs\api
    & $py -m pip install -r requirements\api.txt
}
foreach ($f in "artifacts\final_model.joblib", "artifacts\hmm_config.json") {
    if (-not (Test-Path $f)) { throw "Missing $f - train/finalize the model first (see README)." }
}
if (-not (Test-Path ".env")) { Copy-Item .env.example .env }
$evalDir = (Select-String -Path .env -Pattern '^PATHSNAP_EVAL_DIR=(.*)$').Matches.Groups[1].Value
Write-Host "Trips served from: $evalDir  ($((Get-ChildItem $evalDir -Directory).Count) trips)"

# firewall rule for port 8000 (needs Administrator once; skipped if it already exists)
if (-not (Get-NetFirewallRule -DisplayName "PathSnap API 8000" -ErrorAction SilentlyContinue)) {
    Write-Host "No firewall rule yet. In an *Administrator* PowerShell run once:" -ForegroundColor Yellow
    Write-Host '  New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow' -ForegroundColor Yellow
}
$ips = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }
Write-Host ""
Write-Host "Give Laptop 2 the address of the adapter on the SHARED network (usually Wi-Fi):" -ForegroundColor Green
$ips | ForEach-Object { Write-Host ("   {0,-18} http://{1}:8000" -f $_.InterfaceAlias, $_.IPAddress) -ForegroundColor Green }
Write-Host "Keep this window open - it shows the API logs the evaluator must see." -ForegroundColor Cyan
Write-Host ""
$ErrorActionPreference = "Continue"
& $py -m uvicorn api.main:app --host 0.0.0.0 --port 8000
