$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$certFile = Join-Path $projectRoot "certs\dev-cert.pem"
$keyFile = Join-Path $projectRoot "certs\dev-key.pem"
$python311 = "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (Test-Path $python311) {
    $pythonCmd = $python311
} elseif (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} else {
    $pythonCmd = "python"
}

if (-not (Test-Path $certFile) -or -not (Test-Path $keyFile)) {
    Write-Host "HTTPS certificate files were not found. Generating local development certificates..." -ForegroundColor Yellow
    & $pythonCmd (Join-Path $projectRoot "scripts\generate_dev_cert.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Certificate generation failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$env:CMS_HOST = "0.0.0.0"
$env:CMS_PORT = "8000"
$env:CMS_SSL_CERTFILE = $certFile
$env:CMS_SSL_KEYFILE = $keyFile

Write-Host "Starting Crowd Management System over HTTPS on https://0.0.0.0:8000" -ForegroundColor Green
Write-Host "If this is your first run, install and trust certs\dev-ca.pem on the phone before opening the mobile camera page." -ForegroundColor Cyan
& $pythonCmd (Join-Path $projectRoot "app.py")
