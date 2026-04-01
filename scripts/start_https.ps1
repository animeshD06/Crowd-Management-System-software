$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$certFile = Join-Path $root "certs\dev-cert.pem"
$keyFile = Join-Path $root "certs\dev-key.pem"

if (-not (Test-Path $certFile) -or -not (Test-Path $keyFile)) {
  Write-Host "HTTPS certificate files were not found." -ForegroundColor Yellow
  Write-Host "Expected:" -ForegroundColor Yellow
  Write-Host "  $certFile"
  Write-Host "  $keyFile"
  Write-Host ""
  Write-Host "Add your PEM certificate and key there, then rerun this script." -ForegroundColor Yellow
  exit 1
}

$env:CMS_HOST = "0.0.0.0"
$env:CMS_PORT = "8000"
$env:CMS_RELOAD = "false"
$env:CMS_SSL_CERTFILE = $certFile
$env:CMS_SSL_KEYFILE = $keyFile

Write-Host "Starting Crowd Management System with HTTPS on https://0.0.0.0:8000" -ForegroundColor Green
python app.py
