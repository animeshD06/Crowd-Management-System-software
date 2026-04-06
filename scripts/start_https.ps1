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

Write-Host "Starting Crowd Management System with HTTPS" -ForegroundColor Green
Write-Host "Server bind address: https://0.0.0.0:8000" -ForegroundColor DarkGray
Write-Host "Open in your browser: https://localhost:8000" -ForegroundColor Green
Write-Host "For another device on your Wi-Fi, use: https://<your-laptop-ip>:8000" -ForegroundColor Green
Write-Host "0.0.0.0 is only for binding the server, not for opening in a browser." -ForegroundColor Yellow
python app.py
