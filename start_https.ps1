[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$Reload,
    [switch]$SkipInstall,
    [Alias("Host")]
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8000
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "If this is your first run, install and trust certs\dev-ca.pem on the phone before opening the mobile camera page." -ForegroundColor Cyan
& (Join-Path $projectRoot "run.ps1") -UseHttps -Cuda:$Cuda -Reload:$Reload -SkipInstall:$SkipInstall -BindHost $BindHost -Port $Port
