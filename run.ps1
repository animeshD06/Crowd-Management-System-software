[CmdletBinding()]
param(
    [switch]$UseHttps,
    [switch]$Cuda,
    [switch]$Reload,
    [switch]$SkipInstall,
    [Alias("Host")]
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

function Get-PreferredPython {
    param([string]$ProjectRoot)

    $venvPython = Join-Path $ProjectRoot "$VenvPath\Scripts\python.exe"
    $python311 = "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"

    if (Test-Path $venvPython) { return $venvPython }
    if (Test-Path $python311) { return $python311 }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) { return "py -3.10" }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) { return "python" }

    throw "Python was not found. Install Python 3.10+ and rerun the script."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string]$PythonCommand,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    if ($PythonCommand -eq "py -3.10") {
        & py -3.10 @Arguments
    } else {
        & $PythonCommand @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Get-FileHashHex {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Test-PortAvailable {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [Parameter(Mandatory = $true)][int]$PortNumber
    )

    if ($HostName -eq "0.0.0.0") {
        $address = [System.Net.IPAddress]::Any
    } else {
        $parsedAddress = $null
        if ([System.Net.IPAddress]::TryParse($HostName, [ref]$parsedAddress)) {
            $address = $parsedAddress
        } else {
            $resolved = [System.Net.Dns]::GetHostAddresses($HostName) | Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } | Select-Object -First 1
            if (-not $resolved) {
                throw "Could not resolve host '$HostName' to an IPv4 address."
            }
            $address = $resolved
        }
    }

    $listener = [System.Net.Sockets.TcpListener]::new($address, $PortNumber)
    try {
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        $listener.Stop()
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path $projectRoot).Path
$venvRoot = Join-Path $projectRoot $VenvPath
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$bootstrapPython = Get-PreferredPython -ProjectRoot $projectRoot

$requirementsFileName = if ($Cuda) { "requirements-cuda.txt" } else { "requirements-cpu.txt" }
$requirementsPath = Join-Path $projectRoot $requirementsFileName
$appPath = Join-Path $projectRoot "app.py"
$stateDir = Join-Path $venvRoot ".cms"
$stampPath = Join-Path $stateDir ("installed-" + $requirementsFileName + ".sha256")
$certFile = Join-Path $projectRoot "certs\dev-cert.pem"
$keyFile = Join-Path $projectRoot "certs\dev-key.pem"

if (-not (Test-Path $requirementsPath)) {
    throw "Requirements file not found: $requirementsPath"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvRoot" -ForegroundColor Cyan
    Invoke-Python -PythonCommand $bootstrapPython -Arguments @("-m", "venv", $venvRoot)
}

$runtimePython = $venvPython
$requirementsHash = Get-FileHashHex -Path $requirementsPath
$needsInstall = -not $SkipInstall

if (-not (Test-Path $stateDir)) {
    New-Item -ItemType Directory -Path $stateDir | Out-Null
}

if ($needsInstall) {
    if ((Test-Path $stampPath) -and ((Get-Content $stampPath -Raw).Trim() -eq $requirementsHash)) {
        $needsInstall = $false
    }
}

if ($needsInstall) {
    Write-Host "Installing dependencies from $requirementsFileName" -ForegroundColor Cyan
    Invoke-Python -PythonCommand $runtimePython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Python -PythonCommand $runtimePython -Arguments @("-m", "pip", "install", "-r", $requirementsPath)
    Set-Content -LiteralPath $stampPath -Value $requirementsHash -NoNewline
} elseif (-not $SkipInstall) {
    Write-Host "Dependencies already match $requirementsFileName" -ForegroundColor Green
}

if ($UseHttps) {
    if (-not (Test-Path $certFile) -or -not (Test-Path $keyFile)) {
        Write-Host "HTTPS certificate files were not found. Generating local development certificates..." -ForegroundColor Yellow
        Invoke-Python -PythonCommand $runtimePython -Arguments @((Join-Path $projectRoot "scripts\generate_dev_cert.py"))
    }

    $env:CMS_SSL_CERTFILE = $certFile
    $env:CMS_SSL_KEYFILE = $keyFile
    $scheme = "https"
} else {
    Remove-Item Env:CMS_SSL_CERTFILE -ErrorAction SilentlyContinue
    Remove-Item Env:CMS_SSL_KEYFILE -ErrorAction SilentlyContinue
    $scheme = "http"
}

$env:CMS_HOST = $BindHost
$env:CMS_PORT = [string]$Port
$env:CMS_RELOAD = if ($Reload) { "true" } else { "false" }

$modeLabel = if ($Cuda) { "CUDA" } else { "CPU" }
$reloadLabel = if ($Reload) { "reload on" } else { "reload off" }

if (-not (Test-PortAvailable -HostName $BindHost -PortNumber $Port)) {
    throw "Port $Port is already in use on $BindHost. Start with a different port, for example: .\run.ps1 -Port 8001"
}

Write-Host "Starting Crowd Management System ($modeLabel, $reloadLabel) at ${scheme}://${BindHost}:$Port" -ForegroundColor Green
if (-not $UseHttps) {
    Write-Host "Mobile phone camera sharing will not work over plain HTTP unless the phone opens localhost on the same device." -ForegroundColor Yellow
    Write-Host "For the mobile portal, use: .\run.ps1 -UseHttps -BindHost 0.0.0.0" -ForegroundColor Yellow
}
Invoke-Python -PythonCommand $runtimePython -Arguments @($appPath)
