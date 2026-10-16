# Invoked by Setup PET-TRACER.cmd. Compatible with Windows PowerShell 5.1.
[CmdletBinding()]
param(
    [ValidateSet('CPU', 'NVIDIA')][string]$Device,
    [switch]$FunctionsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-PetNative {
    param([string]$Executable, [string[]]$Arguments)
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Required program was not found: $Executable"
    }
    # Native stderr may contain normal pip progress on Windows PowerShell 5.1.
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    $result = $LASTEXITCODE
    if ($result -ne 0) { throw "Command stopped with exit code ${result}: $Executable $($Arguments -join ' ')" }
}

function Get-PetManager {
    $command = Get-Command pymanager.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $alias = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\pymanager.exe'
    if (Test-Path -LiteralPath $alias -PathType Leaf) { return $alias }
    return $null
}

function Get-PetPythonCandidates {
    # Probe concrete executables/launchers, never an unversioned Store alias.
    $candidates = @(
        @{ Exe = (Join-Path $env:USERPROFILE '.venvs\pet-tracer\Scripts\python.exe'); Prefix = @() },
        @{ Exe = (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'); Prefix = @() }
    )
    $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($launcher) { $candidates += @{ Exe = $launcher.Source; Prefix = @('-3.11-64') } }
    $onPath = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source -notlike '*\WindowsApps\*') {
        $candidates += @{ Exe = $onPath.Source; Prefix = @() }
    }
    $manager = Get-PetManager
    if ($manager) { $candidates += @{ Exe = $manager; Prefix = @('exec', '-V:3.11-64') } }
    return $candidates
}

function Find-PetPython {
    # Single quotes inside Python code survive Windows PowerShell 5.1 native argument passing.
    $probe = "import sys,struct,json; assert sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8; print(json.dumps(sys.executable))"
    foreach ($candidate in (Get-PetPythonCandidates)) {
        if (-not (Test-Path -LiteralPath $candidate.Exe -PathType Leaf)) { continue }
        try {
            $ErrorActionPreference = 'Continue'
            $prefix = $candidate.Prefix
            $output = @(& $candidate.Exe @prefix -c $probe 2>$null)
            if ($LASTEXITCODE -eq 0 -and $output.Count -gt 0) {
                $resolved = $output[-1] | ConvertFrom-Json
                if (Test-Path -LiteralPath $resolved -PathType Leaf) { return $resolved }
            }
        } catch { continue }
    }
    return $null
}

function Install-PetPython {
    $manager = Get-PetManager
    if (-not $manager) {
        Write-Host 'Installing the official Python Install Manager for your Windows account...'
        $winget = Get-Command winget.exe -CommandType Application -ErrorAction SilentlyContinue
        $installed = $false
        if ($winget) {
            try {
                Invoke-PetNative $winget.Source @('install', '--id', '9NQ7512CXL7T', '--exact', '--source', 'msstore', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
                $installed = $true
            } catch {
                Write-Host 'WinGet could not install Python Install Manager. Trying the official Python installer...'
            }
        }
        if (-not $installed) {
            # Official Python fallback when WinGet is unavailable or blocked.
            Add-AppxPackage -AppInstallerFile 'https://www.python.org/ftp/python/pymanager/pymanager.appinstaller'
        }
        $manager = Get-PetManager
    }
    if (-not $manager) {
        throw 'Python Install Manager is not available yet. Close this window and double-click Setup PET-TRACER.cmd again. If it is still missing, install it from https://www.python.org/downloads/windows/.'
    }
    Invoke-PetNative $manager @('install', '3.11-64')
    $python = Find-PetPython
    if (-not $python) { throw '64-bit Python 3.11 could not be found after installation. Restart Setup PET-TRACER.cmd. See LoRACM/docs/WINDOWS.md if this continues.' }
    return $python
}

function Read-PetDevice {
    Write-Host ''
    Write-Host 'Choose your computer:'
    Write-Host '  1 - NVIDIA GPU, including RTX 5060'
    Write-Host '  2 - CPU'
    Write-Host '  Q - Cancel'
    while ($true) {
        $choice = Read-Host 'Enter 1, 2 or Q'
        switch ($choice.Trim().ToUpperInvariant()) {
            '1' { return 'NVIDIA' }
            '2' { return 'CPU' }
            'Q' { return 'Cancel' }
            default { Write-Host 'Please choose 1, 2 or Q.' }
        }
    }
}

function Get-PetSetupPlan {
    param([string]$Root, [string]$Python, [ValidateSet('CPU', 'NVIDIA')][string]$Choice, [string]$Report)
    $environment = Join-Path $env:USERPROFILE '.venvs\pet-tracer'
    $index = if ($Choice -eq 'NVIDIA') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
    $compute = if ($Choice -eq 'NVIDIA') { 'cuda:0' } else { 'cpu' }
    return @{
        Python = $Python
        Environment = $environment
        EnvironmentPython = (Join-Path $environment 'Scripts\python.exe')
        SetupArguments = @('-u', (Join-Path $Root 'setup_environment.py'), '--pytorch-index', $index)
        CheckArguments = @('-u', (Join-Path $Root 'LoRACM\check_environment.py'), '--device', $compute, '--output', $Report)
    }
}

function Invoke-PetSetup {
    param([string]$Root, [string]$Choice, [string]$LogDirectory = (Join-Path $env:LOCALAPPDATA 'PET-TRACER\setup-logs'))
    $transcript = $false
    $code = 1
    $log = $null
    try {
        if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
            throw 'This installer supports 64-bit Windows PCs with Intel or AMD processors.'
        }
        Set-Location -LiteralPath $Root
        foreach ($file in @('setup_environment.py', 'LoRACM\requirements-gui.txt', 'LoRACM\check_environment.py')) {
            if (-not (Test-Path -LiteralPath (Join-Path $Root $file) -PathType Leaf)) {
                throw 'Extract the complete PET-TRACER ZIP before running setup. Keep Setup PET-TRACER.cmd beside setup_environment.py.'
            }
        }
        $logs = $LogDirectory
        New-Item -ItemType Directory -Force -Path $logs | Out-Null
        $stem = 'setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + $PID
        $log = Join-Path $logs ($stem + '.log')
        $report = Join-Path $logs ($stem + '-compute.json')
        Start-Transcript -LiteralPath $log -Force | Out-Null
        $transcript = $true
        Write-Host ''
        Write-Host 'PET-TRACER setup'
        Write-Host 'Setup downloads Python 3.11 if needed and installs the required packages.'
        Write-Host 'The environment is stored outside this repo and can be reused by new downloads.'
        Write-Host 'Close any running PET-TRACER app before updating an existing environment.'
        Write-Host 'Keep this window open until setup finishes. Internet access is required.'
        if (-not $Choice) { $Choice = Read-PetDevice }
        if ($Choice -eq 'Cancel') { Write-Host 'Setup cancelled.'; return 0 }

        Write-Host ''
        Write-Host '[1/3] Preparing 64-bit Python 3.11...'
        $python = Find-PetPython
        if (-not $python) { $python = Install-PetPython }
        Write-Host "Python: $python"
        $plan = Get-PetSetupPlan -Root $Root -Python $python -Choice $Choice -Report $report

        Write-Host ''
        Write-Host '[2/3] Creating or updating your reusable environment...'
        Invoke-PetNative $plan.Python $plan.SetupArguments

        Write-Host ''
        Write-Host '[3/3] Checking installed packages and the selected device...'
        $imports = "import streamlit,torch,numpy,pandas,scipy,tables,jax; print('Package imports passed; JAX CPU:',jax.devices('cpu'))"
        Invoke-PetNative $plan.EnvironmentPython @('-u', '-c', $imports)
        Invoke-PetNative $plan.EnvironmentPython $plan.CheckArguments
        Write-Host ''
        Write-Host 'SETUP COMPLETE' -ForegroundColor Green
        Write-Host "Environment: $($plan.Environment)"
        Write-Host 'Double-click Launch PET-TRACER.cmd to open the GUI.'
        Write-Host "Compute report: $report"
        $code = 0
    } catch {
        Write-Host ''
        Write-Host 'SETUP STOPPED' -ForegroundColor Red
        Write-Host $_.Exception.Message
        Write-Host 'Review the error above and LoRACM/docs/WINDOWS.md. Your project data and trained models have not been removed.'
        Write-Host 'For a CUDA error, check the NVIDIA driver or rerun setup and choose CPU.'
    } finally {
        if ($log) { Write-Host "Setup log: $log" }
        if ($transcript) { Stop-Transcript | Out-Null }
    }
    return $code
}

if (-not $FunctionsOnly) {
    exit (Invoke-PetSetup -Root $PSScriptRoot -Choice $Device)
}
