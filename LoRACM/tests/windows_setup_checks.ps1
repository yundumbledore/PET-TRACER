param([Parameter(Mandatory = $true)][string]$TestPython)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$installer = Join-Path $root 'setup_windows.ps1'
$tokens = $null
$parseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile($installer, [ref]$tokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count -gt 0) { throw ($parseErrors | Out-String) }
. $installer -FunctionsOnly

function Assert-PetTest {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

$testRoot = Join-Path $env:TEMP ('PET setup test ' + [char]0x03BC + ' ' + [guid]::NewGuid())
$oldLocation = Get-Location
try {
    New-Item -ItemType Directory -Path (Join-Path $testRoot 'LoRACM') -Force | Out-Null
    foreach ($file in @('setup_environment.py', 'LoRACM\requirements-gui.txt', 'LoRACM\check_environment.py')) {
        Set-Content -LiteralPath (Join-Path $testRoot $file) -Value '# fixture'
    }
    $report = Join-Path $testRoot 'compute report.json'
    foreach ($choice in @('CPU', 'NVIDIA')) {
        $plan = Get-PetSetupPlan -Root $testRoot -Python $TestPython -Choice $choice -Report $report
        $expectedIndex = if ($choice -eq 'NVIDIA') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
        $expectedDevice = if ($choice -eq 'NVIDIA') { 'cuda:0' } else { 'cpu' }
        Assert-PetTest ($plan.SetupArguments[1] -eq (Join-Path $testRoot 'setup_environment.py')) 'Setup path lost spaces/Unicode'
        Assert-PetTest ($plan.SetupArguments[3] -eq $expectedIndex) 'Wrong PyTorch channel'
        Assert-PetTest ($plan.CheckArguments[3] -eq $expectedDevice) 'Wrong device check'
        Assert-PetTest ($plan.CheckArguments[5] -eq $report) 'Report path lost spaces/Unicode'
        Assert-PetTest ($plan.Environment -eq (Join-Path $env:USERPROFILE '.venvs\pet-tracer')) 'Environment must remain outside the repo'
    }

    # Verify real native argument passing and nonzero exits without installing packages.
    $echo = Join-Path $testRoot 'argument echo.py'
    Set-Content -LiteralPath $echo -Encoding ASCII -Value 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2],encoding="utf-8")'
    $received = Join-Path $testRoot 'received args.txt'
    Invoke-PetNative $TestPython @($echo, $received, $testRoot)
    Assert-PetTest ((Get-Content -LiteralPath $received -Raw -Encoding UTF8) -eq $testRoot) 'Native paths were split or corrupted'
    Invoke-PetNative $TestPython @('-c', "import sys; sys.stderr.write('normal progress')")
    $failed = $false
    try { Invoke-PetNative $TestPython @('-c', 'raise SystemExit(17)') } catch { $failed = $true }
    Assert-PetTest $failed 'A failed child process was treated as success'
    $failed = $false
    try { Invoke-PetNative (Join-Path $testRoot 'missing.exe') @() } catch { $failed = $true }
    Assert-PetTest $failed 'A missing executable was treated as success'

    # Test the real version/architecture probe against the CI Python executable.
    function Get-PetPythonCandidates { return @(@{ Exe = $TestPython; Prefix = @() }) }
    Assert-PetTest ((Find-PetPython) -eq $TestPython) 'Did not resolve 64-bit Python 3.11'
    function Get-PetPythonCandidates { return @(@{ Exe = $TestPython; Prefix = @('-c', 'raise SystemExit(7)') }) }
    Assert-PetTest ($null -eq (Find-PetPython)) 'Accepted a broken Python interpreter'

    # From here all external installation commands are mocked.
    $script:Calls = @()
    $script:Fail = $false
    $script:Bootstrap = 0
    function Invoke-PetNative {
        param([string]$Executable, [string[]]$Arguments)
        $script:Calls += @{ Exe = $Executable; Args = $Arguments }
        if ($script:Fail) { throw 'Mock package install failed' }
    }
    function Find-PetPython { return $TestPython }
    function Get-PetManager { return 'C:\mock\pymanager.exe' }
    Assert-PetTest ((Install-PetPython) -eq $TestPython) 'Python installation did not resolve its executable'
    Assert-PetTest ($script:Calls.Count -eq 1 -and $script:Calls[0].Args[1] -eq '3.11-64') 'Python install did not request 64-bit 3.11'
    function Install-PetPython { $script:Bootstrap++; return $TestPython }
    foreach ($choice in @('CPU', 'NVIDIA')) {
        $script:Calls = @()
        $code = Invoke-PetSetup -Root $testRoot -Choice $choice -LogDirectory (Join-Path $testRoot 'logs')
        Assert-PetTest ($code -eq 0) 'Successful setup returned failure'
        Assert-PetTest ($script:Calls.Count -eq 3) 'Setup/import/device verification did not all run'
        Assert-PetTest ($script:Calls[0].Exe -eq $TestPython) 'Setup used the wrong interpreter'
        Assert-PetTest ($script:Calls[1].Args[1] -eq '-c') 'Installed-package import check is missing'
    }
    $script:Fail = $true
    $script:Calls = @()
    $code = Invoke-PetSetup -Root $testRoot -Choice 'CPU' -LogDirectory (Join-Path $testRoot 'logs')
    Assert-PetTest ($code -ne 0) 'Installation failure returned success'
    Assert-PetTest ($script:Calls.Count -eq 1) 'Verification continued after installation failure'
    $script:Fail = $false
    function Find-PetPython { return $null }
    $code = Invoke-PetSetup -Root $testRoot -Choice 'CPU' -LogDirectory (Join-Path $testRoot 'logs')
    Assert-PetTest ($code -eq 0 -and $script:Bootstrap -eq 1) 'Missing-Python bootstrap was not used'

    # Menu cancellation must not install anything.
    function Read-PetDevice { return 'Cancel' }
    $script:Calls = @()
    $code = Invoke-PetSetup -Root $testRoot -Choice '' -LogDirectory (Join-Path $testRoot 'logs')
    Assert-PetTest ($code -eq 0 -and $script:Calls.Count -eq 0) 'Cancellation started installation'
    Write-Host 'Windows setup checks passed'
} finally {
    Set-Location $oldLocation
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
