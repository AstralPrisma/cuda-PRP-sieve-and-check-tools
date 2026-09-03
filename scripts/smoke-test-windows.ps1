[CmdletBinding()]
param(
    [string]$BinDirectory = (Join-Path $PSScriptRoot "..\release-assets"),
    [string]$Architecture = "sm_89"
)

$ErrorActionPreference = "Stop"
$binPath = [IO.Path]::GetFullPath($BinDirectory)
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("cuda-prp-smoke-" + [Guid]::NewGuid().ToString("N"))
[IO.Directory]::CreateDirectory($scratch) | Out-Null

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program exited with code $LASTEXITCODE" }
}

try {
    Invoke-Checked (Join-Path $binPath "GFPS_$Architecture.exe") @("--selftest")
    Invoke-Checked (Join-Path $binPath "GSRPS_$Architecture.exe") @("--selftest")
    Invoke-Checked (Join-Path $binPath "GSRSV_$Architecture.exe") @(
        "--kmin", "1", "--kmax", "100", "--base", "2", "--exp", "100",
        "--termtype", "1", "--pmin", "2", "--pmax", "10000",
        "--cpu-small-prime", "2", "--prime-generator", "segmented",
        "--outputterms", (Join-Path $scratch "gsrsv.txt"), "--verify", "--quiet"
    )
    Invoke-Checked (Join-Path $binPath "GNCWSV_$Architecture.exe") @(
        "--amin", "2", "--amax", "40", "--base", "3", "--mode", "1",
        "--pmin", "2", "--pmax", "1000", "--prime-generator", "segmented",
        "--outputterms", (Join-Path $scratch "gncwsv.txt"), "--verify", "--quiet"
    )
    Write-Output "smoke tests passed for $Architecture"
}
finally {
    if ([IO.Directory]::Exists($scratch)) { [IO.Directory]::Delete($scratch, $true) }
}
