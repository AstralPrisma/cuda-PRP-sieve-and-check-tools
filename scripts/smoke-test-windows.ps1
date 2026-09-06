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

function Invoke-Gfpps([string[]]$Arguments, [string]$Expected) {
    $output = @(& (Join-Path $binPath "GFPPS_$Architecture.exe") @Arguments)
    if ($LASTEXITCODE -ne 0) { throw "GFPPS exited with code $LASTEXITCODE" }
    $output | Write-Output
    if (($output -join "`n") -notmatch "verification=cpp_int-match, result=$Expected(?:\r?\n|$)") {
        throw "GFPPS did not return the expected verified status: $Expected"
    }
}

try {
    Invoke-Checked (Join-Path $binPath "GFPS_$Architecture.exe") @("--selftest")
    Invoke-Checked (Join-Path $binPath "GSRPS_$Architecture.exe") @("--selftest")
    Invoke-Gfpps @("--check", "1*3!+1", "--verify-cpp-int") "PRP"
    Invoke-Gfpps @("--check", "3*5!+1", "--no-graphs", "--verify-cpp-int") "COMPOSITE"
    $gfppsCheckpoint = Join-Path $scratch "gfpps.ckpt"
    Invoke-Gfpps @("--check", "17*31#+1", "--max-bits", "16",
        "--checkpoint", $gfppsCheckpoint, "--verify-cpp-int") "PARTIAL"
    Invoke-Gfpps @("--check", "17*31#+1", "--checkpoint", $gfppsCheckpoint,
        "--resume-checkpoint", "--verify-cpp-int") "(?:PRP|COMPOSITE)"
    $resumePath = Join-Path $scratch "gfnsv-resume.txt"
    $directPath = Join-Path $scratch "gfnsv-direct.txt"
    Invoke-Checked (Join-Path $binPath "GFNSV_$Architecture.exe") @(
        "--n", "3", "--bmin", "2", "--bmax", "100", "--pmax", "100",
        "--batch", "16", "--out", $resumePath, "--quiet"
    )
    Invoke-Checked (Join-Path $binPath "GFNSV_$Architecture.exe") @(
        "--resume", "--out", $resumePath, "--pmax", "1000", "--full-roots", "--quiet"
    )
    Invoke-Checked (Join-Path $binPath "GFNSV_$Architecture.exe") @(
        "--n", "3", "--bmin", "2", "--bmax", "100", "--pmax", "1000",
        "--out", $directPath, "--quiet"
    )
    $resumedBases = @(Get-Content -LiteralPath $resumePath | Where-Object { $_ -and -not $_.StartsWith('#') })
    $directBases = @(Get-Content -LiteralPath $directPath | Where-Object { $_ -and -not $_.StartsWith('#') })
    if (($resumedBases -join "`n") -cne ($directBases -join "`n")) {
        throw "GFNSV resumed/full-root and direct/paired-root survivors differ"
    }
    Invoke-Checked (Join-Path $binPath "GFNSV_$Architecture.exe") @("--checkpoint-info", $resumePath)
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
