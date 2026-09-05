[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Architectures = @("sm_86", "sm_89", "sm_100", "sm_120")
)

$ErrorActionPreference = "Stop"
$toolDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$sourcePath = Join-Path $toolDirectory "src\GFNSV.cu"
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$nvcc = if ($env:NVCC) { $env:NVCC } else { "nvcc.exe" }
$boostHeader = if ($env:BOOST_ROOT) { Join-Path $env:BOOST_ROOT "boost\multiprecision\cpp_int.hpp" } else { "" }
if (-not $env:BOOST_ROOT -or -not [IO.File]::Exists($boostHeader)) {
    throw "Set BOOST_ROOT to the directory containing boost\multiprecision\cpp_int.hpp"
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Push-Location $toolDirectory
try {
    foreach ($arch in $Architectures) {
        if ($arch -notin @("sm_86", "sm_89", "sm_100", "sm_120")) {
            throw "Unsupported architecture: $arch"
        }
        & $nvcc -O3 -std=c++17 --threads 1 "-arch=$arch" --default-stream per-thread `
            -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4038 `
            "-I$env:BOOST_ROOT" -o (Join-Path $outputPath "GFNSV_$arch.exe") "src\GFNSV.cu"
        if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $arch" }
    }
}
finally {
    Pop-Location
}
