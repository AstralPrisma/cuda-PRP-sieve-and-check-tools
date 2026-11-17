[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build")
)

# PRPS is one unified binary: all four SM targets are embedded in a single
# executable and the CUDA driver picks the right one at run time.
$ErrorActionPreference = "Stop"
$toolDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$sourcePath = Join-Path $toolDirectory "src\PRPS.cu"
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$nvcc = if ($env:NVCC) { $env:NVCC } else { "nvcc.exe" }
$boostHeader = if ($env:BOOST_ROOT) { Join-Path $env:BOOST_ROOT "boost\multiprecision\cpp_int.hpp" } else { "" }
if (-not $env:BOOST_ROOT -or -not [IO.File]::Exists($boostHeader)) {
    throw "Set BOOST_ROOT to the directory containing boost\multiprecision\cpp_int.hpp"
}
if (-not $env:GMP_ROOT -or -not [IO.File]::Exists((Join-Path $env:GMP_ROOT "gmp.h"))) {
    throw "Set GMP_ROOT to the directory containing gmp.h"
}
$gmpLib = Join-Path $env:GMP_ROOT "win64\libgmp-gw1.lib"
if (-not [IO.File]::Exists($gmpLib)) {
    throw "Missing GMP import library: $gmpLib"
}
$sha256 = [Security.Cryptography.SHA256]::Create()
$sourceStream = [IO.File]::OpenRead($sourcePath)
try {
    $sourceHash = -join ($sha256.ComputeHash($sourceStream) | ForEach-Object { $_.ToString("x2") })
}
finally {
    $sourceStream.Dispose()
    $sha256.Dispose()
}
$gencode = @()
foreach ($arch in @("sm_86", "sm_89", "sm_100", "sm_120")) {
    $gencode += "-gencode"
    $gencode += "arch=compute_$($arch.Substring(3)),code=$arch"
}
$buildDefine = '-DPRPS_BUILD_ID=\"{0}-O3-cxx17-ptds-unified\"' -f $sourceHash.Substring(0, 20)

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Push-Location $toolDirectory
try {
    & $nvcc -O3 -std=c++17 @gencode --default-stream per-thread `
        -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4038 `
        $buildDefine "-I$env:BOOST_ROOT" "-I$env:GMP_ROOT" `
        -o (Join-Path $outputPath "PRPS.exe") "src\PRPS.cu" $gmpLib
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed" }
    Copy-Item (Join-Path $env:GMP_ROOT "win64\libgmp-gw1.dll") $outputPath -Force
}
finally {
    Pop-Location
}
