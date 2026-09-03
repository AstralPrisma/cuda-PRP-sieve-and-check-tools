[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Architectures = @("sm_86", "sm_89", "sm_100", "sm_120")
)

$ErrorActionPreference = "Stop"
$toolDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$sourcePath = Join-Path $toolDirectory "src\GSRPS.cu"
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$nvcc = if ($env:NVCC) { $env:NVCC } else { "nvcc.exe" }
$boostHeader = if ($env:BOOST_ROOT) { Join-Path $env:BOOST_ROOT "boost\multiprecision\cpp_int.hpp" } else { "" }
if (-not $env:BOOST_ROOT -or -not [IO.File]::Exists($boostHeader)) {
    throw "Set BOOST_ROOT to the directory containing boost\multiprecision\cpp_int.hpp"
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
$includeArgs = @("-I$env:BOOST_ROOT")

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Push-Location $toolDirectory
try {
    foreach ($arch in $Architectures) {
        $buildId = "$($sourceHash.Substring(0, 20))-O3-cxx17-ptds-$arch"
        $buildDefine = '-DGSRPS_BUILD_ID=\"{0}\"' -f $buildId
        & $nvcc -O3 -std=c++17 "-arch=$arch" --default-stream per-thread `
            -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4038 `
            $buildDefine @includeArgs `
            -o (Join-Path $outputPath "GSRPS_$arch.exe") "src\GSRPS.cu"
        if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $arch" }
    }
}
finally {
    Pop-Location
}
