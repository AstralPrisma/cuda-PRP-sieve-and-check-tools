[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Architectures = @("sm_86", "sm_89", "sm_100", "sm_120")
)

$ErrorActionPreference = "Stop"
$sourceDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\src"))
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$nvcc = if ($env:NVCC) { $env:NVCC } else { "nvcc.exe" }
if (-not $env:BOOST_ROOT) { throw "Set BOOST_ROOT to the directory containing boost\multiprecision\cpp_int.hpp" }
$boostPath = [IO.Path]::GetFullPath($env:BOOST_ROOT)
if (-not [IO.File]::Exists((Join-Path $boostPath "boost\multiprecision\cpp_int.hpp"))) {
    throw "BOOST_ROOT does not contain boost\multiprecision\cpp_int.hpp"
}
if ([IO.Path]::GetPathRoot($sourceDirectory) -ine [IO.Path]::GetPathRoot($boostPath)) {
    throw "Place Boost headers on the same drive as this source tree so published binaries can use relative include paths"
}
$sourceUri = [Uri]($sourceDirectory.TrimEnd('\') + '\')
$boostUri = [Uri]($boostPath.TrimEnd('\') + '\')
$boostRelative = [Uri]::UnescapeDataString($sourceUri.MakeRelativeUri($boostUri).ToString()).Replace('/', '\')
if (-not $boostRelative) { $boostRelative = '.' }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Push-Location $sourceDirectory
try {
    foreach ($arch in $Architectures) {
        if ($arch -notmatch '^sm_\d+$') { throw "Invalid architecture: $arch" }
        & $nvcc -O3 -std=c++17 --threads 1 "-arch=$arch" --default-stream per-thread `
            -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4038 `
            "-I$boostRelative" GFPPS.cu -o (Join-Path $outputPath "GFPPS_$arch.exe")
        if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $arch" }
    }
}
finally { Pop-Location }
