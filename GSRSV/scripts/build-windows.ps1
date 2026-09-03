[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Architectures = @("sm_86", "sm_89", "sm_100", "sm_120")
)

$ErrorActionPreference = "Stop"
$toolDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$nvcc = if ($env:NVCC) { $env:NVCC } else { "nvcc.exe" }

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Push-Location $toolDirectory
try {
    foreach ($arch in $Architectures) {
        & $nvcc -O3 -std=c++17 "-arch=$arch" --default-stream per-thread `
            -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4038 `
            -o (Join-Path $outputPath "GSRSV_$arch.exe") "src\GSRSV.cu"
        if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $arch" }
    }
}
finally {
    Pop-Location
}
