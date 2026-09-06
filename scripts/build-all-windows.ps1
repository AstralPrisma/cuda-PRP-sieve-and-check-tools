[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\release-assets"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Architectures = @("sm_86", "sm_89", "sm_100", "sm_120")
)

$ErrorActionPreference = "Stop"
$repoDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)

foreach ($tool in @("GFPS", "GSRPS", "GFPPS", "GFNSV", "GSRSV", "GNCWSV")) {
    & (Join-Path $repoDirectory "$tool\scripts\build-windows.ps1") `
        -OutputDirectory $outputPath -Architectures $Architectures
}
