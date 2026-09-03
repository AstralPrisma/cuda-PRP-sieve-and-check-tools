[CmdletBinding()]
param(
    [string]$SuiteVersion = "v2026.09.0",
    [string]$RawDirectory = (Join-Path $PSScriptRoot "..\release-assets\raw"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\release-assets\packages")
)

$ErrorActionPreference = "Stop"
$repoDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$rawPath = [IO.Path]::GetFullPath($RawDirectory)
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$releaseRoot = [IO.Path]::GetFullPath((Join-Path $repoDirectory "release-assets"))
$stagingPath = Join-Path $releaseRoot "package-staging"
$architectures = @("sm_86", "sm_89", "sm_100", "sm_120")
$versions = [ordered]@{
    GFPS = "4.0"
    GSRPS = "2.0"
    GSRSV = "2.0"
    GNCWSV = "1.0"
}
$utf8 = New-Object Text.UTF8Encoding($false)

function Assert-ChildPath([string]$Path, [string]$Parent) {
    $parentPrefix = $Parent.TrimEnd('\') + '\'
    if (-not $Path.StartsWith($parentPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside $Parent`: $Path"
    }
}

function Remove-DirectoryIfPresent([string]$Path) {
    if ([IO.Directory]::Exists($Path)) {
        Assert-ChildPath $Path $releaseRoot
        [IO.Directory]::Delete($Path, $true)
    }
}

function Get-Sha256([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $stream.Dispose()
        $sha.Dispose()
    }
}

function Invoke-External([scriptblock]$Command, [string]$Description) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

Push-Location $repoDirectory
try {
    $commit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $commit) { throw "Cannot resolve the source commit" }
    if (git status --porcelain --untracked-files=no) {
        throw "Tracked working tree changes are present; commit them before packaging"
    }

    Remove-DirectoryIfPresent $stagingPath
    Remove-DirectoryIfPresent $outputPath
    [IO.Directory]::CreateDirectory($stagingPath) | Out-Null
    [IO.Directory]::CreateDirectory($outputPath) | Out-Null

    $packageRecords = @()
    foreach ($tool in $versions.Keys) {
        $version = $versions[$tool]
        foreach ($platform in @("linux", "windows")) {
            $suffix = if ($platform -eq "windows") { ".exe" } else { "" }
            $host = if ($platform -eq "windows") { "windows-x86_64" } else { "linux-x86_64" }
            $archiveExtension = if ($platform -eq "windows") { ".zip" } else { ".tar.xz" }
            $rootName = "$tool-$version"
            $platformStage = Join-Path $stagingPath $platform
            $componentStage = Join-Path $platformStage $rootName
            $binStage = Join-Path $componentStage "bin"
            $licensesStage = Join-Path $componentStage "LICENSES"
            [IO.Directory]::CreateDirectory($binStage) | Out-Null
            [IO.Directory]::CreateDirectory($licensesStage) | Out-Null

            $binaryRecords = @()
            foreach ($arch in $architectures) {
                $binaryName = "${tool}_${arch}${suffix}"
                $sourceBinary = Join-Path (Join-Path $rawPath $platform) $binaryName
                if (-not [IO.File]::Exists($sourceBinary)) { throw "Missing binary: $sourceBinary" }
                [IO.File]::Copy($sourceBinary, (Join-Path $binStage $binaryName), $true)
                $binaryRecords += [ordered]@{
                    file = "bin/$binaryName"
                    target = $arch
                    sha256 = Get-Sha256 $sourceBinary
                    runtime_tested = ($arch -eq "sm_89")
                }
            }

            [IO.File]::Copy((Join-Path $repoDirectory "$tool\README.md"), (Join-Path $componentStage "README.md"), $true)
            [IO.File]::Copy((Join-Path $repoDirectory "LICENSE"), (Join-Path $componentStage "LICENSE"), $true)
            [IO.File]::Copy((Join-Path $repoDirectory "THIRD_PARTY_NOTICES.md"), (Join-Path $componentStage "THIRD_PARTY_NOTICES.md"), $true)
            [IO.File]::Copy((Join-Path $repoDirectory "docs\correctness.md"), (Join-Path $componentStage "CORRECTNESS.md"), $true)
            Get-ChildItem -LiteralPath (Join-Path $repoDirectory "LICENSES") -File | ForEach-Object {
                [IO.File]::Copy($_.FullName, (Join-Path $licensesStage $_.Name), $true)
            }

            $compiler = if ($platform -eq "windows") {
                "MSVC 19.51.36248 (x64), nvcc 13.3.33"
            } else {
                "GCC 13.3.0 (x86_64), nvcc 13.3.33"
            }
            $buildInfo = @(
                "Suite release: $SuiteVersion"
                "Component: $tool $version"
                "Source commit: $commit"
                "Platform: $host"
                "Compiler: $compiler"
                "CUDA targets: $($architectures -join ', ')"
                "Runtime-tested target: sm_89"
                "Test GPU: NVIDIA GeForce RTX 4060 Laptop GPU"
                "Source SHA-256: $(Get-Sha256 (Join-Path $repoDirectory "$tool\src\$tool.cu"))"
                "Built with C++17, -O3, and per-thread default-stream semantics."
                "The binaries contain native cubins and no universal PTX fallback."
            )
            [IO.File]::WriteAllLines((Join-Path $componentStage "BUILDINFO.txt"), $buildInfo, $utf8)

            $archiveName = "$($tool.ToLowerInvariant())-$version-$host-cuda13.3$archiveExtension"
            $archivePath = Join-Path $outputPath $archiveName
            if ($platform -eq "windows") {
                Invoke-External { & tar.exe -a -c -f $archivePath -C $platformStage $rootName } "Windows ZIP packaging"
            } else {
                $stageWsl = (& wsl.exe -e wslpath -a $platformStage).Trim()
                $archiveWsl = (& wsl.exe -e wslpath -a $archivePath).Trim()
                Invoke-External {
                    & wsl.exe -e tar --sort=name "--mtime=UTC 2026-09-03" --owner=0 --group=0 --numeric-owner `
                        "--mode=u+rwX,go+rX,go-w" -cJf $archiveWsl -C $stageWsl $rootName
                } "Linux tar.xz packaging"
            }

            $packageRecords += [ordered]@{
                file = $archiveName
                tool = $tool
                component_version = $version
                source_commit = $commit
                source_sha256 = Get-Sha256 (Join-Path $repoDirectory "$tool\src\$tool.cu")
                artifact_sha256 = Get-Sha256 $archivePath
                cuda_toolkit = "13.3.33"
                targets = $architectures
                host = $host
                runtime_tested_targets = @("sm_89")
                test_hardware = "NVIDIA GeForce RTX 4060 Laptop GPU"
                binaries = $binaryRecords
            }
        }
    }

    $sourceBase = "cuda-prp-sieve-and-check-tools-$SuiteVersion-source"
    $sourceTar = Join-Path $outputPath "$sourceBase.tar"
    Invoke-External { & git archive --format=tar "--prefix=$sourceBase/" -o $sourceTar HEAD } "Source archive"
    $sourceTarWsl = (& wsl.exe -e wslpath -a $sourceTar).Trim()
    Invoke-External { & wsl.exe -e xz -f -9 $sourceTarWsl } "Source archive compression"

    $manifest = [ordered]@{
        suite_version = $SuiteVersion
        source_commit = $commit
        generated_utc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
        packages = $packageRecords
    }
    $manifestPath = Join-Path $outputPath "manifest.json"
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8) + "`n", $utf8)

    [string[]]$checksumLines = Get-ChildItem -LiteralPath $outputPath -File |
        Where-Object { $_.Name -ne "SHA256SUMS" } |
        Sort-Object Name |
        ForEach-Object { "$(Get-Sha256 $_.FullName)  $($_.Name)" }
    [IO.File]::WriteAllLines((Join-Path $outputPath "SHA256SUMS"), $checksumLines, $utf8)
}
finally {
    Pop-Location
    Remove-DirectoryIfPresent $stagingPath
}
