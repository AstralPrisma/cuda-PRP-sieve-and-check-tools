[CmdletBinding()]
param(
    [string]$SuiteVersion = "v2026.09.6",
    [string]$RawDirectory = (Join-Path $PSScriptRoot "..\release-assets\raw"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\release-assets\packages"),
    [string]$BuildMetadataPath = ""
)

$ErrorActionPreference = "Stop"
$repoDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$rawPath = [IO.Path]::GetFullPath($RawDirectory)
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$releaseRoot = [IO.Path]::GetFullPath((Join-Path $repoDirectory "release-assets"))
$stagingPath = Join-Path $releaseRoot "package-staging"
$architectures = @("sm_86", "sm_89", "sm_100", "sm_120")
$versions = [ordered]@{
    GFPS = "4.4"
    GSRPS = "2.3"
    GFPPS = "1.0"
    GFNSV = "1.0"
    GSRSV = "2.0"
    GNCWSV = "1.0"
}
$utf8 = New-Object Text.UTF8Encoding($false)
$metadataPath = if ($BuildMetadataPath) { [IO.Path]::GetFullPath($BuildMetadataPath) } else { Join-Path $rawPath "build-metadata.json" }
if (-not [IO.File]::Exists($metadataPath)) {
    throw "Missing build provenance: $metadataPath. Record actual compiler, flags, source and binary hashes, and runtime evidence before packaging."
}
$buildMetadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if (-not $buildMetadata.builds) { throw "Build metadata must contain a builds object" }
$stagingCreated = $false

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

function Get-SourceFiles([string]$Tool) {
    $sourceDirectory = Join-Path $repoDirectory "$Tool\src"
    $records = [ordered]@{}
    Get-ChildItem -LiteralPath $sourceDirectory -File |
        Where-Object { $_.Extension -in @(".cu", ".cuh", ".h", ".hpp") } |
        Sort-Object Name |
        ForEach-Object { $records["src/$($_.Name)"] = Get-Sha256 $_.FullName }
    return $records
}

function Invoke-External([scriptblock]$Command, [string]$Description) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

Push-Location $repoDirectory
try {
    $commit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $commit) { throw "Cannot resolve the source commit" }
    if (git status --porcelain) {
        throw "Working tree changes or untracked source files are present; review and commit them before release packaging"
    }
    $sourceEpoch = (& git show -s --format=%ct HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceEpoch -notmatch '^\d+$') { throw "Cannot resolve source commit timestamp" }

    # Validate the full provenance matrix before replacing any package directory.
    foreach ($tool in $versions.Keys) {
        $sourceFiles = Get-SourceFiles $tool
        foreach ($platform in @("linux", "windows")) {
            $key = "$tool/$platform"
            $property = $buildMetadata.builds.PSObject.Properties[$key]
            if (-not $property) { throw "Missing build metadata for $key" }
            $record = $property.Value
            if (-not $record.compiler -or -not $record.cuda_toolkit -or -not $record.flags -or -not $record.source_files) {
                throw "Incomplete compiler/flags/source metadata for $key"
            }
            if ($record.cuda_toolkit -notmatch '^13\.3(?:\.|$)') { throw "This archive naming scheme requires CUDA 13.3: $key" }
            $recordedSourceNames = @($record.source_files.PSObject.Properties.Name)
            if ($recordedSourceNames.Count -ne $sourceFiles.Count) { throw "Source metadata file count mismatch: $key" }
            foreach ($name in $sourceFiles.Keys) {
                $sourceProperty = $record.source_files.PSObject.Properties[$name]
                if (-not $sourceProperty -or $sourceProperty.Value -ne $sourceFiles[$name]) { throw "Source hash mismatch: $key $name" }
            }
            $suffix = if ($platform -eq "windows") { ".exe" } else { "" }
            if (@($record.binaries).Count -ne $architectures.Count) { throw "Binary metadata count mismatch: $key" }
            foreach ($arch in $architectures) {
                $binaryName = "${tool}_${arch}${suffix}"
                $entries = @($record.binaries | Where-Object { $_.file -eq $binaryName })
                if ($entries.Count -ne 1) { throw "Missing or duplicate binary record: $key $binaryName" }
                $entry = $entries[0]
                if ($entry.runtime_tested -isnot [bool] -or -not $entry.evidence) { throw "Missing test status/evidence: $key $binaryName" }
                $binaryPath = Join-Path (Join-Path $rawPath $platform) $binaryName
                if ((Get-Sha256 $binaryPath) -ne $entry.sha256) { throw "Binary hash mismatch: $key $binaryName" }
                if ($entry.runtime_tested -and -not $record.test_hardware) { throw "Runtime-tested builds require actual test hardware: $key" }
            }
        }
    }

    Remove-DirectoryIfPresent $stagingPath
    Remove-DirectoryIfPresent $outputPath
    [IO.Directory]::CreateDirectory($stagingPath) | Out-Null
    $stagingCreated = $true
    [IO.Directory]::CreateDirectory($outputPath) | Out-Null

    $packageRecords = @()
    foreach ($tool in $versions.Keys) {
        $version = $versions[$tool]
        $sourceFiles = Get-SourceFiles $tool
        foreach ($platform in @("linux", "windows")) {
            $buildRecord = $buildMetadata.builds.PSObject.Properties["$tool/$platform"].Value
            $suffix = if ($platform -eq "windows") { ".exe" } else { "" }
            $hostPlatform = if ($platform -eq "windows") { "windows-x86_64" } else { "linux-x86_64" }
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
                $binaryMetadata = @($buildRecord.binaries | Where-Object { $_.file -eq $binaryName })[0]
                [IO.File]::Copy($sourceBinary, (Join-Path $binStage $binaryName), $true)
                $binaryRecords += [ordered]@{
                    file = "bin/$binaryName"
                    target = $arch
                    sha256 = Get-Sha256 $sourceBinary
                    runtime_tested = $binaryMetadata.runtime_tested
                    evidence = $binaryMetadata.evidence
                }
            }

            # Include component-level supporting Markdown, e.g. GFPPS's sample
            # validation table linked by its README, without local test logs.
            Get-ChildItem -LiteralPath (Join-Path $repoDirectory $tool) -File -Filter "*.md" | ForEach-Object {
                [IO.File]::Copy($_.FullName, (Join-Path $componentStage $_.Name), $true)
            }
            [IO.File]::Copy((Join-Path $repoDirectory "LICENSE"), (Join-Path $componentStage "LICENSE"), $true)
            [IO.File]::Copy((Join-Path $repoDirectory "THIRD_PARTY_NOTICES.md"), (Join-Path $componentStage "THIRD_PARTY_NOTICES.md"), $true)
            [IO.File]::Copy((Join-Path $repoDirectory "docs\correctness.md"), (Join-Path $componentStage "CORRECTNESS.md"), $true)
            Get-ChildItem -LiteralPath (Join-Path $repoDirectory "LICENSES") -File | ForEach-Object {
                [IO.File]::Copy($_.FullName, (Join-Path $licensesStage $_.Name), $true)
            }

            $compiler = $buildRecord.compiler
            $testedTargets = @($binaryRecords | Where-Object { $_.runtime_tested } | ForEach-Object { $_.target })
            $buildInfo = @(
                "Suite release: $SuiteVersion"
                "Component: $tool $version"
                "Source commit: $commit"
                "Platform: $hostPlatform"
                "Compiler: $compiler"
                "CUDA Toolkit: $($buildRecord.cuda_toolkit)"
                "Compiler flags: $($buildRecord.flags -join ' ')"
                "CUDA targets: $($architectures -join ', ')"
                "Runtime-tested targets: $($testedTargets -join ', ')"
                "Test GPU: $($buildRecord.test_hardware)"
                "Source SHA-256: $(Get-Sha256 (Join-Path $repoDirectory "$tool\src\$tool.cu"))"
                "Source file hashes: $($sourceFiles | ConvertTo-Json -Compress)"
                "Runtime test evidence: $($buildRecord.binaries | ConvertTo-Json -Depth 5 -Compress)"
                "Each binary contains native cubins and compiler-emitted PTX matched to its named target."
                "Separate binaries are used instead of one universal executable for all GPU generations."
            )
            [IO.File]::WriteAllLines((Join-Path $componentStage "BUILDINFO.txt"), $buildInfo, $utf8)

            $archiveName = "$($tool.ToLowerInvariant())-$version-$hostPlatform-cuda13.3$archiveExtension"
            $archivePath = Join-Path $outputPath $archiveName
            if ($platform -eq "windows") {
                Invoke-External { & tar.exe -a -c -f $archivePath -C $platformStage $rootName } "Windows ZIP packaging"
            } else {
                $stageWsl = (& wsl.exe -e wslpath -a $platformStage).Trim()
                $archiveWsl = (& wsl.exe -e wslpath -a $archivePath).Trim()
                Invoke-External {
                    & wsl.exe -e env XZ_OPT=-T1 tar --sort=name "--mtime=@$sourceEpoch" --owner=0 --group=0 --numeric-owner `
                        "--mode=u+rwX,go+rX,go-w" -cJf $archiveWsl -C $stageWsl $rootName
                } "Linux tar.xz packaging"
            }

            $packageRecords += [ordered]@{
                file = $archiveName
                tool = $tool
                component_version = $version
                source_commit = $commit
                source_sha256 = Get-Sha256 (Join-Path $repoDirectory "$tool\src\$tool.cu")
                source_files = $sourceFiles
                artifact_sha256 = Get-Sha256 $archivePath
                compiler = $compiler
                compiler_flags = @($buildRecord.flags)
                cuda_toolkit = $buildRecord.cuda_toolkit
                targets = $architectures
                host = $hostPlatform
                runtime_tested_targets = $testedTargets
                test_hardware = $buildRecord.test_hardware
                binaries = $binaryRecords
            }
        }
    }

    $sourceBase = "cuda-prp-sieve-and-check-tools-$SuiteVersion-source"
    $sourceTar = Join-Path $outputPath "$sourceBase.tar"
    Invoke-External { & git archive --format=tar "--prefix=$sourceBase/" -o $sourceTar HEAD } "Source archive"
    $sourceTarWsl = (& wsl.exe -e wslpath -a $sourceTar).Trim()
    # Keep compression workspace bounded on a host also running prime searches.
    Invoke-External { & wsl.exe -e xz -T1 -f -6 $sourceTarWsl } "Source archive compression"

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
    if ($stagingCreated) { Remove-DirectoryIfPresent $stagingPath }
}
