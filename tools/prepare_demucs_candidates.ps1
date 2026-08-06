[CmdletBinding()]
param(
    [string[]]$ContractFile = @(
        "app/src/main/assets/benchmark-contracts/htdemucs_4s_waveform_7p8s_onnx.json",
        "app/src/main/assets/benchmark-contracts/htdemucs_6s_waveform_7p8s_onnx.json"
    ),
    [string]$DestinationRoot = "models/demucs",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$destination = if ([System.IO.Path]::IsPathRooted($DestinationRoot)) {
    [System.IO.Path]::GetFullPath($DestinationRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $DestinationRoot))
}
$destinationPrefix = $destination.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
New-Item -ItemType Directory -Force -Path $destination | Out-Null

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Resolve-ContractFile([string]$Path) {
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $repoRoot $Path
    }
    $resolved = Resolve-Path -LiteralPath $candidate
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Contract is not a file: $resolved"
    }
    return $resolved.Path
}

function Resolve-ArtifactTarget([string]$LocalPath) {
    if ([string]::IsNullOrWhiteSpace($LocalPath) -or
        [System.IO.Path]::IsPathRooted($LocalPath)) {
        throw "Artifact localPath must be a non-empty relative path: $LocalPath"
    }
    $nativePath = $LocalPath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    $target = [System.IO.Path]::GetFullPath((Join-Path $destination $nativePath))
    if (-not $target.StartsWith($destinationPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Artifact path escapes destination root: $LocalPath"
    }
    return $target
}

function Test-Artifact([object]$Artifact, [string]$Path) {
    $file = Get-Item -LiteralPath $Path
    $expectedLength = [long]$Artifact.byteSize
    if ($file.Length -ne $expectedLength) {
        return $false
    }
    return (Get-Sha256 $file.FullName) -ceq [string]$Artifact.sha256
}

function Install-Artifact([object]$Artifact) {
    $target = Resolve-ArtifactTarget ([string]$Artifact.localPath)
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        if (Test-Artifact $Artifact $target) {
            Write-Host "Verified $($Artifact.localPath)"
            return
        }
        if (-not $Force) {
            throw "Existing artifact does not match its contract: $target (use -Force to replace after download verification)"
        }
    }

    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$target.download"
    if ($Force -and (Test-Path -LiteralPath $temporary)) {
        Remove-Item -LiteralPath $temporary -Force
    }

    Write-Host "Downloading $($Artifact.url)"
    & curl.exe `
        --ssl-no-revoke `
        --location `
        --fail `
        --retry 5 `
        --continue-at - `
        --output $temporary `
        ([string]$Artifact.url)
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE for $($Artifact.url)"
    }
    if (-not (Test-Artifact $Artifact $temporary)) {
        $actualLength = (Get-Item -LiteralPath $temporary).Length
        $actualSha256 = Get-Sha256 $temporary
        throw "Downloaded artifact identity mismatch for $($Artifact.localPath): length=$actualLength sha256=$actualSha256"
    }
    Move-Item -LiteralPath $temporary -Destination $target -Force
    Write-Host "Installed $($Artifact.localPath)"
}

$artifactsByPath = [ordered]@{}
foreach ($contractPath in $ContractFile) {
    $resolvedContract = Resolve-ContractFile $contractPath
    $contract = Get-Content -LiteralPath $resolvedContract -Raw | ConvertFrom-Json
    if ($contract.contractSchemaVersion -ne 3 -or $contract.contractKind -cne "tensor-only-candidate") {
        throw "Expected a schema v3 tensor-only candidate contract: $resolvedContract"
    }
    $artifacts = @(
        $contract.conversionSource,
        $contract.conversion.exporterInput,
        $contract.conversion.repositoryLicense,
        $contract.conversion.modelCard,
        $contract.upstream.weight,
        $contract.upstream.metadata,
        $contract.upstream.bagManifest
    )
    foreach ($artifact in $artifacts) {
        $key = [string]$artifact.localPath
        if ($artifactsByPath.Contains($key)) {
            $existing = $artifactsByPath[$key]
            if ($existing.sha256 -cne $artifact.sha256 -or
                [long]$existing.byteSize -ne [long]$artifact.byteSize) {
                throw "Contracts disagree on artifact identity for $key"
            }
        } else {
            $artifactsByPath[$key] = $artifact
        }
    }
}

foreach ($artifact in $artifactsByPath.Values) {
    Install-Artifact $artifact
}

Write-Host "Prepared $($artifactsByPath.Count) pinned Demucs artifacts under $destination"
