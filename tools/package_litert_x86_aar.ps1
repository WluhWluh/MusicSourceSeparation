param(
    [Parameter(Mandatory = $true)]
    [string]$SharedLibrary,

    [Parameter(Mandatory = $true)]
    [string]$LicenseFile,

    [string]$OutputAar = "app/libs/litert-2.1.5-x86.aar"
)

$ErrorActionPreference = "Stop"

$sharedLibraryPath = (Resolve-Path -LiteralPath $SharedLibrary).ProviderPath
$licensePath = (Resolve-Path -LiteralPath $LicenseFile).ProviderPath
$outputPath = [System.IO.Path]::GetFullPath($OutputAar)
$outputDirectory = Split-Path -Parent $outputPath
$manifest = @'
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.ai.edge.litert.x86runtime" />
'@
$fixedTimestamp = [DateTimeOffset]::new(1980, 1, 1, 0, 0, 0, [TimeSpan]::Zero)

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath
}

$output = [System.IO.File]::Open($outputPath, [System.IO.FileMode]::CreateNew)
try {
    $archive = [System.IO.Compression.ZipArchive]::new(
        $output,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false
    )
    try {
        $entries = @(
            @{ Name = "AndroidManifest.xml"; Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($manifest) },
            @{ Name = "LICENSE"; Path = $licensePath },
            @{ Name = "jni/x86/libLiteRt.so"; Path = $sharedLibraryPath }
        )
        foreach ($item in $entries) {
            $entry = $archive.CreateEntry($item.Name, [System.IO.Compression.CompressionLevel]::Optimal)
            $entry.LastWriteTime = $fixedTimestamp
            $entryStream = $entry.Open()
            try {
                if ($item.Bytes) {
                    $entryStream.Write($item.Bytes, 0, $item.Bytes.Length)
                } else {
                    $input = [System.IO.File]::OpenRead($item.Path)
                    try {
                        $input.CopyTo($entryStream)
                    } finally {
                        $input.Dispose()
                    }
                }
            } finally {
                $entryStream.Dispose()
            }
        }
    } finally {
        $archive.Dispose()
    }
} finally {
    $output.Dispose()
}

$aar = Get-Item -LiteralPath $outputPath
$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath
[pscustomobject]@{
    Path = $aar.FullName
    Bytes = $aar.Length
    Sha256 = $hash.Hash.ToLowerInvariant()
}
