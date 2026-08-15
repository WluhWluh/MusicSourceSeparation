param(
    [Parameter(Mandatory = $true)]
    [string]$Device,
    [string]$BssTfliteRoot = "C:\Users\User\Documents\BSSModels\bss-tflite",
    [string]$MatrixFile = "$PSScriptRoot\..\data\mdx-managed-buffer-shape-matrix-v1.json",
    [string]$AppApk = "$PSScriptRoot\..\app\build\outputs\apk\standard\debug\app-standard-debug.apk",
    [string]$TestApk = "$PSScriptRoot\..\app\build\outputs\apk\androidTest\standard\debug\app-standard-debug-androidTest.apk",
    [string]$RunId = "all13-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())",
    [int]$Warmups = 1,
    [int]$Runs = 2,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$runtimeSha = "88cd2f7eaf1443d1c570085b1c24f239db87eb24c788a590adf5158e17443d0e"
$packageRoot = "/sdcard/Android/data/com.example.musicsourceseparation/files/benchmark"
$matrix = Get-Content -LiteralPath $MatrixFile -Raw | ConvertFrom-Json
if ($matrix.schemaVersion -ne 1 -or $matrix.models.Count -ne 13) {
    throw "Expected the frozen 13-shape matrix schema."
}
if ($Warmups -lt 1 -or $Warmups -gt 3 -or $Runs -lt 2 -or $Runs -gt 5) {
    throw "Warmups must be 1..3 and Runs must be 2..5."
}
if ($RunId -notmatch '^[A-Za-z0-9._-]{1,80}$') {
    throw "Invalid run ID."
}

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & adb -s $Device @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed: $($Arguments -join ' ')"
    }
}

function Push-VerifiedFile {
    param(
        [string]$LocalPath,
        [string]$RemotePath,
        [string]$ExpectedSha
    )
    if (-not (Test-Path -LiteralPath $LocalPath -PathType Leaf)) {
        throw "Missing file: $LocalPath"
    }
    $actualSha = (Get-FileHash -LiteralPath $LocalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha -ne $ExpectedSha) {
        throw "Local SHA mismatch for ${LocalPath}: $actualSha"
    }
    Invoke-Adb push $LocalPath $RemotePath
    $remoteLine = (& adb -s $Device shell sha256sum $RemotePath).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to hash $RemotePath on device."
    }
    $remoteSha = ($remoteLine -split '\s+')[0].ToLowerInvariant()
    if ($remoteSha -ne $ExpectedSha) {
        throw "Remote SHA mismatch for ${RemotePath}: $remoteSha"
    }
}

$deviceState = (& adb -s $Device get-state).Trim()
if ($LASTEXITCODE -ne 0 -or $deviceState -ne "device") {
    throw "Device is not ready: $Device"
}

if (-not $SkipInstall) {
    Invoke-Adb install -r $AppApk
    Invoke-Adb install -r $TestApk
}

Invoke-Adb shell mkdir -p "$packageRoot/models" "$packageRoot/contracts" "$packageRoot/matrices"
foreach ($entry in $matrix.models) {
    $contractPath = Join-Path $BssTfliteRoot "contracts\v2\$($entry.contractFile)"
    $modelPath = Join-Path $BssTfliteRoot "artifacts\all-candidates-fp32\$($entry.modelFile)"
    $contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
    if ($contract.modelId -ne $entry.modelId -or
        $contract.artifact.fileName -ne $entry.modelFile -or
        $contract.artifact.sha256 -ne $entry.sha256) {
        throw "Contract identity mismatch for $($entry.modelId)."
    }
    Push-VerifiedFile $modelPath "$packageRoot/models/$($entry.modelFile)" $entry.sha256
    $contractSha = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Push-VerifiedFile $contractPath "$packageRoot/contracts/$($entry.contractFile)" $contractSha
}

$matrixSha = (Get-FileHash -LiteralPath $MatrixFile -Algorithm SHA256).Hash.ToLowerInvariant()
Push-VerifiedFile $MatrixFile "$packageRoot/matrices/$([IO.Path]::GetFileName($MatrixFile))" $matrixSha

$testClass = "com.example.musicsourceseparation.benchmark.MdxManagedBufferShapeMatrixInstrumentedTest#runAllShapes"
$runner = "com.example.musicsourceseparation.test/androidx.test.runner.AndroidJUnitRunner"
Invoke-Adb shell am instrument -w -r `
    -e class $testClass `
    -e matrixFile ([IO.Path]::GetFileName($MatrixFile)) `
    -e warmups $Warmups `
    -e runs $Runs `
    -e runId $RunId `
    $runner

$resultRoot = Join-Path $PSScriptRoot "..\.tmp\mdx-managed-buffer-shape-matrix"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$localReport = Join-Path $resultRoot "$($Device.Replace(':', '-'))-$RunId.json"
Invoke-Adb pull "$packageRoot/mdx-managed-buffer-shape-matrix/$RunId/report.json" $localReport
$report = Get-Content -LiteralPath $localReport -Raw | ConvertFrom-Json
if ($report.status -ne "complete" -or $report.qualification -ne "qualified" -or
    $report.completedShapes -ne 13 -or $report.runtimeArtifactSha256 -ne $runtimeSha) {
    throw "Shape matrix rejected. Report: $localReport"
}

[pscustomobject]@{
    Device = $Device
    RunId = $RunId
    CompletedShapes = $report.completedShapes
    MeasuredWindows = $report.completedMeasuredWindows
    ThermalStart = $report.thermalStatusStart
    ThermalEnd = $report.thermalStatusEnd
    PssGrowthKb = $report.afterClosePssGrowthKb
    NativeHeapGrowthBytes = $report.afterCloseNativeHeapGrowthBytes
    Report = (Resolve-Path -LiteralPath $localReport).Path
}
