param(
    [Parameter(Mandatory = $true)]
    [string]$Serial,

    [ValidateSet("ort", "litert_cpu", "litert_gpu", "litert_gpu_fp32")]
    [string]$Backend = "ort",

    [int]$Iterations = 5,
    [int]$Warmups = 1,
    [int]$Threads = 8,
    [double]$SampleIntervalSeconds = 2,
    [long]$Seed = 9482,
    [string]$Tag = "",
    [string]$ModelId = "uvr_mdxnet_9482",
    [int]$Height = 2048,
    [int]$Width = 256,
    [switch]$UploadModels,
    [string]$InputFile = "",
    [string]$OnnxModel = "models/uvr-mdx/UVR_MDXNET_9482.onnx",
    [string]$LiteRtModel = "$env:TEMP/uvr-mdx-9482-litert-onnxstatic/UVR_MDXNET_9482_static_float32.tflite"
)

$ErrorActionPreference = "Stop"

$adb = Join-Path $env:LOCALAPPDATA "Android/Sdk/platform-tools/adb.exe"
$package = "com.example.musicsourceseparation"
$component = "$package/.benchmark.InferenceBenchmarkService"
$externalRoot = "/sdcard/Android/data/$package/files/benchmark"
$modelRoot = "$externalRoot/models"
$inputRoot = "$externalRoot/inputs"
$reportRoot = "$externalRoot/reports"

if (-not (Test-Path -LiteralPath $adb)) {
    throw "adb was not found at $adb"
}

$onnxModelName = Split-Path -Leaf $OnnxModel
$liteRtModelName = Split-Path -Leaf $LiteRtModel
foreach ($name in @($onnxModelName, $liteRtModelName)) {
    if ($name -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Model file names must contain only ASCII letters, digits, dot, underscore, or hyphen: $name"
    }
}

function Invoke-Adb {
    & $adb -s $Serial @args
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: $args"
    }
}

function Reset-BenchmarkHost {
    Invoke-Adb shell am force-stop $package

    # Samsung's Android 15 build rejects a shell-started FGS while force-stop has
    # left the package in the stopped state. Launching the exported activity once
    # clears that state while still giving every benchmark a fresh app process.
    Invoke-Adb shell am start -W -n "$package/.MainActivity"
    Start-Sleep -Milliseconds 500
    Invoke-Adb shell input keyevent KEYCODE_HOME
    Start-Sleep -Milliseconds 500
}

if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = "{0}-{1:yyyyMMdd-HHmmss}" -f $Backend, (Get-Date)
}

$inputName = ""
$remoteInput = ""
if (-not [string]::IsNullOrWhiteSpace($InputFile)) {
    if (-not (Test-Path -LiteralPath $InputFile)) {
        throw "Input file not found: $InputFile"
    }
    $inputName = (Split-Path -Leaf $InputFile).ToLowerInvariant()
    if ($inputName -notmatch '^[a-z0-9._-]+$') {
        throw "Input file name must contain only ASCII letters, digits, dot, underscore, or hyphen: $inputName"
    }
    $remoteInput = "$inputRoot/$inputName"
}

if ($UploadModels) {
    if (-not (Test-Path -LiteralPath $OnnxModel)) {
        throw "ONNX model not found: $OnnxModel"
    }
    if (-not (Test-Path -LiteralPath $LiteRtModel)) {
        throw "LiteRT model not found: $LiteRtModel"
    }
    Invoke-Adb shell am force-stop $package
    Invoke-Adb shell rm -rf $externalRoot
    Invoke-Adb shell am start -W -n "$package/.MainActivity"
    Start-Sleep -Milliseconds 500
    Invoke-Adb shell input keyevent KEYCODE_HOME
    Invoke-Adb shell am start-foreground-service `
        -a com.example.musicsourceseparation.RUN_INFERENCE_BENCHMARK `
        -n $component `
        --es backend init `
        --es tag init
    $initReport = "$reportRoot/init.json"
    $initDeadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 250
        & $adb -s $Serial shell test -f $initReport
    } while (($LASTEXITCODE -ne 0) -and ((Get-Date) -lt $initDeadline))
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark directory initialization timed out: $initReport"
    }
    Invoke-Adb push $OnnxModel "$modelRoot/$onnxModelName"
    Invoke-Adb push $LiteRtModel "$modelRoot/$liteRtModelName"
    if ($remoteInput) {
        Invoke-Adb push $InputFile $remoteInput
    }
}

$latest = "$reportRoot/latest.json"
$remoteReport = "$reportRoot/$Tag.json"

$sampleDir = Join-Path $PSScriptRoot "../outputs/android-benchmark/$($Serial.Replace(':', '_'))/$Tag"
New-Item -ItemType Directory -Force -Path $sampleDir | Out-Null
$samples = Join-Path $sampleDir "device-samples.jsonl"

Reset-BenchmarkHost
Invoke-Adb shell rm -f $remoteReport
$serviceArgs = @(
    "-a", "com.example.musicsourceseparation.RUN_INFERENCE_BENCHMARK",
    "-n", $component,
    "--es", "backend", $Backend,
    "--ei", "iterations", $Iterations,
    "--ei", "warmups", $Warmups,
    "--ei", "threads", $Threads,
    "--el", "seed", $Seed,
    "--es", "tag", $Tag,
    "--es", "modelId", $ModelId,
    "--ei", "height", $Height,
    "--ei", "width", $Width,
    "--es", "onnxModel", $onnxModelName,
    "--es", "litertModel", $liteRtModelName
)
if ($inputName) {
    $serviceArgs += @("--es", "inputFile", $inputName)
}
Invoke-Adb shell am start-foreground-service @serviceArgs

$deadline = (Get-Date).AddMinutes(30)
do {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $thermal = (Invoke-Adb shell dumpsys thermalservice | Select-String -Pattern "Thermal Status|mValue=.*mName=(AP|BAT|SKIN)" | ForEach-Object { $_.Line.Trim() }) -join " | "
    $battery = (Invoke-Adb shell dumpsys battery | Select-Object -First 30 | Select-String -Pattern "level:|temperature:|status:|voltage:|current now:|charge counter:|AC powered:|USB powered:|Wireless powered:" | ForEach-Object { $_.Line.Trim() }) -join " | "
    [ordered]@{
        timestampMs = $timestamp
        thermal = $thermal
        battery = $battery
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $samples -Encoding utf8

    $report = & $adb -s $Serial shell cat $remoteReport 2>$null
    if ($report -and (($report -join "`n") -match '"status":\s*"(complete|error)"')) {
        $report -join "`n" | Set-Content -LiteralPath (Join-Path $sampleDir "report.json") -Encoding utf8
        $report -join "`n"
        break
    }
    Start-Sleep -Milliseconds ([Math]::Max(100, [int]($SampleIntervalSeconds * 1000)))
} while ((Get-Date) -lt $deadline)

if ((Get-Date) -ge $deadline) {
    throw "Benchmark timed out. Samples: $samples"
}

Write-Host "Saved report and device samples to $sampleDir"
