param(
    [Parameter(Mandatory = $true)]
    [string]$Serial,

    [Parameter(Mandatory = $true)]
    [string]$SourceAudio,

    [Parameter(Mandatory = $true)]
    [string]$LiteRtModel,

    [string]$Tag = "",
    [string]$ModelId = "uvr_mdxnet_3_9662",
    [float]$ModelOutputScale = 1.035,
    [double]$SampleIntervalSeconds = 2,
    [switch]$ReuseDeviceFiles
)

$ErrorActionPreference = "Stop"

$adb = Join-Path $env:LOCALAPPDATA "Android/Sdk/platform-tools/adb.exe"
$package = "com.example.musicsourceseparation"
$component = "$package/.benchmark.InferenceBenchmarkService"
$externalRoot = "/sdcard/Android/data/$package/files/benchmark"
$modelRoot = "$externalRoot/models"
$audioInputRoot = "$externalRoot/audio-input"
$audioOutputRoot = "$externalRoot/audio-output"
$reportRoot = "$externalRoot/reports"

if (-not (Test-Path -LiteralPath $adb)) {
    throw "adb was not found at $adb"
}
foreach ($path in @($SourceAudio, $LiteRtModel)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Input file not found: $path"
    }
}
if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = "qnn-audio-{0:yyyyMMdd-HHmmss}" -f (Get-Date)
}
if ($Tag -notmatch '^[A-Za-z0-9._-]+$') {
    throw "Tag must contain only ASCII letters, digits, dot, underscore, or hyphen: $Tag"
}
if (-not [float]::IsFinite($ModelOutputScale) -or $ModelOutputScale -le 0) {
    throw "ModelOutputScale must be finite and positive."
}

$modelName = Split-Path -Leaf $LiteRtModel
$audioName = Split-Path -Leaf $SourceAudio
foreach ($name in @($modelName, $audioName)) {
    if ($name -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Device file names must contain only ASCII letters, digits, dot, underscore, or hyphen: $name"
    }
}

function Invoke-Adb {
    & $adb -s $Serial @args
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: $args"
    }
}

function Start-HostActivity {
    Invoke-Adb shell am start -W -n "$package/.MainActivity" | Out-Null
    Start-Sleep -Milliseconds 500
    Invoke-Adb shell input keyevent KEYCODE_HOME
    Start-Sleep -Milliseconds 500
}

function Initialize-DeviceDirectories {
    $initReport = "$reportRoot/init.json"
    Invoke-Adb shell am start-foreground-service `
        -a "$package.RUN_INFERENCE_BENCHMARK" `
        -n $component `
        --es backend init `
        --es tag init | Out-Null
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 250
        $text = & $adb -s $Serial shell cat $initReport 2>$null
    } while ((-not $text) -and ((Get-Date) -lt $deadline))
    if (-not $text) {
        throw "Benchmark directory initialization timed out: $initReport"
    }
}

$deviceDirectory = $Serial.Replace(':', '_')
$sampleDir = Join-Path $PSScriptRoot "../outputs/android-benchmark/$deviceDirectory/$Tag"
New-Item -ItemType Directory -Force -Path $sampleDir | Out-Null
$samples = Join-Path $sampleDir "device-samples.jsonl"

Invoke-Adb shell am force-stop $package
Start-HostActivity
Initialize-DeviceDirectories
if ($ReuseDeviceFiles) {
    Invoke-Adb shell test -s "$modelRoot/$modelName"
    Invoke-Adb shell test -s "$audioInputRoot/$audioName"
} else {
    Invoke-Adb push $LiteRtModel "$modelRoot/$modelName"
    Invoke-Adb push $SourceAudio "$audioInputRoot/$audioName"
}

Invoke-Adb shell am force-stop $package
Start-HostActivity
Invoke-Adb shell rm -f "$reportRoot/$Tag.json"
Invoke-Adb shell rm -rf "$audioOutputRoot/$Tag" "$externalRoot/qnn/$Tag"
Invoke-Adb logcat -c

Invoke-Adb shell am start-foreground-service `
    -a "$package.RUN_INFERENCE_BENCHMARK" `
    -n $component `
    --es backend litert_qnn_audio `
    --es tag $Tag `
    --es modelId $ModelId `
    --es litertModel $modelName `
    --es audioFile $audioName `
    --ef modelOutputScale $ModelOutputScale | Out-Null

$remoteReport = "$reportRoot/$Tag.json"
$deadline = (Get-Date).AddMinutes(15)
$reportText = ""
do {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $thermal = (Invoke-Adb shell dumpsys thermalservice |
        Select-String -Pattern "Thermal Status|mValue=.*mName=(AP|BAT|SKIN)" |
        ForEach-Object { $_.Line.Trim() }) -join " | "
    $battery = (Invoke-Adb shell dumpsys battery |
        Select-Object -First 30 |
        Select-String -Pattern "level:|temperature:|status:|voltage:|current now:|charge counter:|AC powered:|USB powered:|Wireless powered:" |
        ForEach-Object { $_.Line.Trim() }) -join " | "
    $memory = (Invoke-Adb shell dumpsys meminfo $package |
        Select-String -Pattern "TOTAL PSS:|Graphics:" |
        ForEach-Object { $_.Line.Trim() }) -join " | "
    [ordered]@{
        timestampMs = $timestamp
        thermal = $thermal
        battery = $battery
        memory = $memory
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $samples -Encoding utf8

    $report = & $adb -s $Serial shell cat $remoteReport 2>$null
    $reportText = $report -join "`n"
    if ($reportText -match '"status":\s*"(complete|error)"') {
        break
    }
    Start-Sleep -Milliseconds ([Math]::Max(100, [int]($SampleIntervalSeconds * 1000)))
} while ((Get-Date) -lt $deadline)

if ($reportText -notmatch '"status":\s*"(complete|error)"') {
    throw "QNN audio benchmark timed out. Samples: $samples"
}
$reportPath = Join-Path $sampleDir "report.json"
$reportText | Set-Content -LiteralPath $reportPath -Encoding utf8
$parsedReport = $reportText | ConvertFrom-Json

$runtimeLog = & $adb -s $Serial logcat -d -v threadtime 2>&1 |
    Select-String -Pattern "MSS-QNN|LiteRt|LiteRT|QNN|Qnn|HTP|Qualcomm|CompiledModel" |
    ForEach-Object { $_.Line }
$runtimeLog | Set-Content -LiteralPath (Join-Path $sampleDir "runtime-logcat.txt") -Encoding utf8

if ($parsedReport.status -eq "complete") {
    $audioDir = Join-Path $sampleDir "audio"
    New-Item -ItemType Directory -Force -Path $audioDir | Out-Null
    foreach ($stem in @("vocals", "instrumental")) {
        $evidence = $parsedReport.audio.outputs.$stem
        $local = Join-Path $audioDir "$stem.wav"
        Invoke-Adb pull $evidence.path $local
        $actualBytes = (Get-Item -LiteralPath $local).Length
        $actualHash = (Get-FileHash -LiteralPath $local -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualBytes -ne $evidence.bytes -or $actualHash -ne $evidence.sha256) {
            throw "Pulled $stem output does not match the device report."
        }
    }

    $remoteEvidence = "$externalRoot/qnn/$Tag"
    & $adb -s $Serial shell test -d $remoteEvidence
    if ($LASTEXITCODE -eq 0) {
        $localEvidence = Join-Path $sampleDir "qnn-ir"
        New-Item -ItemType Directory -Force -Path $localEvidence | Out-Null
        Invoke-Adb pull "$remoteEvidence/." $localEvidence
    }
} else {
    throw "QNN audio benchmark failed. Report: $reportPath"
}

$reportText
Write-Host "Saved QNN audio report, samples, IR, and verified WAV files to $sampleDir"
