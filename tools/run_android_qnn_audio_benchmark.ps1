param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._:-]*$')]
    [string]$Serial,

    [Parameter(Mandatory = $true)]
    [string]$SourceAudio,

    [Parameter(Mandatory = $true)]
    [string]$ContractFile,

    [Parameter(Mandatory = $true)]
    [string]$OnnxModel,

    [Parameter(Mandatory = $true)]
    [string]$LiteRtModel,

    [Parameter(Mandatory = $true)]
    [string]$AppApk,

    [Parameter(Mandatory = $true)]
    [string]$RuntimeArtifact,

    [Parameter(Mandatory = $true)]
    [string]$AcceleratorBundleManifest,

    [string]$Tag = "",
    [ValidateRange(0.1, 60)]
    [double]$SampleIntervalSeconds = 2,
    [switch]$ReuseDeviceFiles
)

$ErrorActionPreference = "Stop"

$adb = Join-Path $env:LOCALAPPDATA "Android/Sdk/platform-tools/adb.exe"
$package = "com.example.musicsourceseparation"
$component = "$package/.benchmark.InferenceBenchmarkService"
$externalRoot = "/sdcard/Android/data/$package/files/benchmark"
$modelRoot = "$externalRoot/models"
$contractRoot = "$externalRoot/contracts"
$audioInputRoot = "$externalRoot/audio-input"
$audioOutputRoot = "$externalRoot/audio-output"
$reportRoot = "$externalRoot/reports"

if (-not (Test-Path -LiteralPath $adb)) {
    throw "adb was not found at $adb"
}
function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label was not found: $Path"
    }
    (Resolve-Path -LiteralPath $Path).Path
}
$SourceAudio = Require-File $SourceAudio "Source audio"
$ContractFile = Require-File $ContractFile "Model contract"
$OnnxModel = Require-File $OnnxModel "ONNX model"
$LiteRtModel = Require-File $LiteRtModel "LiteRT model"
$AppApk = Require-File $AppApk "App APK"
$RuntimeArtifact = Require-File $RuntimeArtifact "LiteRT runtime artifact"
$AcceleratorBundleManifest = Require-File $AcceleratorBundleManifest "QNN runtime manifest"

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$contract = Get-Content -LiteralPath $ContractFile -Raw | ConvertFrom-Json
if ($contract.contractSchemaVersion -ne 2 -or
    $contract.contractId -cne "uvr_mdxnet_3_9662@2" -or
    $contract.modelId -cne "uvr_mdxnet_3_9662") {
    throw "QNN audio requires the uvr_mdxnet_3_9662@2 contract."
}
$contractName = Split-Path -Leaf $ContractFile
$onnxName = Split-Path -Leaf $OnnxModel
$modelName = Split-Path -Leaf $LiteRtModel
$audioName = Split-Path -Leaf $SourceAudio
$contractSha256 = Get-Sha256 $ContractFile
$onnxSha256 = Get-Sha256 $OnnxModel
$modelSha256 = Get-Sha256 $LiteRtModel
$audioSha256 = Get-Sha256 $SourceAudio
$appApkSha256 = Get-Sha256 $AppApk
$runtimeArtifactSha256 = Get-Sha256 $RuntimeArtifact
$acceleratorBundleSha256 = Get-Sha256 $AcceleratorBundleManifest
if ($contract.source.fileName -cne $onnxName -or
    $contract.source.byteSize -ne (Get-Item $OnnxModel).Length -or
    $contract.source.sha256 -cne $onnxSha256) {
    throw "ONNX model identity does not match the contract."
}
if ($contract.artifact.fileName -cne $modelName -or
    $contract.artifact.byteSize -ne (Get-Item $LiteRtModel).Length -or
    $contract.artifact.sha256 -cne $modelSha256) {
    throw "LiteRT model identity does not match the contract."
}
if ([System.IO.Path]::GetExtension($ContractFile) -cne ".json" -or
    [System.IO.Path]::GetExtension($OnnxModel) -cne ".onnx" -or
    [System.IO.Path]::GetExtension($LiteRtModel) -cne ".tflite" -or
    [System.IO.Path]::GetExtension($SourceAudio) -cne ".wav" -or
    [System.IO.Path]::GetExtension($AppApk) -cne ".apk" -or
    [System.IO.Path]::GetExtension($RuntimeArtifact) -cne ".aar" -or
    [System.IO.Path]::GetExtension($AcceleratorBundleManifest) -cne ".json") {
    throw "Unexpected file extension in the frozen QNN audio inputs."
}
if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = "qnn-audio-{0:yyyyMMdd-HHmmss}" -f (Get-Date)
}
if ($Tag -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "Tag must start with an ASCII letter or digit and contain only letters, digits, dot, underscore, or hyphen: $Tag"
}
foreach ($name in @($contractName, $onnxName, $modelName, $audioName)) {
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

function Get-DeviceSha256([string]$Path) {
    $line = (Invoke-Adb shell sha256sum $Path | Select-Object -First 1)
    if ($line -notmatch '^([0-9a-fA-F]{64})\s') {
        throw "Could not parse device SHA-256 for ${Path}: $line"
    }
    return $Matches[1].ToLowerInvariant()
}

function Start-HostActivity {
    Invoke-Adb shell am start -W -n "$package/.MainActivity" | Out-Null
    Start-Sleep -Milliseconds 500
    Invoke-Adb shell input keyevent KEYCODE_HOME
    Start-Sleep -Milliseconds 500
}

function Initialize-DeviceDirectories {
    $initReport = "$reportRoot/init.json"
    Invoke-Adb shell rm -f $initReport "$initReport.partial"
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
if (Test-Path -LiteralPath $sampleDir) {
    throw "Host output directory already exists; choose a unique Tag: $sampleDir"
}

$sourceRevisionOutput = & git -C $PSScriptRoot rev-parse --verify HEAD
if ($LASTEXITCODE -ne 0 -or -not $sourceRevisionOutput) {
    throw "Unable to resolve the benchmark source revision."
}
$sourceRevision = ($sourceRevisionOutput | Select-Object -First 1).Trim()
$sourceStatus = & git -C $PSScriptRoot status --porcelain=v1 --untracked-files=normal
if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine whether the benchmark source tree is dirty."
}
$sourceDirty = if ($sourceStatus) { "true" } else { "false" }

$packagePathLine = Invoke-Adb shell pm path $package | Select-Object -First 1
if ($packagePathLine -notmatch '^package:(/.+\.apk)$') {
    throw "Could not resolve the installed base APK for $package."
}
$installedApkPath = $Matches[1]
$installedApkSha256 = Get-DeviceSha256 $installedApkPath
if ($installedApkSha256 -cne $appApkSha256) {
    throw "Installed APK SHA-256 does not match -AppApk; install the frozen APK before running."
}

Invoke-Adb shell am force-stop $package
Start-HostActivity
Initialize-DeviceDirectories
if ($ReuseDeviceFiles) {
    Invoke-Adb shell test -s "$contractRoot/$contractName"
    Invoke-Adb shell test -s "$modelRoot/$onnxName"
    Invoke-Adb shell test -s "$modelRoot/$modelName"
    Invoke-Adb shell test -s "$audioInputRoot/$audioName"
    $deviceIdentities = [ordered]@{
        "$contractRoot/$contractName" = $contractSha256
        "$modelRoot/$onnxName" = $onnxSha256
        "$modelRoot/$modelName" = $modelSha256
        "$audioInputRoot/$audioName" = $audioSha256
    }
    foreach ($identity in $deviceIdentities.GetEnumerator()) {
        if ((Get-DeviceSha256 $identity.Key) -cne $identity.Value) {
            throw "Reused device file does not match its frozen host input: $($identity.Key)."
        }
    }
} else {
    Invoke-Adb push $ContractFile "$contractRoot/$contractName"
    Invoke-Adb push $OnnxModel "$modelRoot/$onnxName"
    Invoke-Adb push $LiteRtModel "$modelRoot/$modelName"
    Invoke-Adb push $SourceAudio "$audioInputRoot/$audioName"
}
New-Item -ItemType Directory -Path $sampleDir | Out-Null
$samples = Join-Path $sampleDir "device-samples.jsonl"
$hostIdentity = [ordered]@{
    schemaVersion = 1
    tag = $Tag
    backend = "litert_qnn_audio"
    modelId = $contract.modelId
    contract = [ordered]@{ path = $ContractFile; bytes = (Get-Item $ContractFile).Length; sha256 = $contractSha256; id = $contract.contractId }
    onnx = [ordered]@{ path = $OnnxModel; bytes = (Get-Item $OnnxModel).Length; sha256 = $onnxSha256 }
    litert = [ordered]@{ path = $LiteRtModel; bytes = (Get-Item $LiteRtModel).Length; sha256 = $modelSha256 }
    input = [ordered]@{ path = $SourceAudio; bytes = (Get-Item $SourceAudio).Length; sha256 = $audioSha256; format = "canonical-pcm16-wave" }
    source = [ordered]@{ revision = $sourceRevision; dirty = $sourceDirty }
    appApk = [ordered]@{ path = $AppApk; bytes = (Get-Item $AppApk).Length; sha256 = $appApkSha256; installedPath = $installedApkPath }
    runtimeArtifact = [ordered]@{ path = $RuntimeArtifact; bytes = (Get-Item $RuntimeArtifact).Length; sha256 = $runtimeArtifactSha256 }
    acceleratorBundle = [ordered]@{ path = $AcceleratorBundleManifest; bytes = (Get-Item $AcceleratorBundleManifest).Length; sha256 = $acceleratorBundleSha256 }
}
$hostIdentity | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $sampleDir "host-identity.json") -Encoding utf8

Invoke-Adb shell am force-stop $package
Start-HostActivity
Invoke-Adb shell rm -f "$reportRoot/$Tag.json" "$reportRoot/$Tag.json.partial"
Invoke-Adb shell rm -rf "$audioOutputRoot/$Tag" "$audioOutputRoot/$Tag.partial" "$externalRoot/qnn/$Tag"
Invoke-Adb logcat -c

Invoke-Adb shell am start-foreground-service `
    -a "$package.RUN_INFERENCE_BENCHMARK" `
    -n $component `
    --es backend litert_qnn_audio `
    --es tag $Tag `
    --es modelId uvr_mdxnet_3_9662 `
    --es contractFile $contractName `
    --es contractSha256 $contractSha256 `
    --es onnxModel $onnxName `
    --es onnxModelSha256 $onnxSha256 `
    --es litertModel $modelName `
    --es liteRtModelSha256 $modelSha256 `
    --es inputSha256 $audioSha256 `
    --es audioFile $audioName `
    --ef modelOutputScale $([float]$contract.dsp.modelOutputScale) | Out-Null

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

if ($parsedReport.status -cne "complete") {
    throw "QNN audio benchmark failed. Report: $reportPath"
}
if ($parsedReport.schemaVersion -ne 2 -or
    $parsedReport.tag -cne $Tag -or
    $parsedReport.backend -cne "litert_qnn_audio" -or
    $parsedReport.modelId -cne $contract.modelId -or
    $parsedReport.identity.contract.id -cne $contract.contractId -or
    $parsedReport.identity.contract.fileName -cne $contractName -or
    $parsedReport.identity.contract.bytes -ne (Get-Item $ContractFile).Length -or
    $parsedReport.identity.contract.sha256 -cne $contractSha256 -or
    $parsedReport.identity.onnx.fileName -cne $onnxName -or
    $parsedReport.identity.onnx.bytes -ne (Get-Item $OnnxModel).Length -or
    $parsedReport.identity.onnx.sha256 -cne $onnxSha256 -or
    $parsedReport.identity.litert.fileName -cne $modelName -or
    $parsedReport.identity.litert.bytes -ne (Get-Item $LiteRtModel).Length -or
    $parsedReport.identity.litert.sha256 -cne $modelSha256 -or
    $parsedReport.identity.input.fileName -cne $audioName -or
    $parsedReport.identity.input.format -cne "canonical-pcm16-wave" -or
    $parsedReport.identity.input.bytes -ne (Get-Item $SourceAudio).Length -or
    $parsedReport.identity.input.sha256 -cne $audioSha256) {
    throw "Device report identity does not match the frozen QNN audio inputs."
}
if ($parsedReport.app.applicationId -cne $package -or
    $parsedReport.app.sourceRevision -cne $sourceRevision -or
    $parsedReport.app.sourceDirty -cne $sourceDirty -or
    $parsedReport.app.runtimeArtifactSha256 -cne $runtimeArtifactSha256 -or
    $parsedReport.app.acceleratorBundleSha256 -cne $acceleratorBundleSha256) {
    throw "Installed app build identity does not match the frozen source/runtime inputs."
}
$delegationEvidence = $parsedReport.audio.backendEvidence
if ($delegationEvidence.delegationStatus -cne "delegated" -or
    @($delegationEvidence.irFiles).Count -lt 1 -or
    @($delegationEvidence.nativeLibraries).Count -lt 1) {
    throw "QNN audio completed without complete delegation/runtime evidence."
}
if ($parsedReport.audio.contract.contractId -cne $contract.contractId -or
    $parsedReport.audio.model.sha256 -cne $modelSha256 -or
    $parsedReport.audio.source.sha256 -cne $audioSha256) {
    throw "QNN audio pipeline report does not match the frozen contract/model/source."
}

$runtimeLog = & $adb -s $Serial logcat -d -v threadtime 2>&1 |
    Select-String -Pattern "MSS-QNN|LiteRt|LiteRT|QNN|Qnn|HTP|Qualcomm|CompiledModel" |
    ForEach-Object { $_.Line }
$runtimeLog | Set-Content -LiteralPath (Join-Path $sampleDir "runtime-logcat.txt") -Encoding utf8

$audioDir = Join-Path $sampleDir "audio"
New-Item -ItemType Directory -Path $audioDir | Out-Null
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
Invoke-Adb shell test -d $remoteEvidence
$localEvidence = Join-Path $sampleDir "qnn-ir"
New-Item -ItemType Directory -Path $localEvidence | Out-Null
Invoke-Adb pull "$remoteEvidence/." $localEvidence
if (-not (Get-ChildItem -LiteralPath $localEvidence -File -Recurse | Where-Object Length -gt 0)) {
    throw "Pulled QNN IR evidence is empty."
}

$reportText
Write-Host "Saved QNN audio report, samples, IR, and verified WAV files to $sampleDir"
