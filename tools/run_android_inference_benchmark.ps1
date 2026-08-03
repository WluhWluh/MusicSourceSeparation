param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._:-]*$')]
    [string]$Serial,

    [ValidateSet("ort", "litert_cpu", "litert_gpu", "litert_gpu_fp32", "litert_gpu_bounded", "litert_qnn")]
    [string]$Backend = "ort",

    [ValidateRange(1, 1000)]
    [int]$Iterations = 5,
    [ValidateRange(0, 20)]
    [int]$Warmups = 1,
    [ValidateRange(1, 16)]
    [int]$Threads = 8,
    [ValidateRange(0.1, 60)]
    [double]$SampleIntervalSeconds = 2,
    [long]$Seed = 9482,
    [string]$Tag = "",
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9_]+$')]
    [string]$ModelId,
    [Parameter(Mandatory = $true)]
    [string]$ContractFile,
    [switch]$QnnProfiling,
    [switch]$KeepActivityForeground,
    [ValidateRange(0, 100)]
    [int]$SwipeCount = 0,
    [ValidateRange(0, 120)]
    [double]$UiStartDelaySeconds = 10,
    [switch]$ExportOutputTensor,
    [switch]$UploadModels,
    [Parameter(Mandatory = $true)]
    [string]$InputFile,
    [Parameter(Mandatory = $true)]
    [string]$AppApk,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeArtifact,
    [string]$AcceleratorBundleManifest = "",
    [Parameter(Mandatory = $true)]
    [string]$OnnxModel,
    [Parameter(Mandatory = $true)]
    [string]$LiteRtModel
)

$ErrorActionPreference = "Stop"

$adb = Join-Path $env:LOCALAPPDATA "Android/Sdk/platform-tools/adb.exe"
$package = "com.example.musicsourceseparation"
$component = "$package/.benchmark.InferenceBenchmarkService"
$externalRoot = "/sdcard/Android/data/$package/files/benchmark"
$modelRoot = "$externalRoot/models"
$contractRoot = "$externalRoot/contracts"
$inputRoot = "$externalRoot/inputs"
$reportRoot = "$externalRoot/reports"

if (-not (Test-Path -LiteralPath $adb)) {
    throw "adb was not found at $adb"
}

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label was not found: $Path"
    }
    (Resolve-Path -LiteralPath $Path).Path
}

$ContractFile = Require-File $ContractFile "Model contract"
$OnnxModel = Require-File $OnnxModel "ONNX model"
$LiteRtModel = Require-File $LiteRtModel "LiteRT model"
$InputFile = Require-File $InputFile "Input tensor"
$AppApk = Require-File $AppApk "App APK"
$RuntimeArtifact = Require-File $RuntimeArtifact "Runtime artifact"
if ($Backend -eq "litert_qnn") {
    if ([string]::IsNullOrWhiteSpace($AcceleratorBundleManifest)) {
        throw "-AcceleratorBundleManifest is required for litert_qnn."
    }
    $AcceleratorBundleManifest = Require-File $AcceleratorBundleManifest "Accelerator bundle manifest"
} elseif (-not [string]::IsNullOrWhiteSpace($AcceleratorBundleManifest)) {
    $AcceleratorBundleManifest = Require-File $AcceleratorBundleManifest "Accelerator bundle manifest"
}

try {
    $contract = Get-Content -LiteralPath $ContractFile -Raw | ConvertFrom-Json
} catch {
    throw "Model contract is not valid JSON: $ContractFile. $($_.Exception.Message)"
}
if ($contract.contractSchemaVersion -ne 2) {
    throw "Expected model contract schema 2, got $($contract.contractSchemaVersion)."
}
if ($contract.modelId -cne $ModelId) {
    throw "Contract model ID $($contract.modelId) does not match requested $ModelId."
}
if ($contract.contractId -cne "$ModelId@2") {
    throw "Contract ID must be $ModelId@2, got $($contract.contractId)."
}
$contractName = Split-Path -Leaf $ContractFile
$contractSha256 = Get-Sha256 $ContractFile
$onnxSha256 = Get-Sha256 $OnnxModel
$liteRtSha256 = Get-Sha256 $LiteRtModel
$inputSha256 = Get-Sha256 $InputFile
$appApkSha256 = Get-Sha256 $AppApk
$runtimeArtifactSha256 = Get-Sha256 $RuntimeArtifact
$acceleratorBundleSha256 = if ($AcceleratorBundleManifest) {
    Get-Sha256 $AcceleratorBundleManifest
} else {
    "none"
}
if ($contract.source.fileName -cne (Split-Path -Leaf $OnnxModel) -or
    $contract.source.byteSize -ne (Get-Item -LiteralPath $OnnxModel).Length -or
    $contract.source.sha256 -cne $onnxSha256) {
    throw "ONNX model identity does not match contract $($contract.contractId)."
}
if ($contract.artifact.fileName -cne (Split-Path -Leaf $LiteRtModel) -or
    $contract.artifact.byteSize -ne (Get-Item -LiteralPath $LiteRtModel).Length -or
    $contract.artifact.sha256 -cne $liteRtSha256) {
    throw "LiteRT model identity does not match contract $($contract.contractId)."
}
$inputShape = @($contract.tensorContract.input.shape)
if ($contract.tensorContract.batchSize -ne 1 -or
    $contract.tensorContract.complexChannelCount -ne 4 -or
    $contract.tensorContract.input.name -notmatch '\S' -or
    $contract.tensorContract.output.name -notmatch '\S' -or
    $contract.tensorContract.input.dtype -cne "float32" -or
    $contract.tensorContract.output.dtype -cne "float32" -or
    $contract.tensorContract.input.layout -cne "NHWC" -or
    $contract.tensorContract.output.layout -cne "NHWC" -or
    $inputShape.Count -ne 4 -or $inputShape[0] -ne 1 -or $inputShape[3] -ne 4) {
    throw "Contract must define named static float32 NHWC [1,H,W,4] input and output tensors."
}
$outputShape = @($contract.tensorContract.output.shape)
if (($outputShape -join ",") -cne ($inputShape -join ",")) {
    throw "Contract output shape must match its input shape."
}
$Height = [int]$inputShape[1]
$Width = [int]$inputShape[2]
if ($Height -ne $contract.dsp.dimF -or
    $Width -ne $contract.dsp.modelTimeFrames -or
    $Width -ne [math]::Pow(2, [int]$contract.dsp.dimTPower)) {
    throw "Contract tensor shape does not match its DSP dimensions."
}
$expectedInputBytes = [long]$inputShape[0] * [long]$inputShape[1] *
    [long]$inputShape[2] * [long]$inputShape[3] * 4L
$actualInputBytes = (Get-Item -LiteralPath $InputFile).Length
if ($actualInputBytes -ne $expectedInputBytes) {
    throw "Input tensor must contain exactly $expectedInputBytes little-endian float32 bytes; got $actualInputBytes."
}

$onnxModelName = Split-Path -Leaf $OnnxModel
$liteRtModelName = Split-Path -Leaf $LiteRtModel
foreach ($name in @($onnxModelName, $liteRtModelName, $contractName)) {
    if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
        throw "Model and contract file names must start with an ASCII letter or digit and contain only letters, digits, dot, underscore, or hyphen: $name"
    }
}
if ([System.IO.Path]::GetExtension($ContractFile) -cne ".json" -or
    [System.IO.Path]::GetExtension($OnnxModel) -cne ".onnx" -or
    [System.IO.Path]::GetExtension($LiteRtModel) -cne ".tflite" -or
    [System.IO.Path]::GetExtension($InputFile) -cne ".bin" -or
    [System.IO.Path]::GetExtension($AppApk) -cne ".apk" -or
    [System.IO.Path]::GetExtension($RuntimeArtifact) -cne ".aar") {
    throw "Expected .json contract, .onnx source, .tflite model, .bin input, .apk app, and .aar runtime files."
}
if ($AcceleratorBundleManifest -and
    [System.IO.Path]::GetExtension($AcceleratorBundleManifest) -cne ".json") {
    throw "Accelerator bundle manifest must be a .json file."
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
    if ($KeepActivityForeground) {
        Invoke-Adb shell input keyevent KEYCODE_WAKEUP
        Invoke-Adb shell wm dismiss-keyguard
    }
    $activityArgs = @("shell", "am", "start", "-W", "-n", "$package/.MainActivity")
    if ($KeepActivityForeground) {
        $activityArgs += @("--ez", "benchmarkKeepScreenOn", "true")
    }
    Invoke-Adb @activityArgs
    Start-Sleep -Milliseconds 500
    if (-not $KeepActivityForeground) {
        Invoke-Adb shell input keyevent KEYCODE_HOME
        Start-Sleep -Milliseconds 500
    }
}

if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = "{0}-{1:yyyyMMdd-HHmmss}" -f $Backend, (Get-Date)
}
if ($Tag -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "Tag must start with an ASCII letter or digit and contain only letters, digits, dot, underscore, or hyphen: $Tag"
}
if ($SwipeCount -gt 0 -and -not $KeepActivityForeground) {
    throw "-SwipeCount requires -KeepActivityForeground."
}

$inputName = (Split-Path -Leaf $InputFile).ToLowerInvariant()
if ($inputName -notmatch '^[a-z0-9][a-z0-9._-]*$') {
    throw "Input file name must start with an ASCII letter or digit and contain only letters, digits, dot, underscore, or hyphen: $inputName"
}
$remoteInput = "$inputRoot/$inputName"

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
$remoteReport = "$reportRoot/$Tag.json"
$sampleDir = Join-Path $PSScriptRoot "../outputs/android-benchmark/$($Serial.Replace(':', '_'))/$Tag"
if (Test-Path -LiteralPath $sampleDir) {
    throw "Benchmark tag is write-once and already exists: $sampleDir"
}
$packagePathLine = Invoke-Adb shell pm path $package | Select-Object -First 1
if ($packagePathLine -notmatch '^package:(/.+\.apk)$') {
    throw "Could not resolve the installed base APK for $package."
}
$installedApkPath = $Matches[1]
$installedApkLine = Invoke-Adb shell sha256sum $installedApkPath | Select-Object -First 1
if ($installedApkLine -notmatch '^([0-9a-fA-F]{64})\s') {
    throw "Could not read the installed base APK SHA-256."
}
$installedApkSha256 = $Matches[1].ToLowerInvariant()
if ($installedApkSha256 -cne $appApkSha256) {
    throw "Installed APK SHA-256 does not match -AppApk; install the frozen APK before running."
}
New-Item -ItemType Directory -Path $sampleDir | Out-Null
$samples = Join-Path $sampleDir "device-samples.jsonl"

$hostIdentity = [ordered]@{
    schemaVersion = 1
    tag = $Tag
    backend = $Backend
    modelId = $ModelId
    contract = [ordered]@{ path = $ContractFile; bytes = (Get-Item $ContractFile).Length; sha256 = $contractSha256; id = $contract.contractId }
    onnx = [ordered]@{ path = $OnnxModel; bytes = (Get-Item $OnnxModel).Length; sha256 = $onnxSha256 }
    litert = [ordered]@{ path = $LiteRtModel; bytes = (Get-Item $LiteRtModel).Length; sha256 = $liteRtSha256 }
    input = [ordered]@{ path = $InputFile; bytes = $actualInputBytes; sha256 = $inputSha256; layout = "NCHW" }
    source = [ordered]@{ revision = $sourceRevision; dirty = $sourceDirty }
    execution = [ordered]@{
        keepActivityForeground = $KeepActivityForeground.IsPresent
        keepScreenOn = $KeepActivityForeground.IsPresent
    }
    appApk = [ordered]@{ path = $AppApk; bytes = (Get-Item $AppApk).Length; sha256 = $appApkSha256; installedPath = $installedApkPath }
    runtimeArtifact = [ordered]@{ path = $RuntimeArtifact; bytes = (Get-Item $RuntimeArtifact).Length; sha256 = $runtimeArtifactSha256 }
    acceleratorBundle = if ($AcceleratorBundleManifest) { [ordered]@{ path = $AcceleratorBundleManifest; bytes = (Get-Item $AcceleratorBundleManifest).Length; sha256 = $acceleratorBundleSha256 } } else { $null }
}
$hostIdentity | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $sampleDir "host-identity.json") -Encoding utf8

if ($UploadModels) {
    Invoke-Adb shell am force-stop $package
    Invoke-Adb shell rm -rf $externalRoot
    if ($KeepActivityForeground) {
        Invoke-Adb shell input keyevent KEYCODE_WAKEUP
        Invoke-Adb shell wm dismiss-keyguard
    }
    $activityArgs = @("shell", "am", "start", "-W", "-n", "$package/.MainActivity")
    if ($KeepActivityForeground) {
        $activityArgs += @("--ez", "benchmarkKeepScreenOn", "true")
    }
    Invoke-Adb @activityArgs
    Start-Sleep -Milliseconds 500
    if (-not $KeepActivityForeground) {
        Invoke-Adb shell input keyevent KEYCODE_HOME
    }
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
    Invoke-Adb push $ContractFile "$contractRoot/$contractName"
    Invoke-Adb push $InputFile $remoteInput
}

Reset-BenchmarkHost
Invoke-Adb shell rm -f $remoteReport
if ($Backend -eq "litert_qnn") {
    Invoke-Adb logcat -c
}
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
    "--es", "contractFile", $contractName,
    "--es", "contractSha256", $contractSha256,
    "--es", "onnxModelSha256", $onnxSha256,
    "--es", "liteRtModelSha256", $liteRtSha256,
    "--es", "inputSha256", $inputSha256,
    "--ei", "height", $Height,
    "--ei", "width", $Width,
    "--ez", "qnnProfiling", $QnnProfiling.IsPresent.ToString().ToLowerInvariant(),
    "--ez", "exportOutputTensor", $ExportOutputTensor.IsPresent.ToString().ToLowerInvariant(),
    "--es", "onnxModel", $onnxModelName,
    "--es", "litertModel", $liteRtModelName
)
$serviceArgs += @("--es", "inputFile", $inputName)
Invoke-Adb shell am start-foreground-service @serviceArgs

if ($SwipeCount -gt 0) {
    Start-Sleep -Milliseconds ([int]($UiStartDelaySeconds * 1000))
    $sizeLine = (Invoke-Adb shell wm size | Select-String -Pattern "Physical size:" | Select-Object -First 1).Line
    if ($sizeLine -notmatch '(\d+)x(\d+)') {
        throw "Unable to parse display size: $sizeLine"
    }
    $displayWidth = [int]$Matches[1]
    $displayHeight = [int]$Matches[2]
    $x = [int]($displayWidth * 0.5)
    $top = [int]($displayHeight * 0.25)
    $bottom = [int]($displayHeight * 0.8)

    for ($index = 0; $index -lt 4; $index++) {
        if (($index % 2) -eq 0) {
            Invoke-Adb shell input swipe $x $bottom $x $top 500
        } else {
            Invoke-Adb shell input swipe $x $top $x $bottom 500
        }
    }
    Invoke-Adb shell dumpsys gfxinfo $package reset | Out-Null
    $uiStartedMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    for ($index = 0; $index -lt $SwipeCount; $index++) {
        if (($index % 2) -eq 0) {
            Invoke-Adb shell input swipe $x $bottom $x $top 500
        } else {
            Invoke-Adb shell input swipe $x $top $x $bottom 500
        }
    }
    $uiEndedMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    & $adb -s $Serial shell dumpsys gfxinfo $package framestats |
        Set-Content -LiteralPath (Join-Path $sampleDir "gfxinfo-framestats.txt") -Encoding utf8
    [ordered]@{
        startedMs = $uiStartedMs
        endedMs = $uiEndedMs
        durationMs = $uiEndedMs - $uiStartedMs
        swipeCount = $SwipeCount
        swipeDurationMs = 500
        displayWidth = $displayWidth
        displayHeight = $displayHeight
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $sampleDir "ui-sweep.json") -Encoding utf8
}

$deadline = (Get-Date).AddMinutes(30)
$reportFinished = $false
do {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $thermal = (Invoke-Adb shell dumpsys thermalservice | Select-String -Pattern "Thermal Status|mValue=.*mName=(AP|BAT|SKIN)" | ForEach-Object { $_.Line.Trim() }) -join " | "
    $battery = (Invoke-Adb shell dumpsys battery | Select-Object -First 30 | Select-String -Pattern "level:|temperature:|status:|voltage:|current now:|charge counter:|AC powered:|USB powered:|Wireless powered:" | ForEach-Object { $_.Line.Trim() }) -join " | "
    $memory = (Invoke-Adb shell dumpsys meminfo $package | Select-String -Pattern "TOTAL PSS:|Graphics:" | ForEach-Object { $_.Line.Trim() }) -join " | "
    [ordered]@{
        timestampMs = $timestamp
        thermal = $thermal
        battery = $battery
        memory = $memory
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $samples -Encoding utf8

    $report = & $adb -s $Serial shell cat $remoteReport 2>$null
    if ($report -and (($report -join "`n") -match '"status":\s*"(complete|error)"')) {
        $report -join "`n" | Set-Content -LiteralPath (Join-Path $sampleDir "report.json") -Encoding utf8
        $report -join "`n"
        $reportFinished = $true
        break
    }
    Start-Sleep -Milliseconds ([Math]::Max(100, [int]($SampleIntervalSeconds * 1000)))
} while ((Get-Date) -lt $deadline)

if (-not $reportFinished) {
    throw "Benchmark timed out. Samples: $samples"
}

$parsedReport = ($report -join "`n") | ConvertFrom-Json
if ($parsedReport.status -cne "complete") {
    throw "Benchmark failed: $($parsedReport.message)"
}
if ($parsedReport.schemaVersion -ne 2 -or
    $parsedReport.tag -cne $Tag -or
    $parsedReport.backend -cne $Backend -or
    $parsedReport.modelId -cne $ModelId -or
    $parsedReport.iterations -ne $Iterations -or
    $parsedReport.warmups -ne $Warmups -or
    $parsedReport.threads -ne $Threads -or
    $parsedReport.seed -ne $Seed -or
    $parsedReport.identity.contract.id -cne $contract.contractId -or
    $parsedReport.identity.contract.schemaVersion -ne $contract.contractSchemaVersion -or
    $parsedReport.identity.contract.fileName -cne $contractName -or
    $parsedReport.identity.contract.bytes -ne (Get-Item -LiteralPath $ContractFile).Length -or
    $parsedReport.identity.contract.sha256 -cne $contractSha256 -or
    $parsedReport.identity.onnx.fileName -cne $onnxModelName -or
    $parsedReport.identity.onnx.bytes -ne (Get-Item -LiteralPath $OnnxModel).Length -or
    $parsedReport.identity.onnx.sha256 -cne $onnxSha256 -or
    $parsedReport.identity.litert.fileName -cne $liteRtModelName -or
    $parsedReport.identity.litert.bytes -ne (Get-Item -LiteralPath $LiteRtModel).Length -or
    $parsedReport.identity.litert.sha256 -cne $liteRtSha256 -or
    $parsedReport.identity.input.fileName -cne $inputName -or
    $parsedReport.identity.input.format -cne "nchw-float32-le" -or
    $parsedReport.identity.input.bytes -ne $actualInputBytes -or
    $parsedReport.identity.input.sha256 -cne $inputSha256) {
    throw "Device report identity does not match the frozen host inputs."
}
if ($parsedReport.app.applicationId -cne $package -or
    $parsedReport.app.sourceRevision -cne $sourceRevision -or
    $parsedReport.app.sourceDirty -cne $sourceDirty -or
    $parsedReport.app.runtimeArtifactSha256 -cne $runtimeArtifactSha256) {
    throw "Installed app build identity does not match the frozen source and LiteRT runtime artifact."
}
if ((@($parsedReport.inputShapeNchw) -join ",") -cne "1,4,$Height,$Width") {
    throw "Device report input shape does not match the contract-derived NCHW shape."
}
if ($Backend -eq "litert_qnn" -and $parsedReport.app.acceleratorBundleSha256 -cne $acceleratorBundleSha256) {
    throw "QNN app accelerator bundle identity does not match the frozen runtime manifest."
}
if ($Backend -eq "litert_qnn" -and $parsedReport.backendEvidence.delegationStatus -cne "delegated") {
    throw "QNN completed without verified delegation: $($parsedReport.backendEvidence.delegationStatus)"
}
if ($Backend -eq "litert_gpu_bounded" -and $parsedReport.backendEvidence.dispatchCount -lt 1) {
    throw "Bounded GPU completed without a recorded accelerator dispatch."
}

if ($ExportOutputTensor) {
    if ($parsedReport.status -ne "complete" -or -not $parsedReport.outputTensor.path) {
        throw "Benchmark did not publish an output tensor: $($report -join "`n")"
    }
    $localTensor = Join-Path $sampleDir "output-nchw-f32.bin"
    Invoke-Adb pull $parsedReport.outputTensor.path $localTensor
    $actualHash = (Get-FileHash -LiteralPath $localTensor -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -cne $parsedReport.outputTensor.sha256) {
        throw "Pulled output tensor SHA-256 mismatch: $actualHash"
    }
}

if ($Backend -eq "litert_qnn") {
    $logcatPath = Join-Path $sampleDir "runtime-logcat.txt"
    $runtimeLog = & $adb -s $Serial logcat -d -v threadtime 2>&1 |
        Select-String -Pattern "MSS-QNN|LiteRt|LiteRT|QNN|Qnn|HTP|Qualcomm|CompiledModel" |
        ForEach-Object { $_.Line }
    $runtimeLog | Set-Content -LiteralPath $logcatPath -Encoding utf8

    $remoteEvidence = "$externalRoot/qnn/$Tag"
    & $adb -s $Serial shell test -d $remoteEvidence
    if ($LASTEXITCODE -eq 0) {
        $localEvidence = Join-Path $sampleDir "qnn-ir"
        New-Item -ItemType Directory -Force -Path $localEvidence | Out-Null
        Invoke-Adb pull "$remoteEvidence/." $localEvidence
    }
}

Write-Host "Saved report and device samples to $sampleDir"
