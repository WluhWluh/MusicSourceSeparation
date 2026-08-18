param(
    [string]$Serial = "192.168.8.197:35131",
    [ValidateSet("cpu", "gpu-bounded")]
    [string]$Backend = "cpu",
    [ValidateRange(1, 16)]
    [int]$Threads = 4,
    [ValidateRange(5, 120)]
    [double]$PlaybackSeconds = 30,
    [ValidateRange(0, 119)]
    [double]$SeekAtSeconds = 12,
    [ValidateRange(0, 600)]
    [double]$SeekToSeconds = 45,
    [ValidateRange(256, 4096)]
    [int]$BlockSamples = 1024,
    [ValidateRange(1, 30)]
    [int]$TimeoutMinutes = 15,
    [string]$RunId = "",
    [string]$RuntimeAar = "",
    [string]$ModelFile = "",
    [string]$SourceAudio = "",
    [string]$AppApk = "",
    [string]$TestApk = "",
    [switch]$SkipBuild,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$package = "com.example.musicsourceseparation"
$runner = "$package.test/androidx.test.runner.AndroidJUnitRunner"
$testClass = "$package.benchmark.TfcTdfStreamingFullChainInstrumentedTest#runFullChainBenchmark"
$remoteRoot = "/sdcard/Android/data/$package/files/benchmark/tfc-tdf-streaming"

if ([string]::IsNullOrWhiteSpace($RuntimeAar)) {
    $RuntimeAar = "C:\Users\User\Documents\BSSModels\bss-litert-android\.cache\bounded-gpu-v2.2.0-bss.2-release\litert-android-2.2.0-bss.2.aar"
}
if ([string]::IsNullOrWhiteSpace($ModelFile)) {
    $ModelFile = Join-Path $repoRoot "models\tfc-tdf\default-compact\tfc_tdf_default_vocals_core_fp32.tflite"
}
if ([string]::IsNullOrWhiteSpace($SourceAudio)) {
    $SourceAudio = Join-Path $repoRoot "data\samples\coast_town.mp3"
}
if ([string]::IsNullOrWhiteSpace($AppApk)) {
    $AppApk = Join-Path $repoRoot "app\build\outputs\apk\standard\debug\app-standard-debug.apk"
}
if ([string]::IsNullOrWhiteSpace($TestApk)) {
    $TestApk = Join-Path $repoRoot "app\build\outputs\apk\androidTest\standard\debug\app-standard-debug-androidTest.apk"
}
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "$Backend-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
}
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$') {
    throw "RunId must contain only ASCII letters, digits, dot, underscore, and hyphen."
}

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label was not found: $Path"
    }
    (Resolve-Path -LiteralPath $Path).Path
}

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$RuntimeAar = Require-File $RuntimeAar "LiteRT AAR"
$ModelFile = Require-File $ModelFile "TFC-TDF model"
$SourceAudio = Require-File $SourceAudio "source audio"

if (-not $SkipBuild) {
    $gradlew = Join-Path $repoRoot "gradlew.bat"
    $revision = (& git -C $repoRoot rev-parse --verify HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($revision)) {
        throw "Unable to resolve the source revision."
    }
    $dirty = if (& git -C $repoRoot status --porcelain=v1 --untracked-files=normal) { "true" } else { "false" }
    $runtimeSha = Get-Sha256 $RuntimeAar
    $buildArgs = @(
        ":app:assembleStandardDebug",
        ":app:assembleStandardDebugAndroidTest",
        "--no-daemon",
        "--console=plain",
        "--no-watch-fs",
        "-PliteRtAar=$RuntimeAar",
        "-PbenchmarkSourceRevision=$revision",
        "-PbenchmarkSourceDirty=$dirty",
        "-PbenchmarkRuntimeId=local-bss-litert-android",
        "-PbenchmarkRuntimeVersion=2.2.0-bss.2",
        "-PbenchmarkRuntimeArtifactSha256=$runtimeSha"
    )
    & $gradlew @buildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Gradle build failed."
    }
}

$AppApk = Require-File $AppApk "standard debug APK"
$TestApk = Require-File $TestApk "standard debug test APK"
$adbCommand = Get-Command adb -ErrorAction SilentlyContinue
$adb = if ($adbCommand) { $adbCommand.Source } else { Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe" }
if (-not (Test-Path -LiteralPath $adb)) {
    throw "adb was not found: $adb"
}

function Invoke-Adb {
    param([string[]]$Arguments)
    & $adb -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed: $($Arguments -join ' ')"
    }
}

function Invoke-AdbText {
    param([string[]]$Arguments)
    $result = & $adb -s $Serial @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    ($result -join "`n").Trim()
}

function Get-RemoteSha([string]$Path) {
    $line = Invoke-AdbText @("shell", "sha256sum", $Path)
    if ($line -notmatch '^([0-9a-fA-F]{64})\s') {
        throw "Unable to read remote SHA-256 for $Path"
    }
    $Matches[1].ToLowerInvariant()
}

$deviceState = Invoke-AdbText @("get-state")
if ($deviceState -ne "device") {
    throw "Device is not ready: $Serial"
}

$modelName = Split-Path -Leaf $ModelFile
$sourceName = Split-Path -Leaf $SourceAudio
$modelSha = Get-Sha256 $ModelFile
$sourceSha = Get-Sha256 $SourceAudio
$runtimeSha = Get-Sha256 $RuntimeAar
$appSha = Get-Sha256 $AppApk
$testSha = Get-Sha256 $TestApk
$outputRoot = Join-Path $repoRoot "outputs\tfc-tdf-streaming-s25\$RunId"
if (Test-Path -LiteralPath $outputRoot) {
    throw "Run output already exists; choose a new RunId: $outputRoot"
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null

[ordered]@{
    schemaVersion = 1
    runId = $RunId
    serial = $Serial
    backend = $Backend
    threads = $Threads
    runtimeAar = [ordered]@{ path = $RuntimeAar; bytes = (Get-Item $RuntimeAar).Length; sha256 = $runtimeSha }
    model = [ordered]@{ path = $ModelFile; bytes = (Get-Item $ModelFile).Length; sha256 = $modelSha }
    source = [ordered]@{ path = $SourceAudio; bytes = (Get-Item $SourceAudio).Length; sha256 = $sourceSha }
    appApk = [ordered]@{ path = $AppApk; bytes = (Get-Item $AppApk).Length; sha256 = $appSha }
    testApk = [ordered]@{ path = $TestApk; bytes = (Get-Item $TestApk).Length; sha256 = $testSha }
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputRoot "host-identity.json") -Encoding utf8

if (-not $SkipInstall) {
    Invoke-Adb -Arguments @("install", "-r", $AppApk)
    Invoke-Adb -Arguments @("install", "-r", $TestApk)
}

Invoke-Adb -Arguments @("shell", "mkdir", "-p", "$remoteRoot/results/$RunId")
Invoke-Adb -Arguments @("push", $ModelFile, "$remoteRoot/$modelName")
Invoke-Adb -Arguments @("push", $SourceAudio, "$remoteRoot/$sourceName")
if ((Get-RemoteSha "$remoteRoot/$modelName") -cne $modelSha) {
    throw "Remote model SHA-256 mismatch."
}
if ((Get-RemoteSha "$remoteRoot/$sourceName") -cne $sourceSha) {
    throw "Remote source SHA-256 mismatch."
}

$remoteReport = "$remoteRoot/results/$RunId/report.json"
Invoke-Adb -Arguments @("shell", "rm", "-f", $remoteReport)
$sampleFile = Join-Path $outputRoot "host-resource-samples.jsonl"
$stdoutFile = Join-Path $outputRoot "instrumentation.stdout.txt"
$stderrFile = Join-Path $outputRoot "instrumentation.stderr.txt"
$instrumentArguments = @(
    "-s", $Serial, "shell", "am", "instrument", "-w", "-r",
    "-e", "class", $testClass,
    "-e", "backend", $Backend,
    "-e", "threads", $Threads.ToString(),
    "-e", "runId", $RunId,
    "-e", "sourceFile", $sourceName,
    "-e", "modelFile", $modelName,
    "-e", "sourceSha256", $sourceSha,
    "-e", "modelSha256", $modelSha,
    "-e", "playbackSeconds", $PlaybackSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-e", "seekAtSeconds", $SeekAtSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-e", "seekToSeconds", $SeekToSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "-e", "blockSamples", $BlockSamples.ToString(),
    $runner
)
$instrumentProcess = Start-Process -FilePath $adb -ArgumentList $instrumentArguments -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$reportText = ""
try {
    do {
        $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $thermal = (Invoke-AdbText @("shell", "dumpsys", "thermalservice") -split "`n" |
            Select-String -Pattern "Thermal Status|mValue=.*mName=(AP|BAT|SKIN)" |
            ForEach-Object { $_.Line.Trim() }) -join " | "
        $battery = (Invoke-AdbText @("shell", "dumpsys", "battery") -split "`n" |
            Select-String -Pattern "level:|temperature:|status:|voltage:|current now:" |
            ForEach-Object { $_.Line.Trim() }) -join " | "
        $memory = (Invoke-AdbText @("shell", "dumpsys", "meminfo", $package) -split "`n" |
            Select-String -Pattern "TOTAL PSS:|Native Heap:|Graphics:" |
            ForEach-Object { $_.Line.Trim() }) -join " | "
        $cpu = (Invoke-AdbText @("shell", "dumpsys", "cpuinfo", $package) -split "`n" |
            Select-String -Pattern ([regex]::Escape($package)) |
            ForEach-Object { $_.Line.Trim() }) -join " | "
        [ordered]@{
            timestampMs = $timestamp
            thermal = $thermal
            battery = $battery
            memory = $memory
            cpu = $cpu
        } | ConvertTo-Json -Compress | Add-Content -LiteralPath $sampleFile -Encoding utf8

        $reportLines = & $adb -s $Serial shell cat $remoteReport 2>$null
        $reportText = $reportLines -join "`n"
        if ($reportText -match '"status"\s*:\s*"(complete|error)"') {
            break
        }
        if ($instrumentProcess.HasExited -and [string]::IsNullOrWhiteSpace($reportText)) {
            Start-Sleep -Milliseconds 500
            if ($instrumentProcess.HasExited) {
                throw "Instrumentation exited without writing a report."
            }
        }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
} finally {
    if (-not $instrumentProcess.HasExited) {
        $instrumentProcess.Kill()
    }
    $instrumentProcess.WaitForExit()
}

if ($reportText -notmatch '"status"\s*:\s*"(complete|error)"') {
    throw "Streaming benchmark timed out. See $sampleFile and $stderrFile"
}
$reportPath = Join-Path $outputRoot "report.json"
$reportText | Set-Content -LiteralPath $reportPath -Encoding utf8

$report = $reportText | ConvertFrom-Json
if ($report.status -cne "complete") {
    throw "Streaming benchmark failed: $($report.message)"
}
if ($report.runId -cne $RunId -or $report.backend -cne $Backend -or
    $report.model.sha256 -cne $modelSha -or $report.source.sha256 -cne $sourceSha -or
    $report.runtime.artifactSha256 -cne $runtimeSha) {
    throw "Device report identity does not match the host inputs."
}
if ($Backend -eq "gpu-bounded") {
    $dispatches = @($report.timing.engine.sessions | ForEach-Object { $_.gpuEvidence.dispatchCount }) |
        Measure-Object -Sum
    if ($dispatches.Sum -lt 1) {
        throw "GPU run completed without positive bounded-GPU dispatch evidence."
    }
}

[pscustomobject]@{
    RunId = $RunId
    Backend = $Backend
    Device = $Serial
    FirstWetReadyMs = $report.playback.firstWetReadyWallMs
    FirstWetBlockMs = $report.playback.firstWetBlockWallMs
    SeekRewetMs = $report.playback.seekRewetLatencyMs
    DryBlocks = $report.playback.dryBlockCount
    WetBlocks = $report.playback.wetBlockCount
    FullChainRtf = $report.timing.fullChainRtf
    WallRtf = $report.playback.wallRtf
    Output = (Resolve-Path $outputRoot).Path
}
