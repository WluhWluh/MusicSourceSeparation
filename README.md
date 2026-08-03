# Music Source Separation research harness

This repository is an Android research app and benchmark harness for local,
two-stem UVR/MDX music source separation. It was used to validate the Android
audio pipeline, reproduce MDX DSP, compare ONNX Runtime with LiteRT, and test
converted TFLite models on real devices and emulators before integrating the
work into [Booming SS](https://github.com/WluhWluh/BoomingSS).

It is not a release build of Booming SS and is not intended as a polished
end-user application. ONNX Runtime and LiteRT are deliberately present at the
same time so the same APK can make controlled runtime comparisons.

## What is in the repository

- `app/`: a small native Android app for selecting audio, exporting decoded
  PCM as WAV, running ONNX-based MDX separation, and executing headless
  inference benchmarks.
- `tools/mdx_reference.py`: the desktop ONNX reference pipeline and model
  inspection utility.
- `tools/run_android_inference_benchmark.ps1`: the ADB benchmark driver. It
  uploads models and inputs, starts the foreground service, samples device
  state, and pulls JSON results.
- `tools/prepare_litert_qnn_runtime.py`: prepares a pinned, ignored
  LiteRT/QAIRT HTP v69, v73, v75, v79, or v81 JIT payload after explicit
  license acceptance.
- `tools/run_android_qnn_audio_benchmark.ps1`: runs one complete WAV through a
  single QNN session and pulls verified stems, device samples, and QNN IR.
- `tools/evaluate_mdx_tensor_window.py`: prepares real MDX windows and
  reconstructs reference/candidate output tensors into listening files.
- `tools/generate_ort_tensor_reference.py`: generates an element-wise desktop
  ORT output tensor for numerical comparisons.
- `docs/model_contracts.md`: MDX tensor and DSP assumptions used by the Android
  implementation.
- `docs/android-litert-benchmark-2026-07-19.md`: S10 and S25 ORT/LiteRT CPU and
  GPU results for UVR MDXNET 9482.
- `docs/android-litert-x86-build-2026-07-20.md`: the original 32-bit x86 LiteRT
  build investigation and emulator results.
- `docs/android-litert-qnn-s25-2026-07-29.md`: Qualcomm HTP v79 delegation,
  UI, lifecycle, reconstructed-audio, and full-song results on Galaxy S25.
- `docs/progress.md`: the chronological MVP development log. It is useful for
  history, but the focused benchmark reports above are the current result
  summaries.

## Models and test data

Model weights, source audio, generated stems, raw benchmark captures, and APKs
are intentionally excluded from Git. Put the two ONNX models used by the app
under `models/uvr-mdx/`:

```text
models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx
models/uvr-mdx/UVR_MDXNET_9482.onnx
```

The Gradle build exposes this directory as Android assets. A build can compile
without the files, but the interactive ONNX actions will fail at runtime until
the selected model is present. Converted TFLite models used by the benchmark
service are uploaded separately over ADB and are not packaged as assets.

Do not add model files to this repository without first resolving their exact
provenance and redistribution terms. Conversion tooling and release metadata
for Booming SS presets are maintained separately in
[bss-tflite](https://github.com/WluhWluh/bss-tflite).

## Prerequisites

- JDK 17 or newer.
- Android SDK platform 37 and an SDK path available through `local.properties`,
  `ANDROID_HOME`, or `ANDROID_SDK_ROOT`.
- An Android device or emulator visible to ADB for device benchmarks.
- PowerShell 7 for the checked-in Android benchmark driver.
- Python and the packages pinned in `requirements-phase3.txt` for desktop ONNX
  reference work.

The benchmark driver looks for ADB at the standard Windows SDK location,
`%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe`.

## Build and unit tests

From the repository root on Windows:

```powershell
.\gradlew.bat testDebugUnitTest assembleDebug
```

The debug APK is written to:

```text
app/build/outputs/apk/debug/app-debug.apk
```

For the API 26 pure x86 emulator, use the non-streaming installer if the
streamed install stalls:

```powershell
adb -s emulator-5554 install -r --no-streaming app/build/outputs/apk/debug/app-debug.apk
```

## Desktop reference pipeline

Create a virtual environment and install the pinned dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase3.txt
.\.venv\Scripts\python.exe tools/mdx_reference.py self-test
```

Inspect a model and run a zero-input smoke inference:

```powershell
.\.venv\Scripts\python.exe tools/mdx_reference.py inspect `
  --model models/uvr-mdx/UVR_MDXNET_9482.onnx `
  --smoke-run
```

Run `python tools/mdx_reference.py --help` for separation commands and
`python tools/generate_ort_tensor_reference.py --help` for raw tensor reference
generation.

Compare a Phase 7 Android export with its desktop ORT reference (the Android
runner writes `cache-manifest.json` beside the exported stems):

```powershell
.\.venv\Scripts\python.exe tools/compare_phase7_audio.py `
  --actual-dir <BoomingSS-build>\<device>\<run>-artifacts `
  --reference-dir outputs/phase7/9662-full-wav `
  --source data/samples/_-_Coast_Town__decoded.wav `
  --cache-manifest <BoomingSS-build>\<device>\<run>-artifacts\cache-manifest.json `
  --fixture-id coast_town_full_wav `
  --output <comparison-report>.json
```

The report distinguishes one-LSB PCM quantization from larger numerical
differences, evaluates every exported segment boundary, and reports the
actual-versus-reference reconstruction error delta required by the current
Phase 7 threshold revision. It is a local validation artifact; full audio and
generated reports remain outside Git.

### Phase 7 source-format corpus

Generate the deterministic 15-second format matrix used to validate Booming
SS window decoding and full-song fallback routes:

```powershell
python tools/generate_phase7_format_corpus.py
```

The generator requires the pinned Gyan FFmpeg 8.1.1 full build. It creates a
44.1 kHz stereo PCM source with signal markers at MDX generation boundaries,
then writes WAV, FLAC, Vorbis, gapless and no-Xing MP3, AAC, 48 kHz fallback,
and wrong-extension fixtures under `data/phase7/source-formats-v1/`. Ogg stream
serials are normalized and page checksums are rebuilt so repeated runs produce
identical hashes. `manifest.generated.json` records each command, format,
duration, byte size, and SHA-256.

The generated audio and manifest remain ignored. The frozen hashes and
expected Android decode routes live with the app-side validation contract in
the Booming SS repository. Regenerate the complete corpus after changing the
signal, FFmpeg build, codec settings, or normalization logic; do not replace an
individual fixture under an existing corpus schema version.

## Android inference benchmark

The benchmark service compares these backends:

- `ort`: ONNX Runtime CPU.
- `litert_cpu`: LiteRT CPU with XNNPACK.
- `litert_gpu`: LiteRT GPU with FP16 precision.
- `litert_gpu_fp32`: LiteRT GPU with FP32 precision.
- `litert_gpu_bounded`: the audited N=1 bounded OpenCL FP32 runtime.
- `litert_qnn`: Qualcomm HTP through the QNN JIT plugin in a matching
  generation-specific QNN flavor.

Build from a clean revision and embed the benchmark source/runtime identity in
the APK. The runtime AAR passed to the runner must be the same file used by
Gradle:

```powershell
$revision = (git rev-parse HEAD).Trim()
$runtimeAar = "<path-to-litert-android-2.1.5-bss.2.aar>"
.\gradlew.bat :app:assembleStandardDebug `
  "-PliteRtAar=$runtimeAar" `
  "-PbenchmarkSourceRevision=$revision" `
  -PbenchmarkSourceDirty=false `
  -PbenchmarkRuntimeId=litert-android-2.1.5-bss.2
```

Install that APK, then define one frozen parameter set for the 9662 calibration
model. Tensor dimensions come from the versioned contract and cannot be
overridden on the command line:

```powershell
$benchmark = @{
  Serial = "DEVICE_SERIAL"
  ModelId = "uvr_mdxnet_3_9662"
  ContractFile = "app/src/main/assets/benchmark-contracts/uvr_mdxnet_3_9662.json"
  OnnxModel = "models/uvr-mdx-candidates/trvlvr-all-public-uvr-models/UVR_MDXNET_3_9662.onnx"
  LiteRtModel = "<path-to-UVR_MDXNET_3_9662_static_float32.tflite>"
  InputFile = "<path-to-frozen-input-nchw-f32.bin>"
  AppApk = "app/build/outputs/apk/standard/debug/app-standard-debug.apk"
  RuntimeArtifact = $runtimeAar
  Iterations = 20
  Warmups = 2
  Threads = 4
  KeepActivityForeground = $true
}

.\tools\run_android_inference_benchmark.ps1 @benchmark `
  -Backend ort -UploadModels -Tag 9662-ort
.\tools\run_android_inference_benchmark.ps1 @benchmark `
  -Backend litert_cpu -Tag 9662-litert-cpu
.\tools\run_android_inference_benchmark.ps1 @benchmark `
  -Backend litert_gpu_bounded -Tag 9662-litert-gpu-bounded
```

Run ORT first when numerical comparison is required. The service stores its
output as the device-side reference used by subsequent LiteRT runs. The first
`-UploadModels` run uploads the contract, both model artifacts, and the required
little-endian float32 NCHW input. It also clears any previous device-side ORT
reference, so do not use `-UploadModels` again within the same comparison batch.

Every invocation writes `host-identity.json` with APK/runtime/model/input hashes
and rejects an existing local tag. The completed device report must match the
contract, ONNX, LiteRT, input, source revision, dirty state, and runtime AAR
identities before the runner accepts it. The runner also hashes the installed
base APK and requires it to match `AppApk` exactly.
Foreground mode wakes and unlocks the test device and keeps the benchmark
Activity's screen on; this prevents Android 15 FGS rejection and long-run UID
throttling after the normal display timeout.

Host-side reports and thermal/battery samples are written below:

```text
outputs/android-benchmark/<device-serial>/<tag>/
```

These raw files remain ignored; stable conclusions belong in `docs/`.

## Qualcomm HTP research flavors

The `qnnV69`, `qnnV73`, `qnnV75`, and `qnnV79` flavors are arm64/API 31
research builds. They pin LiteRT 2.1.5 and QAIRT 2.44.0.260225 while packaging
one HTP generation per APK. Review the QAIRT license before preparing an
ignored local runtime:

```powershell
python -m pip install -r requirements-qnn.txt
python tools/prepare_litert_qnn_runtime.py `
  --htp-version 75 `
  --accept-qairt-license

$revision = (git rev-parse HEAD).Trim()
$runtimeAar = "<path-to-litert-android-2.1.5-bss.2.aar>"
.\gradlew.bat :app:assembleQnnV75Debug `
  "-PliteRtAar=$runtimeAar" `
  "-PbenchmarkSourceRevision=$revision" `
  -PbenchmarkSourceDirty=false `
  -PbenchmarkRuntimeId=litert-android-2.1.5-bss.2
```

For a QNN tensor run, install the generated QNN APK and use the same frozen
arguments as the CPU/GPU batch, including `ContractFile`, `InputFile`,
`RuntimeArtifact`, `AcceleratorBundleManifest`, and the QNN `AppApk`. Run ORT
with `-UploadModels` first on that device, then invoke `-Backend litert_qnn`; a
result is accepted only when
the report records verified delegation and a frozen accelerator bundle hash.

The runtime preparation tool also has frozen file metadata for HTP v69, v73,
and v81. Add a Gradle flavor only when that generation is ready for device
validation. The legacy `prepare_litert_qnn_v79_runtime.py` command remains as
a v79-compatible entry point.

LiteRT 2.1.5's built-in Qualcomm compatibility checker only recognizes
SM8550 and newer SoCs. The research harness preserves that checker and adds an
explicit API 31+ allowlist for QTI/Qualcomm SM8450 and SM8475 so HTP v69 can be
validated on Snapdragon 8 Gen 1 and 8+ Gen 1 devices. This only bypasses the
Java preflight gate; successful provider registration, graph compilation, and
inference remain required before a device result is accepted.

To package the self-contained BrowserStack fixtures for a v75 validation APK,
prepare the runtime first and then run:

```powershell
python tools/prepare_app_live_validation_assets.py `
  --htp-version 75 `
  --campaign app-live-oneplus-13r-qnn-v75-v1
```

Fresh installs default to `BrowserStack App Live`; `Local control` remains a
persistent manual option. App Live assets contain local fixtures and a
write-only relay credential, so they remain ignored by Git.

App Live validation is fail-closed for QNN. Provider readiness and an `NPU`
accelerator label are only preflight signals; every QNN tensor and audio stage
must emit at least one non-empty QNN IR partition. Reports record
`delegationStatus` as `delegated`, `cpu_fallback`, `provider_unavailable`, or
`indeterminate`. Each stage uploads its report and IR before validation, then
uploads an app-process logcat and a progress checkpoint. Device identity also
includes the full build fingerprint, security patch, vendor properties, APK
inventory, and hashes of all packaged native libraries.

One BrowserStack Galaxy Tab S8 session emitted no QNN IR and behaved like a
single-thread CPU fallback. That remote device was unusually difficult to
connect to and severely laggy, so the observation is not a model-wide Tab S8
compatibility result. It remains a useful failed-session artifact pending a
repeat on an independently healthy SM-X706B or another SM8450 tablet.

The full-song driver accepts a canonical 44.1 kHz stereo PCM16 WAV (up to 128
MiB of PCM), uploads it with the frozen 9662 TFLite model, creates one QNN
session, samples device state, and pulls SHA-256-verified stems and QNN IR:

```powershell
.\tools\run_android_qnn_audio_benchmark.ps1 `
  -Serial <adb-serial> `
  -SourceAudio <path-to-canonical-pcm16-test.wav> `
  -ContractFile app/src/main/assets/benchmark-contracts/uvr_mdxnet_3_9662.json `
  -OnnxModel models/uvr-mdx-candidates/trvlvr-all-public-uvr-models/UVR_MDXNET_3_9662.onnx `
  -LiteRtModel <path-to-UVR_MDXNET_3_9662_static_float32.tflite> `
  -AppApk app/build/outputs/apk/qnnV79/debug/app-qnnV79-debug.apk `
  -RuntimeArtifact $runtimeAar `
  -AcceleratorBundleManifest .tmp/litert-qnn-v79-runtime/runtime-manifest.json `
  -Tag qnn-full-song
```

`-ReuseDeviceFiles` compares the contract, ONNX, LiteRT model, and audio SHA-256
values on both host and device. Tags are write-once on the host so reruns cannot
append samples to an earlier result directory.

These flavors are not redistributable runtime packages or claimed Booming SS
backends. See
[the S25 QNN report](docs/android-litert-qnn-s25-2026-07-29.md) for the
measured benefits and 145.2 MiB installed runtime cost, and the
[cross-generation device matrix](docs/android-litert-qnn-device-matrix-2026-07-31.md)
for the v69 through v79 results, failed delegation boundaries, and closure
decision.

## 32-bit x86 LiteRT runtime

The official LiteRT 2.1.5 AAR omits Android's 32-bit `x86` ABI. The test app
therefore includes `app/libs/litert-2.1.5-x86.aar`, the exact supplemental AAR
used for the July 20 benchmark. Its native payload has SHA-256:

```text
02b6556ec235926c11eb0c067eb16e459adcddb1568a42eefe0c40f4cc4b59af
```

The reproducible producer and canonical release artifacts now live in
[bss-litert-android](https://github.com/WluhWluh/bss-litert-android). New x86
runtime builds must be made there. The historical scripts retained here exist
only to reproduce the documented experiment; they are not the release
pipeline.

## Generated and local-only files

The following are intentionally ignored:

- Gradle, Kotlin, and Android build outputs.
- `models/`, `data/`, `outputs/`, `dist/`, and `.tmp/`.
- Python virtual environments and bytecode.
- `local.properties`, IDE metadata, APKs, and app bundles.

Before committing a benchmark result, summarize the stable measurements in a
document under `docs/` rather than checking in raw tensors, audio, or device
captures.

## Licensing and provenance

Unless otherwise noted, source code and documentation authored for this
repository are licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for project attribution and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency and provenance
details.

This license does not apply to model weights, converted model artifacts,
audio files, generated stems, benchmark input tensors, or third-party
binaries. Those materials remain subject to their respective source terms.
In particular, the historical supplemental LiteRT x86 AAR is an upstream
derivative retained for benchmark reproducibility and is not relicensed as
original MusicSourceSeparation code.

No model license or redistribution permission is implied by model
compatibility, conversion instructions, hashes, or benchmark results in this
repository.
