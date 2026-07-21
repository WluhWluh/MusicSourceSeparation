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
- `tools/generate_ort_tensor_reference.py`: generates an element-wise desktop
  ORT output tensor for numerical comparisons.
- `docs/model_contracts.md`: MDX tensor and DSP assumptions used by the Android
  implementation.
- `docs/android-litert-benchmark-2026-07-19.md`: S10 and S25 ORT/LiteRT CPU and
  GPU results for UVR MDXNET 9482.
- `docs/android-litert-x86-build-2026-07-20.md`: the original 32-bit x86 LiteRT
  build investigation and emulator results.
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

## Android inference benchmark

The benchmark service compares these backends:

- `ort`: ONNX Runtime CPU.
- `litert_cpu`: LiteRT CPU with XNNPACK.
- `litert_gpu`: LiteRT GPU with FP16 precision.
- `litert_gpu_fp32`: LiteRT GPU with FP32 precision.

Build and install the app first. Then define one parameter set so every backend
uses the same model files and identity:

```powershell
$benchmark = @{
  Serial = "DEVICE_SERIAL"
  ModelId = "uvr_mdxnet_9482"
  OnnxModel = "models/uvr-mdx/UVR_MDXNET_9482.onnx"
  LiteRtModel = "C:\path\to\UVR_MDXNET_9482_static_float32.tflite"
  Height = 2048
  Width = 256
  Threads = 8
}

.\tools\run_android_inference_benchmark.ps1 @benchmark `
  -Backend ort -UploadModels -Tag 9482-ort
.\tools\run_android_inference_benchmark.ps1 @benchmark `
  -Backend litert_cpu -Tag 9482-litert-cpu
.\tools\run_android_inference_benchmark.ps1 @benchmark `
  -Backend litert_gpu_fp32 -Tag 9482-litert-gpu-fp32
```

Run ORT first when numerical comparison is required. The service stores its
output as the device-side reference used by subsequent LiteRT runs. Add
`-InputFile <path>` to upload a little-endian float32 NCHW tensor instead of
using the deterministic generated input.

Host-side reports and thermal/battery samples are written below:

```text
outputs/android-benchmark/<device-serial>/<tag>/
```

These raw files remain ignored; stable conclusions belong in `docs/`.

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

This repository does not currently declare a repository-wide source license.
Third-party dependencies retain their own licenses. The supplemental LiteRT
AAR contains the upstream LiteRT license, and the canonical runtime releases
publish complete notices separately. No model license is implied by model
compatibility or by the conversion and benchmark instructions in this
repository.
