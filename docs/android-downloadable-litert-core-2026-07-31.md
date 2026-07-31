# Downloadable LiteRT CPU core loading experiment

Date: 2026-07-31

## Decision

The explicit native-loader contract passes the CPU loading gate on Android API
26, 29, 31, 35, and 37 across `arm64-v8a`, `armeabi-v7a`, `x86_64`, and `x86`.
The classes-only API and separately downloaded CPU core are therefore viable
for the next Booming SS integration experiment.

The passing release is
[`downloadable-runtime-v2.1.5-bss.2-exp.2`](https://github.com/WluhWluh/bss-litert-android/releases/tag/downloadable-runtime-v2.1.5-bss.2-exp.2).
Its source-built API AAR is 86,013 bytes with SHA-256
`a68b51546f268b6db0b64bec3d1d95389ba44a48c59beaa1769794682c94b4f9`.
Two independent GitHub runners produced byte-identical release directories
before the prerelease was published.

This result does not yet make the component store production-ready. An
inter-process installation lock, cancellation and process-death tests, and GPU
component dependency ordering remain separate gates.

## Probe design

The `downloadableCore` product flavor contains the source-built classes-only
API but no LiteRT native library. ONNX Runtime remains in this research APK so
the same input can use a frozen numerical reference; it is unrelated to the
runtime-loading result.

Before using LiteRT, the probe:

1. Selects the artifact from the actual process bitness and ABI list.
2. Downloads an immutable release asset into a staging directory under
   `noBackupFilesDir`.
3. Verifies the exact ZIP size and SHA-256.
4. Accepts only `manifest.json` and `libLiteRt.so`; the native library is
   streamed to disk through a 64 KiB buffer while its size and SHA-256 are
   checked.
5. Verifies the manifest contract, inner sizes, ABI, SONAME, and file hashes.
6. Flushes both files, marks them read-only, and atomically installs the
   versioned directory.
7. Calls `LiteRtNativeLibraryLoader.configureAbsolutePath()` and `load()`
   before first use of any other LiteRT API class.
8. Confirms same-path reconfiguration is idempotent and a different path is
   rejected with `IllegalStateException`.

`Environment`, `CompiledModel`, and `TensorBuffer` route their static native
initialization through the same loader. If no path is configured, the API
retains complete-AAR compatibility through `System.loadLibrary("LiteRt")`.

The final debug APK sizes were:

| Flavor | APK size | LiteRT native entries | LiteRT native bytes |
| --- | ---: | ---: | ---: |
| `standardDebug` | 215.56 MiB | 6 | 29,717,216 |
| `downloadableCoreDebug` | 184.33 MiB | 0 | 0 |

The downloadable flavor is 31.23 MiB smaller. This is not a Booming SS size
projection: both research flavors still package ONNX Runtime and benchmark
assets. The downloadable APK contains no `libLiteRt.so`, GPU accelerator, or
BSS OpenCL library.

## Results

All inference rows used the same Coast Town input window and converted
`UVR_MDXNET_3_9662` FP32 model. Each row used one untimed setup followed by one
recorded inference, so timings are functional observations rather than stable
performance baselines.

| Device | API / ABI | Scenario | Runtime result | 9662 result |
| --- | --- | --- | --- | --- |
| Galaxy S25 SM-S9310 | 35 / arm64-v8a | First install | Downloaded and loaded in 3605.35 ms; loader 1.30 ms | 2141.29 ms; 97.597 dB SNR |
| Galaxy S25 SM-S9310 | 35 / arm64-v8a | New process | Disk reuse in 9.60 ms; loader 0.75 ms | 1949.03 ms; 97.597 dB SNR |
| Galaxy S25 SM-S9310 | 35 / arm64-v8a | Same process | PID unchanged; `reusedInProcess=true` | Complete; 97.597 dB SNR |
| Galaxy S25 SM-S9310 | 35 / arm64-v8a | Corrupt first library byte | Bad hash rejected; redownloaded in 2057.57 ms | Complete; restored SHA-256; 97.597 dB SNR |
| Galaxy S10 SM-G9730 | 31 / arm64-v8a | First install | Downloaded and loaded in 4462.47 ms; loader 2.02 ms | 2590.43 ms; 97.737 dB SNR |
| Galaxy S10 SM-G9730 | 31 / arm64-v8a | New process | Disk reuse in 28.45 ms; loader 1.04 ms | 2594.29 ms; 97.737 dB SNR |
| Galaxy S10 SM-G9730 | 31 / armeabi-v7a | First install | 32-bit process; downloaded in 2027.81 ms; loader 1.60 ms | 2997.88 ms; 97.598 dB SNR |
| Galaxy S10 SM-G9730 | 31 / armeabi-v7a | New process | Disk reuse in 45.68 ms; loader 1.64 ms | 3216.48 ms; 97.598 dB SNR |
| API 26 emulator | 26 / x86 | First install | Downloaded and loaded in 3869.49 ms; loader 0.89 ms | 3503.26 ms; 97.556 dB SNR |
| API 26 emulator | 26 / x86 | New process | Disk reuse in 89.72 ms; loader 1.44 ms | 2175.43 ms; 97.556 dB SNR |
| API 29 emulator | 29 / x86 | 16 MiB growth limit | Streaming install and explicit load completed | Harness input `readBytes()` exceeded the test heap |
| API 29 emulator | 29 / x86 | 512 MiB growth limit | Disk reuse in 138.66 ms; loader 4.63 ms | 2932.40 ms; 97.767 dB SNR |
| API 37 emulator | 37 / x86_64 | First install | Downloaded and loaded in 15644.73 ms; loader 57.86 ms | 14310.00 ms; 97.599 dB SNR |
| API 37 emulator | 37 / x86_64 | New process | Disk reuse in 178.69 ms; loader 35.36 ms | 3731.71 ms; 97.599 dB SNR |

The S10, S25, and API 26 first runs executed concurrently. API 37 also showed
a large cold-host penalty. These timings must not be used to compare device or
ABI performance. Across all complete runs, maximum absolute error to the ORT
reference remained between `5.71e-05` and `7.11e-05`.

The installed arm64 library retained SHA-256
`ae2b996fde27021b070e88b56eebc9626a5261feb72f09791bdac38b2f09abd2`.
The corruption test changed that identity, triggered replacement, and restored
the exact expected hash before inference.

## Minimum-API attribution

The exp.1 probe called application-side `System.load()` and then encountered
the API classes' hardcoded `System.loadLibrary("LiteRt")`. On API 26, the
second call searched the APK native directories and failed even though the
same file was already mapped by absolute path. Changing the x86 library's
historical SONAME from `LiteRt` to `libLiteRt.so` did not change that result.

Exp.2 removes the second ClassLoader lookup. All three API static initializers
call `LiteRtNativeLibraryLoader.load()`, which reuses the configured absolute
path. The same API 26 AVD then completed real x86 inference, confirming that
the old failure was in Java-side load routing rather than x86 LiteRT itself.

The API 29 AVD initially had a corrupt `settings_system.xml`, which prevented
`system_server` from completing boot. A clean data partition fixed that
unrelated AVD failure. Its 16 MiB app growth limit then exposed a duplicate
7.48 MiB allocation in ZIP extraction. Streaming extraction fixed the runtime
installation; the later 8 MiB benchmark-input allocation required restoring
the intended 512 MiB test growth limit.

## Remaining gates

- Add a cross-process installation lock before Booming SS main and isolated
  processes can install the same component concurrently.
- Test cancellation and process death during download, extraction, atomic
  move, and first native initialization.
- Apply the same store to the bounded GPU component, loading dependencies in
  order and verifying accelerator discovery and CPU fallback.
- Define retention, update, and rollback policy for downloaded runtime
  versions independently from model weights.
- Repeat low-memory tests in the Booming SS process architecture; the harness
  keeps large Java input/output arrays that the production pipeline does not.

Raw reports, APKs, models, tensors, and emulator logs remain ignored local
artifacts. Stable implementation and release inputs live in Git.
