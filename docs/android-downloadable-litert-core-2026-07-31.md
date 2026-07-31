# Downloadable LiteRT CPU core loading experiment

Date: 2026-07-31

## Decision

Absolute-path preloading is viable on the tested Android 12, 15, and 16
processes, but it is not sufficient for the app's current minimum Android 8
runtime contract.

The existing unmodified LiteRT API classes can be retained on the successful
systems: after `System.load(<app-private-path>/libLiteRt.so)`, their hardcoded
`System.loadLibrary("LiteRt")` calls complete and real 9662 inference works.
On the API 26 pure-x86 emulator, the later `System.loadLibrary` call fails even
though the same native library was already loaded successfully. A pure API AAR
plus an external native core is therefore a no-go for all supported Android
versions until the API layer gains an explicit native loader contract.

## Probe design

The `downloadableCore` product flavor depends only on the pure API AAR from
[`downloadable-runtime-v2.1.5-bss.2-exp.1`](https://github.com/WluhWluh/bss-litert-android/releases/tag/downloadable-runtime-v2.1.5-bss.2-exp.1).
Its APK contains no `libLiteRt.so`, GPU accelerator, or BSS OpenCL library.
ONNX Runtime remains in this research APK so the same input can retain a frozen
numerical reference; it is unrelated to the runtime-loading result.

Before using LiteRT, the probe:

1. Selects the artifact from the process bitness and supported ABI list.
2. Downloads an immutable release asset into a staging directory under
   `noBackupFilesDir`.
3. Verifies the exact ZIP size and SHA-256.
4. Accepts only `manifest.json` and `libLiteRt.so`, with bounded extraction.
5. Verifies the manifest contract, inner sizes, SHA-256 values, ABI, and
   SONAME.
6. Flushes both files, marks them read-only, and atomically installs the
   versioned directory.
7. Calls `System.load()` with the installed absolute path before first use of
   `Environment`, `CompiledModel`, or `TensorBuffer`.

The benchmark report records the selected release, ABI, URL, all hashes,
installed path, whether a download occurred, file writability, and loader
timings. The implementation is intentionally pinned to one prerelease; it is
not a mutable catalog or production update channel.

The rebuilt debug APKs measured:

| Flavor | APK size | LiteRT native entries | LiteRT native bytes |
| --- | ---: | ---: | ---: |
| `standardDebug` | 215.55 MiB | 6 | 29,717,216 |
| `downloadableCoreDebug` | 184.36 MiB | 0 | 0 |

The downloadable flavor is 31.19 MiB smaller. This is a research-harness APK,
not a Booming SS size projection: both flavors still package ONNX Runtime and
local benchmark assets. The downloadable APK retained eight ONNX Runtime
native entries, while the audit found no `libLiteRt.so`, GPU accelerator, or
BSS OpenCL entry.

## Results

All successful runs used the same Coast Town input window and the converted
UVR_MDXNET_3_9662 FP32 model.

| Device | OS / ABI | Scenario | Loader result | 9662 result |
| --- | --- | --- | --- | --- |
| Galaxy S25 SM-S9310 | API 35 / arm64-v8a | First install | Downloaded and loaded in 2427.95 ms; `System.load` 2.02 ms | Complete; 939.59 ms mean, 97.597 dB SNR to ORT |
| Galaxy S25 SM-S9310 | API 35 / arm64-v8a | New process | Reused disk install in 7.95 ms; `System.load` 1.09 ms | Complete; 1009.17 ms mean |
| Galaxy S25 SM-S9310 | API 35 / arm64-v8a | Same process | `reusedInProcess=true`; PID unchanged | Complete |
| Galaxy S25 SM-S9310 | API 35 / arm64-v8a | One-byte corrupt library | Rejected local copy and redownloaded in 1967.85 ms | Complete; 97.597 dB SNR to ORT |
| Galaxy S10 SM-G9730 | API 31 / arm64-v8a | First install | Downloaded and loaded in 2640.76 ms; `System.load` 4.41 ms | Complete; 97.737 dB SNR to ORT |
| API 37 emulator | API 37 / x86_64 | First install | Downloaded and loaded in 12420 ms; `System.load` 14 ms | Complete with finite output |
| API 37 emulator | API 37 / x86_64 | New process | Reused disk install in 81.79 ms; `System.load` 13.44 ms | Complete; 97.599 dB SNR to ORT |
| API 26 emulator | API 26 / x86 | First install | `System.load` succeeded; later API class initialization failed | No LiteRT inference |

The S10 was already at thermal status 3 and 5-6% battery, so its inference
time is only a functional result, not a performance baseline. The API 37
emulator's first download time is similarly host-network dependent.

The S25 process map showed executable pages coming from:

```text
/data/data/com.example.musicsourceseparation/no_backup/
  downloadable-litert-core/2.1.5-bss.2-exp.1/arm64-v8a/libLiteRt.so
```

The installed file had the expected SHA-256
`ae2b996fde27021b070e88b56eebc9626a5261feb72f09791bdac38b2f09abd2`,
mode `0500`, and no second LiteRT mapping from the APK native directory.

## API 26 failure attribution

The x86 release library has historical SONAME `LiteRt`, while the other CPU
libraries use `libLiteRt.so`. To determine whether that difference caused the
failure, a local test copy was changed only to SONAME `libLiteRt.so`, its
manifest and expected hashes were updated, and it was installed directly into
the probe directory.

The result was unchanged. Absolute-path loading completed in 1.59 ms, then
`Environment.<clinit>` failed because the API 26 `PathClassLoader` could not
find `libLiteRt.so` in the APK native search directories. All those directories
were confirmed empty. This attributes the failure to old Android's
`Runtime.loadLibrary0` / ClassLoader path resolution, not to the custom x86
SONAME or a stale packaged library.

An API 29 low-memory x86 AVD connected to ADB but did not complete framework
boot during this run, so the exact Android-version transition remains unknown.
Production code must not infer support from an assumed API threshold.

## Next contract

The next experiment should make native loading explicit in the classes-only
AAR produced by `bss-litert-android`:

- expose a one-time loader configuration before any LiteRT API class is used;
- let the API classes call that loader instead of unconditional
  `System.loadLibrary("LiteRt")`;
- reject conflicting paths and preserve thread-safe, idempotent initialization;
- test first download, process restart, same-process reuse, corruption,
  cancellation, and process death on API 26, 29, 31, 35, and 37;
- repeat for arm64-v8a, armeabi-v7a, x86_64, and x86 where devices are
  available;
- add an inter-process installation lock before integrating with Booming SS,
  because its main and isolated inference processes must not install the same
  component concurrently.

GPU components were deliberately outside this gate. Once the CPU loader works
across the minimum OS matrix, the bounded GPU bundle can reuse the same
verified component store and add dependency-order and accelerator-discovery
tests.

Raw reports, models, tensors, APKs, and the SONAME probe remain ignored local
artifacts. Stable implementation and release inputs live in Git; benchmark
captures do not.
