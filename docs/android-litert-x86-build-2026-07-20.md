# LiteRT 2.1.5 CPU-only x86 build and benchmark

Date: 2026-07-20

> This document records the original local experiment and the exact AAR used
> for its benchmark. Reproducible builds and canonical release artifacts now
> live in
> [bss-litert-android](https://github.com/WluhWluh/bss-litert-android). Release
> `v2.1.5-bss.1` contains the same `libLiteRt.so` payload (SHA-256 `02b6556e...`)
> in a more complete release package. Its AAR hash differs because the release
> includes normalized licensing and package metadata.

## Conclusion

LiteRT 2.1.5 can be built and packaged for Android's 32-bit `x86` ABI. The
published Maven AAR omits this ABI, but the source and XNNPACK dependency still
contain the required x86 build rules and kernels.

The resulting supplemental AAR loaded successfully on a pure x86 API 26
emulator. Two 2048-bin UVR presets completed with correct output and were about
14.65-15.55 times faster than `tensorflow-lite:2.16.1` on the same emulator.
The larger 2560-bin HQ4 preset still could not fit its XNNPACK runtime in this
2 GB, 32-bit environment.

## Pinned build inputs

- LiteRT version: `2.1.5`.
- LiteRT source commit: `9d26e89d88ef8785b6a1e54ec41ac8add215a125`.
- Bazel: `7.7.0`.
- Android NDK: `25.1.8937393` (r25b), Linux archive.
- NDK archive SHA-256:
  `403ac3e3020dd0db63a848dcaba6ceb2603bf64de90949d5c4361f848e44b005`.
- `rules_android_ndk`: `0.1.3`, archive SHA-256:
  `89bf5012567a5bade4c78eac5ac56c336695c3bfd281a9b0894ff6605328d2d5`.
- Build host: WSL Ubuntu 24.04 in the `Ubuntu-LiteRT` distribution.

The source patch used for this historical run is
`tools/litert-2.1.5-x86/litert-2.1.5-x86.patch`. It:

1. Registers the modern `rules_android_ndk` toolchain because Bazel's legacy
   built-in rule supports only NDK r21 and older layouts.
2. Adds two missing direct dependencies needed by LiteRT's `cpu_only` build.
   The implementations remain CPU-only; the GPU environment dependency resolves
   to the upstream stub under this build configuration.

The build disables AVX-VNNI, AVX-VNNI-INT8, AVX512-FP16, and AMX microkernels.
They are not required by the API 26 x86 target and are not accepted uniformly
by the older compiler path. SSE through AVX2 and supported AVX512 families
remain available for XNNPACK runtime dispatch.

## Why NDK r21 was rejected

The legacy Bazel NDK rule can analyze and compile much of the target with NDK
r21e, but it is not a valid LiteRT 2.1.5 toolchain:

- Clang 9 rejects newer XNNPACK switches such as `-mavxvnni`.
- Its default GNU linker does not support LiteRT's
  `--undefined-glob=LiteRt*` option.
- Switching to its bundled `lld` exposes missing Android libc++
  `std::filesystem` implementations and a missing compiler-rt
  `__isOSVersionAtLeast` symbol.

Pointing the legacy rule at r25b also does not work. It generates removed GCC
4.9, `platforms/android-23`, and `sources/cxx-stl` paths. The standalone
`rules_android_ndk` repository is the supported bridge between Bazel 7.7 and
the unified r25b toolchain.

## Reproduction

From Windows, with the pinned LiteRT source and NDK installed in WSL:

```powershell
wsl -d Ubuntu-LiteRT -- bash /mnt/c/Users/User/Documents/MusicSourceSeparation/tools/litert-2.1.5-x86/build.sh

./tools/package_litert_x86_aar.ps1 `
  -SharedLibrary .tmp/litert-2.1.5-x86/libLiteRt.so `
  -LicenseFile \\wsl.localhost\Ubuntu-LiteRT\opt\litert-v2.1.5\LICENSE
```

The build script normalizes CRLF in Bazel and shell files before patching.
This is required when the source archive has passed through a Windows checkout,
because multiline Bazel genrules otherwise pass a literal carriage return to
Linux `/bin/bash`.

The AAR packager uses a fixed entry order and timestamp. Repeated packaging of
the same library and license produces the same SHA-256.

## Artifact verification

`libLiteRt.so`:

- Size: `7,482,132` bytes (7.14 MiB uncompressed).
- SHA-256:
  `02b6556ec235926c11eb0c067eb16e459adcddb1568a42eefe0c40f4cc4b59af`.
- ELF class: `ELF32`.
- Machine: `Intel 80386`.
- Type: shared object.
- `DT_NEEDED`: `libandroid.so`, `libdl.so`, `libm.so`, `liblog.so`, and
  `libc.so` only.
- No EGL, GLES, OpenCL, Vulkan, or shared libc++ dependency.

`app/libs/litert-2.1.5-x86.aar`:

- Size: `2,994,968` bytes (2.86 MiB compressed).
- SHA-256:
  `5b392c09b02a33b03fcbdf249dc0a7feebf71d6ce4fd9c22b9c3f79e24947f41`.
- Entries: `AndroidManifest.xml`, `LICENSE`, and
  `jni/x86/libLiteRt.so`.
- It deliberately has no `classes.jar`; Java and Kotlin APIs continue to come
  from `com.google.ai.edge.litert:litert:2.1.5`.

The test app consumes both artifacts:

```kotlin
implementation("com.google.ai.edge.litert:litert:2.1.5")
implementation(files("libs/litert-2.1.5-x86.aar"))
```

AGP produced an APK containing `lib/x86/libLiteRt.so`, and Android selected
`primaryCpuAbi=x86` at install time. The install used `adb install
--no-streaming`; the API 26 emulator's streamed installer timed out while the
push installer completed normally.

## Benchmark conditions

- Device: `Android SDK built for x86`, API 26, pure `x86` system image.
- AVD memory: 2 GB.
- Host exposed to the AVD: AMD Ryzen AI 9 HX 370.
- Runtime: LiteRT 2.1.5 CPU accelerator with XNNPACK.
- Models: static batch-1 float32 TFLite conversions.
- Input: the same first Coast Town MDX window used by the earlier first-batch
  report.
- Threads: 8.
- Timing: 1 warmup followed by 3 measured inferences.
- Scope: model inference and output read only; no audio decode, STFT/ISTFT,
  cache IO, or playback.

Desktop ORT 1.26 generated the element-wise reference tensors from the same
ONNX models and NCHW inputs. The files were uploaded to the benchmark service,
which performed the existing float-by-float comparison on device.

## Results

| Model | Runtime | Median wall ms | Mean CPU ms | PSS delta MiB | SNR vs ORT dB | Max abs error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `uvr_mdxnet_3_9662` | TFLite 2.16.1 | 19,549.3 | 58,440 | 675.6 | 97.55 | 0.0000552 |
| `uvr_mdxnet_3_9662` | LiteRT 2.1.5 x86 | 1,334.8 | 5,088 | 504.5 | 97.60 | 0.0000595 |
| `uvr_mdxnet_kara` | TFLite 2.16.1 | 20,758.5 | 60,692 | 674.3 | 111.21 | 0.0000124 |
| `uvr_mdxnet_kara` | LiteRT 2.1.5 x86 | 1,335.3 | 5,233 | 496.4 | 110.73 | 0.0000173 |
| `uvr_mdxnet_inst_hq_4` | TFLite 2.16.1 | error | - | - | - | - |
| `uvr_mdxnet_inst_hq_4` | LiteRT 2.1.5 x86 | error | - | - | - | - |

For the two completed models, LiteRT reduced median wall time by 93.2-93.6%,
total process CPU time by about 91%, and measured PSS growth by about 170-179
MiB. Both outputs contained only finite values, had the expected shape, and
had cosine similarity above `0.9999999999` against ORT.

`PSS delta` is a before/after process measurement, not isolated runtime memory.
The TFLite and LiteRT APKs also have different process contents, so the values
show scale rather than a clean library-only allocation difference.

## HQ4 failure

HQ4 delegated all 183 nodes to XNNPACK, then failed during the first invocation:

```text
XNNPack delegate failed to reshape runtime
Node number 183 (TfLiteXNNPackDelegate) failed to prepare.
Failed to allocate tensors
```

Repeating with one CPU thread failed at the same stage. This rules out an
eight-thread workspace-only issue. The 2560-bin HQ4 preset should remain
unsupported on this 2 GB pure x86 environment; a higher-level fallback would
not make it a practical playback target.

## Recommendation

The supplemental x86 AAR is technically viable and substantially better than
replacing LiteRT with TensorFlow Lite 2.16.1 solely for x86 support. Keep it
pinned to the exact LiteRT 2.1.5 Java API and publish it separately from model
weights. This producer role is now handled by the dedicated
`bss-litert-android` repository; the scripts here remain only for reproducing
this report.

For Booming SS, treat 32-bit x86 as emulator compatibility rather than a
performance target. Include 9662 and KARA support, reject or clearly mark HQ4
as unavailable on x86, and retain x86_64 as the preferred emulator ABI. If APK
delivery is split by ABI, this library adds no installed native-code cost for
ARM users; a universal APK still carries its 2.86 MiB compressed payload.

## Raw data

- LiteRT 9662:
  `outputs/android-benchmark/emulator-5554/x86-litert215-uvr_mdxnet_3_9662-cpu-snr/report.json`
- LiteRT KARA:
  `outputs/android-benchmark/emulator-5554/x86-litert215-uvr_mdxnet_kara-cpu-snr/report.json`
- LiteRT HQ4:
  `outputs/android-benchmark/emulator-5554/x86-litert215-uvr_mdxnet_inst_hq_4-cpu/report.json`
- TFLite 2.16.1 baselines:
  `../MusicPlayer/tflite216-benchmark/results/x86/`
