# S25 BandBuddy HTDemucs 6-stem LiteRT/QNN reproduction

Date: 2026-08-03 (host local time)

Status: neural-core CPU and QNN HTP mixed execution reproduced; full-graph QNN
GPU FP16/FP32 failed during graph preparation; throughput-only candidate;
end-to-end DSP not reproduced

This experiment evaluates a third-party HTDemucs neural-core FlatBuffer on the
S25. It does not establish product support, full-song stability, separation
quality, or conformance to this repository's 80 dB FP32 gate.

## Outcome

The fixed BandBuddy FlatBuffer executed on the S25 with both four-thread
XNNPACK FP32 and BandBuddy's exact QNN 2.48 mixed graph. The QNN log proves that
11 of 3,504 operators were delegated as seven HTP V79 partitions and that all
seven partitions executed successfully. The other supported operators ran
through XNNPACK FP32.

For one deterministic non-zero 7.8-second neural-core window, cached QNN took
4.997 seconds of native inference versus 6.090 seconds on CPU, a 17.9% model-
only reduction. QNN-to-CPU output SNR was 66.78 dB over 20,642,832 values. That
is useful low-precision performance evidence, but it fails this repository's
80 dB numerical gate and remains throughput-only.

The S25 also reproduced the previously reported full-graph QNN GPU failure.
Both GPU precision modes selected 3,490 of 3,504 nodes in six partitions, so
this was not a delegate-discovery failure or silent CPU fallback. FP16 failed
while finalizing the third partition: Adreno GPU allocation returned
`errno 12`, OpenCL program compilation failed, and QNN surfaced
`qnn_graph_finalize failed. Error 6022`. FP32 reached the same third-partition
finalize stage but exhausted native addressable memory in the Qualcomm OpenCL
compiler and aborted the process with `SIGABRT`. Neither mode reached
inference or produced output.

BandBuddy's native STFT/iSTFT test could not run because the public source has
a JNI package mismatch: Kotlin uses `cn.lonelyme...`, while
`native_demucs.cpp` exports `Java_com_lonelyme...`. Neural-core execution is
therefore reproduced; the published app's complete DSP boundary is not.
After a temporary symbol-only repair in the isolated checkout, the JNI loaded
but the existing STFT calibration still failed (`component=2, bin=163,
frame=86`: expected `1.7259982`, actual `-0.08726111`). This confirms a second
DSP numerical blocker rather than merely a naming problem.

## Frozen identities

| Item | Identity |
| --- | --- |
| Device | Samsung `SM-S9310`, `SM8750`, Android 15/API 35, arm64-v8a |
| ADB serial | `192.168.8.197:5555` |
| BandBuddy source | `dourgey/BandBuddy-Android@010d28948607e2d9c62b56ab1038681359869736` |
| ModelScope upload revision | `c879fc738ec248cbfe7f01ce20b83885e1747e2c` (`v1.0.0`) |
| Model | `htdemucs_6s.core.tflite`, 117,784,760 bytes |
| Model SHA-256 | `a9fcc89e84aa65313e0540b582e710007ed12064969a0d49a3c85e49f1ae4e3d` |
| schema v4 contract | 5,022 bytes, SHA-256 `ce0dfb4481eeb97a5d8dc32554ce90bcd6d2786bfbd63766909f836794288e09` |
| Upstream app APK | 108,503,484 bytes, SHA-256 `3337278230ac39c494dce14d4d60c1585f57d342b288843c4a3b5c965948b53e` |
| Upstream test APK | 2,536,795 bytes, SHA-256 `6596cf302d38d8867d99429ff4e872ee0da79fd76d93b6aef5dc4504fd35d945` |
| GPU probe test APK | 2,656,964 bytes, SHA-256 `0db66d533e988fd84889881da90cdd4583a02e6ad187bbdb4960f6d39a0c486d` |

Those APK identities are the binaries used for the reported upstream and
deterministic runs. After the temporary JNI diagnostic was reverted, a fresh
source-restored debug APK was installed on the device (`108,659,546` bytes,
SHA-256 `f56cc029b383803419ca8106004c915fb1ad719279226013d06fa190bf70c893`);
that cleanup rebuild is not mixed into the CPU/HTP timing table. The GPU rows
below used this source-restored app APK and the separately identified GPU
probe test APK.

The fixed download URL is:

```text
https://www.modelscope.cn/models/Zzzzzzorz/BandBuddy-HTDemucs-6s/resolve/v1.0.0/htdemucs_6s.core.tflite
```

BandBuddy declares `org.tensorflow:tensorflow-lite:2.17.0`, but that coordinate
is a relocation POM. In this build it resolves to
`com.google.ai.edge.litert:litert:1.0.1`, not this repository's LiteRT 2.1.5.
The isolated APK is therefore required for a faithful reproduction.

| Runtime artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `litert:1.0.1` AAR | 6,487,444 | `467679820f836fe70418f01c9f6689f328f79fbf2cbabe197eab70b209b43e8d` |
| `qnn-runtime:2.48.0` AAR | 68,393,877 | `ef47797c557e124e18eeea32c8cd2e346233d2d28a69f85e22da9fb0b639de06` |
| `qnn-litert-delegate:2.48.0` AAR | 659,678 | `b8bd8bada3a11c36add5768804a49234bbeb0253ce50b9a603dfce23e6061acb` |

The Qualcomm artifacts use the Qualcomm AI Hub Model License. They are not
copied into this repository.

## FlatBuffer contract

Static inspection found one TFL3 subgraph, 3,504 operators, 4,332 tensors, 27
builtin operator kinds, no custom/Flex operators, no quantization, and no
dynamic I/O dimensions. Combined I/O storage is 96,333,216 bytes (91.87 MiB).

| Index | Runtime name | Type and shape |
| ---: | --- | --- |
| input 0 | `serving_default_args_0` | float32 `[1,2,343980]` |
| input 1 | `serving_default_args_1` | float32 `[1,4,2048,336]` |
| output 0 | `serving_default_output_0_output` | float32 `[1,6,4,2048,336]` |
| output 1 | `serving_default_output_1_output` | float32 `[1,6,2,343980]` |

The signature is `serving_default`, binding `args_0`, `args_1`, `output_0`, and
`output_1`. The four-element feature axis is
`left-real,left-imag,right-real,right-imag`. Stem semantics are not stored in
the FlatBuffer and must remain frozen from the official model metadata:

```text
drums, bass, other, vocals, guitar, piano
```

## Delegate contract

The delegate receives the 3,493 CPU node IDs through `setSkipNodeIds(int[])`.
Only these fixed-model node IDs remain eligible for HTP:

```text
586, 867,
2639, 2645, 2777, 2784, 2923, 2929,
3157, 3164, 3303
```

They are six `CONV_2D` and five `TRANSPOSE_CONV` operators. The options are HTP
backend, FP16, HMX enabled, BURST performance, optimize-for-prepare, basic
profiling, and info logging. XNNPACK uses four CPU threads. The node IDs are
valid only for the frozen SHA; any conversion or graph rewrite requires a new
mapping.

The decisive runtime evidence was:

```text
TfLiteQnnDelegate delegate: 11 nodes delegated out of 3504 nodes with 7 partitions.
Replacing 3145 out of 3500 node(s) with delegate (TfLiteXNNPackDelegate) node.
Graph qnn_delegate_graph_0 execution finished with result 0
...
Graph qnn_delegate_graph_6 execution finished with result 0
```

## Full-graph GPU probe

The isolated `gpuCandidateSingleWindowProbe` uses the verified FlatBuffer and
the same bounded deterministic inputs as the CPU/HTP comparison. It bypasses
the known JNI/STFT blockers. Each candidate deletes its own QNN cache, selects
the QNN GPU backend in high-performance mode, and leaves all 3,504 TFLite node
IDs eligible. XNNPACK is configured for the unsupported remainder, but no
inference begins if GPU graph preparation fails.

The device exposed `/vendor/lib64/libOpenCL.so`; QNN reported Qualcomm OpenCL
3.0 build `0800.35`, compiler `E031.47.18.28`. Thermal status was 0 before and
after both short probes.

| Candidate | Actual QNN selection | Prepare outcome | Test duration | Inference/output |
| --- | --- | --- | ---: | --- |
| GPU FP16 | 3,490 / 3,504 nodes, 6 partitions | FAIL: OpenCL program build OOM, QNN Error 6022 | 19.3 s | not reached / none |
| GPU FP32 | 3,490 / 3,504 nodes, 6 partitions | FAIL: native allocator OOM, `SIGABRT` | about 15 s process uptime | not reached / none |

FP16 finalized two partitions (1.231 s and 9.041 s host time). Finalizing the
third produced the decisive sequence:

```text
Adreno-GSL: sharedmem_gpumem_alloc: mmap failed errno 12 Out of memory
Adreno-GSL: kgsl_sharedmem_alloc() failed
[Qnn] Could not build cl::Program (-6) for legacy_buffer_hwc_to_image2darray_chw
[Qnn] GPU_ERROR_OPENCL(10014) - OpenCL program build error
Failed to apply delegate: qnn_graph_finalize failed. Error 6022
```

FP32 finalized two partitions (0.884 s and 7.378 s host time), then aborted
during the third. No `failure.txt` could be written because the native process
died before Kotlin regained control. Tombstone `tombstone_28` records:

```text
signal 6 (SIGABRT)
Abort message: 'Scudo ERROR: internal map failure (error desc=Out of memory)'
libllvm-qcom.so: generateExecutableFromBinary
libOpenCL_adreno.so: qCLDrvAPI_clCreateProgramWithBinary
libQnnGpu.so
libQnnTFLiteDelegate.so
```

This reproduces the material full-graph GPU failure on SM8750. It does not
show that every smaller GPU partition is unusable; broad or convolution-only
node subsets were deliberately not run after the requested full candidate had
already failed and FP32 had terminated the process.

## Single-window performance

All runs used one 343,980-sample window. No full song or sustained loop was
run. `native` is `Interpreter.lastNativeInferenceDurationNanoseconds`; total is
measured from before Interpreter construction through inference. RTF divides
native inference by 7.8 seconds.

| Input / backend | Cache | Total | Native | RTF | Result |
| --- | --- | ---: | ---: | ---: | --- |
| zero / QNN mixed | cold save | 16.074 s | 5.021 s | 0.644 | pass, exact upstream test |
| zero / QNN mixed | restore | 5.542 s | 5.043 s | 0.647 | pass, exact upstream test |
| zero / CPU | n/a | 6.588 s | 6.248 s | 0.801 | pass, local CPU extension |
| deterministic / CPU | n/a | 6.392 s | 6.090 s | 0.781 | pass, full outputs exported |
| deterministic / QNN mixed | restore | 5.438 s | 4.997 s | 0.641 | pass, full outputs exported |

The QNN compiled cache was 3,998,464 bytes. Cold graph preparation dominated
the first run. QNN's own finalize log estimated 175,841,792 bytes for one
finalized graph allocation; this is not a process peak-PSS measurement.

An independent temporary symbol-probe rebuild repeated the same deterministic
fixture at 4.928 s QNN native and 6.062 s CPU native. These repeated values
support the observed 4.93--5.00 s QNN and 6.06--6.09 s CPU range, but are not
presented as a statistical multi-run benchmark.

## Numerical comparison

The deterministic extension fills both core inputs with bounded, non-zero,
index-derived float32 values: `mix[i] = (((17*i) mod 257)-128)/512` and
`spec[i] = (((29*i) mod 251)-125)/512`. It is useful for CPU/QNN tensor parity
but is not a valid audio/STFT quality fixture. CPU XNNPACK is the comparison
reference.

| Output | Elements | SNR | Relative L2 | RMSE | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| spectral branch | 16,515,072 | 66.786 dB | 0.0004578 | 2.697e-5 | 0.0015613 |
| time branch | 4,127,760 | 59.654 dB | 0.0010406 | 1.427e-6 | 4.821e-5 |
| combined | 20,642,832 | 66.783 dB | 0.0004580 | 2.413e-5 | n/a |

Every CPU and QNN value was finite. The raw tensor hashes and complete metrics
are under the ignored evidence directory listed below.

## Classification

| Gate | Classification | Reason |
| --- | --- | --- |
| Static model identity/ABI | PASS | exact bytes/SHA, static two-input/two-output graph |
| S25 CPU single-window allocation/execution | PROVISIONAL PASS | finite output and measured execution; peak PSS not captured |
| S25 QNN delegation/execution | PASS | 11 nodes, seven real HTP V79 graphs, seven successful executions |
| S25 QNN GPU full-graph prepare | FAIL | FP16 Error 6022 after OpenCL OOM; FP32 native OOM/SIGABRT |
| Numerical/product gate | FAIL | 66.78 dB versus required 80 dB; no official PyTorch parity |
| End-to-end host DSP | BLOCKED | JNI symbol mismatch plus stable STFT calibration mismatch |
| Sustained/full-song stability | NOT RUN | outside this short reproduction |

FastRPC printed permission and service-death messages during delegate teardown,
after all seven graphs had returned result 0 and the instrumentation test had
passed. Repeated short runs succeeded, but these messages must be investigated
before any sustained claim. Process peak PSS was not captured reliably; only
the QNN allocation estimate is retained. Thermal status was 0 before and after
the short runs.

The temporary JNI-symbol repair was reverted in the third-party checkout. With
the repair applied, the native boundary loaded and produced finite core output,
but `nativeStftMatchesPyTorchAndReconstructsTheWindow` stopped at the first
calibration mismatch above. The exported finite tensors are consequently
neural-core evidence only, not an end-to-end audio quality result.

## Reproduction

Download and inspect the model:

```powershell
curl.exe -L --fail --retry 3 `
  --output .tmp\bandbuddy-htdemucs-6s\htdemucs_6s.core.tflite `
  https://www.modelscope.cn/models/Zzzzzzorz/BandBuddy-HTDemucs-6s/resolve/v1.0.0/htdemucs_6s.core.tflite

.\.venv\Scripts\python.exe tools\inspect_tflite_model.py `
  .tmp\bandbuddy-htdemucs-6s\htdemucs_6s.core.tflite `
  --contract app\src\main\assets\benchmark-contracts\bandbuddy_htdemucs_6s_core_v1_0_0.json
```

Build the isolated reference app at the frozen commit, using the local Android
SDK. Its `packaging.jniLibs.useLegacyPackaging=true` setting is required so the
CDSP daemon can map the HTP skeleton.

```powershell
git clone https://github.com/dourgey/BandBuddy-Android.git .tmp\BandBuddy-Android
git -C .tmp\BandBuddy-Android checkout 010d28948607e2d9c62b56ab1038681359869736
.\.tmp\BandBuddy-Android\gradlew.bat -p .tmp\BandBuddy-Android `
  :app:assembleDebug :app:assembleDebugAndroidTest --no-daemon
```

The exact upstream zero-input QNN test is
`originalWindowCandidateRunsThroughVerifiedMixedNpuPartition`. The local CPU
and deterministic/GPU probe extension is preserved as
`tools/patches/bandbuddy_s25_deterministic_core.patch`; apply it only to the
frozen third-party checkout, never to a different model or source revision.

Run the two cold-cache full-graph GPU probes independently:

```powershell
adb -s 192.168.8.197:5555 shell am instrument -w -r `
  -e gpuCandidate gpu-fp16 `
  -e class 'cn.lonelyme.bandbuddy.DemucsInferenceInstrumentedTest#gpuCandidateSingleWindowProbe' `
  cn.lonelyme.bandbuddy.test/androidx.test.runner.AndroidJUnitRunner

adb -s 192.168.8.197:5555 shell am instrument -w -r `
  -e gpuCandidate gpu-fp32 `
  -e class 'cn.lonelyme.bandbuddy.DemucsInferenceInstrumentedTest#gpuCandidateSingleWindowProbe' `
  cn.lonelyme.bandbuddy.test/androidx.test.runner.AndroidJUnitRunner
```

Generated evidence is ignored under:

```text
outputs/bandbuddy-s25-20260803/
```

Important files are `qnn-logcat.txt`, `qnn-warm-logcat.txt`,
`cpu-deterministic-logcat.txt`, `qnn-deterministic-logcat.txt`,
`deterministic-comparison.json`, the four exported raw tensors, and
`gpu/{gpu-fp16-logcat.txt,gpu-fp16-instrumentation.txt,gpu-fp16-failure.txt,
gpu-fp32-logcat.txt,gpu-fp32-instrumentation.txt}`.

## Next batch

1. Fix the `cn`/`com` JNI exports, then fix the exposed native STFT mismatch
   against pinned PyTorch vectors before calling this end to end.
2. Generate an official-safetensors PyTorch neural-core reference for the same
   real `mix + spec` window. Keep the 80 dB gate; do not promote the current
   66.78 dB candidate.
3. Move PSS, RSS, swap, thermal, and watchdog sampling into the probe process so
   the short instrumentation lifetime cannot evade host polling.
4. Only then run one warmup plus three measured CPU/QNN windows. Report median
   and maximum, not a fabricated P95.
5. Treat a LiteRT 2.1.5 `CompiledModel` conversion as a separate candidate; it
   cannot inherit this QNN 2.48 result or the fixed node IDs.
6. Do not retry the full GPU graph as a performance benchmark. If GPU
   diagnostics continue, bisect partition/node subsets with a hard memory
   watchdog and classify them as feasibility-only experiments.
