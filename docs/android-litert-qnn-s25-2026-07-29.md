# LiteRT Qualcomm HTP v79 experiment on Galaxy S25

Date: 2026-07-29 (host local time; device logs crossed into 2026-07-30 UTC)

Status: research result, not a Booming SS product-support decision

## Decision summary

Qualcomm HTP is technically promising for Booming SS on the tested Galaxy S25.
It ran the complete 9662 graph, was faster than CPU and GPU at steady state,
did not reproduce the severe foreground UI contention of the GPU paths, and
produced a complete 273.7-second song whose stems remained close to the ORT
reference.

It should not yet become a normal application backend:

- the JIT graph setup costs about 8 seconds for every new session;
- the nine Qualcomm JIT libraries add 101.7 MiB of extracted native code and
  43.5 MiB of compressed APK data with the required legacy JNI packaging;
- output is not FP32-equivalent, and no human ABX/listening qualification has
  been completed;
- this experiment covers one SM8750 device and one HTP v79 runtime only;
- Android's battery API and one BAT thermal channel reported an implausible
  4.7-8.0 C while another BAT channel stayed near 35.9 C, so the conflicting
  thermal and battery observations are not suitable for power conclusions;
- QAIRT redistribution and per-SoC runtime maintenance remain product costs.

The strongest reason to continue is UI isolation, not cold-session throughput.
Using the same model-only measurements, QNN's setup plus 48 windows is only
0.32 seconds lower than the bounded N=1 GPU estimate. A reusable or precompiled
QNN context could change that result substantially.

## Frozen configuration

Device:

| Property | Value |
| --- | --- |
| Product | Samsung Galaxy S25, SM-S9310 (`pa1q`) |
| SoC | Qualcomm SM8750 |
| Android | API 35 |
| Process ABI | `arm64-v8a` |
| Display | 1080 x 2340, 120 Hz |

Runtime:

| Component | Value |
| --- | --- |
| LiteRT API/runtime | 2.1.5 |
| Bounded GPU comparison runtime | `2.1.5-bss.2` |
| Qualcomm AI Runtime | 2.44.0.260225 |
| Accelerator target | HTP v79, JIT |
| HTP performance mode | `SUSTAINED_HIGH_PERFORMANCE` |
| QNN optimization | `HTP_OPTIMIZE_FOR_INFERENCE` |
| Performance profiling | off |

Model:

| Property | Value |
| --- | --- |
| ID | `uvr_mdxnet_3_9662` |
| TFLite SHA-256 | `f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378` |
| Shape | NHWC `[1, 2048, 256, 4]`, float32 |
| DSP | 44.1 kHz, FFT 6144, hop 1024, periodic Hann |
| Output semantic | vocals |
| Output compensation | 1.035 |

The TFLite artifact and contract come from
[bss-tflite](https://github.com/WluhWluh/bss-tflite). The bounded GPU runtime
comes from
[bss-litert-android](https://github.com/WluhWluh/bss-litert-android/releases/tag/runtime-v2.1.5-bss.2).

## HTP residency

The runtime evidence establishes accelerator execution, not merely plugin
discovery:

- LiteRT selected 183 of 183 LiteRT ops into one QNN partition.
- The runtime loaded `libQnnHtp.so`, the v79 stub, and the v79 CDSP skeleton.
- `QnnGraph_finalize` completed and generated a 20,873,216-byte context binary.
- The full-song run recorded 48 `QnnGraph_execute started` and 48 successful
  completions.
- Detailed profiling in a separate proof run reported nonzero accelerator
  cycles. Profiling increased one window from about 120 ms to about 417 ms, so
  it was disabled for every performance result.

`9662` is a model name, not an operation count. The IR contains 277 QNN nodes
before finalization; neither that number nor the 183 LiteRT ops should be
described as the final number of HTP kernels.

## Tensor performance

The table uses the same real Coast Town tensor, one warmup, and the same S25.
Means include output readback. QNN setup includes JIT.

| Backend | Samples | Setup | Mean/window | SNR vs ORT |
| --- | ---: | ---: | ---: | ---: |
| ORT CPU, 8 threads | 3 | 32 ms | 912 ms | reference |
| LiteRT CPU, 4 threads | 3 | 214 ms | 978.04 ms | 97.71 dB |
| LiteRT stock GPU FP32 | 3 | 543 ms | 244.92 ms | 97.64 dB |
| LiteRT bounded GPU FP32, N=1 | 5 | 567 ms | 295.04 ms | 97.64 dB |
| LiteRT QNN HTP v79 | 3 | 8,573 ms | 121.54 ms | 35.04 dB |

Steady-state QNN was 8.05x faster than 4-thread LiteRT CPU, 2.02x faster than
stock FP32 GPU, and 2.43x faster than bounded FP32 GPU in this model-only test.
Those ratios exclude QNN's approximately 8-second JIT setup and must not be
presented as first-run or end-to-end speedups.

One hundred repeated real windows remained stable:

- mean: 123.25 ms;
- range: 115.60-133.70 ms;
- Android thermal status: always 0;
- peak PSS during JIT: about 698 MiB in that run;
- steady PSS after JIT: about 300-325 MiB.

This repeated one tensor. The full-song test below uses 48 different windows.

## Foreground UI

The activity stayed foreground while ADB issued twenty alternating 500 ms
swipes. The table uses Android's complete `gfxinfo` aggregate, not the
truncated framestats ring-buffer tail.

| Concurrent work | Rendered frames | Modern jank | P50 | P99 |
| --- | ---: | ---: | ---: | ---: |
| Idle | 909 | 11 / 1.21% | 5 ms | 9 ms |
| QNN NPU | 906 | 8 / 0.88% | 5 ms | 8 ms |
| LiteRT CPU | 902 | 5 / 0.55% | 5 ms | 6 ms |
| Bounded GPU N=1 | 873 | 71 / 8.13% | 6 ms | 32 ms |
| Stock GPU, no explicit queue cap | 195 | 50 / 25.64% | 200 ms | 200 ms |

The CPU run heated from thermal status 0 to 3 and its inference time degraded,
so it is useful for UI non-contention only, not as a fair sustained performance
comparison. The QNN result strongly supports the hypothesis that HTP avoids the
shared GPU queue contention responsible for the visible GPU stalls on this
device.

## Numerical and audio output

QNN output is finite and stable, but it is not FP32-equivalent. Five selected
real windows were reconstructed with the frozen 9662 output scale and stem
semantics:

| Source position | Tensor SNR | Vocals waveform SNR | Instrumental SNR | Vocals max error |
| --- | ---: | ---: | ---: | ---: |
| 0.0 s | 35.06 dB | 22.25 dB | 61.09 dB | 0.000306 |
| 34.7 s | 54.97 dB | 63.08 dB | 64.78 dB | 0.000715 |
| 115.6 s | 55.30 dB | 62.16 dB | 69.66 dB | 0.000610 |
| 231.3 s | 48.03 dB | 49.35 dB | 59.11 dB | 0.002231 |
| Separate 37 s vocal fixture | 55.55 dB | 63.20 dB | 62.14 dB | 0.000578 |

The 0-second vocals reference is nearly silent (RMS 0.000434), so its relative
22.25 dB result must be read alongside the small absolute error and the 61.09
dB instrumental result. Averaging these SNR values would be misleading.

The Android full-song output was also compared sample-for-sample with the
desktop ORT reference:

| Stem | Frames | SNR vs ORT | Max absolute error | Cosine similarity |
| --- | ---: | ---: | ---: | ---: |
| Vocals | 12,070,130 | 57.18 dB | 0.003723 | 0.999999064 |
| Instrumental | 12,070,130 | 66.81 dB | 0.003693 | 0.999999896 |

All frame counts were exact. Across the 47 MDX generation boundaries, the
largest QNN-versus-ORT change in join step was 0.000946. Reconstruction error
remained essentially equal to the ORT reference. At the quantized PCM level,
the QNN instrumental contained 7,804 positive and 5,376 negative
maximum-magnitude samples, versus 7,803 and 5,371 for ORT. Vocals had none. The
pre-quantization QNN tracker counted 7,801 samples at or above 1.0 and 5,374 at
or below -1.0.

These metrics establish close backend parity, not separation quality against
ground-truth stems and not inaudibility. The ignored local result directories
contain ORT/QNN pairs and 20x difference WAV files for a later human listening
or ABX pass.

## Full-song result

The research-only Android path validated and decoded one canonical,
273.699-second PCM16 WAV, created one QNN session, processed 48 distinct MDX
windows, applied the 1.035 vocals scale, computed the instrumental residual,
and atomically published both complete PCM16 WAV files as one directory.

| Measurement | Result |
| --- | ---: |
| Processing wall time | 42.083 s |
| End-to-end time including output SHA-256 | 42.308 s |
| Realtime factor | 0.1546x |
| QNN setup/JIT | 7.939 s |
| HTP dispatch total | 5.673 s |
| HTP dispatch mean / P95 | 118.18 / 127.00 ms |
| Decode | 0.479 s |
| STFT | 6.947 s |
| ISTFT | 11.568 s |
| Input/output layout conversion | 0.786 s |
| PCM conversion, finite checks, and statistics | 7.743 s |
| WAV writes and atomic commits | 0.052 s |
| Output SHA-256 | 0.225 s |

After HTP acceleration, host DSP and output validation dominate steady-state
time. The PCM stage includes finite-value rejection, float peak and clipping
counts, and PCM16 quantization for both stems; it is not accelerator time. The
host sampler observed a 682,856 KiB peak PSS during JIT, while the service
reported 255,709 KiB after cleanup; graphics PSS stayed at 3,768 KiB. An
adjacent final repeat sampled 755,296 KiB at the JIT peak, so the two-second
host sampling cadence is suitable for a range, not an exact peak allocation.

Cold-session break-even is important. Using only the model-test setup and
per-window values in the tensor table, QNN overtakes bounded N=1 GPU after 46.1
windows and stock GPU after 65.1 windows. At 48 windows, QNN setup plus model
work is only 0.32 seconds lower than bounded GPU and remains 2.11 seconds slower
than stock GPU. A resident session would remove that penalty for later songs; a
disk-cached or AOT context could remove it from the first song as well.

## Lifecycle observations

- Force-stop triggered immediately after `QnnGraph_finalize started` removed
  the app PID and service in 151 ms. No false completion report remained.
- Force-stop after five HTP executions removed them in 167 ms, again without a
  false report.
- The final transactional-output test force-stopped 88.7 ms after observing
  the second HTP execution. The final output directory and completion report
  did not exist; only a sibling `<tag>.partial` directory containing two
  1,019,948-byte `.wav.partial` files remained.
- Immediate cold restart after each kill rebuilt the graph and completed at
  about 120-122 ms/window.
- A deliberate post-JIT shape-contract failure logged `QnnContext_free` and
  `contexts: 0, graphs: 0`; a valid run then completed in the same app PID.
- A 100-window foreground service continued through Dozing and completed. The
  device was Awake when completion was polled, so this proves progress while
  Dozing, not that the display remained off for the complete interval.

A production device does not expose enough FastRPC state to prove global CDSP
session counts after SIGKILL. The defensible force-stop result is that the app
PID/service disappeared, no DSP crash occurred, and subsequent cold sessions
repeatedly succeeded.

## Package and licensing cost

The pinned runtime contains nine arm64 files:

- raw JNI payload: 106,638,964 bytes (101.7 MiB);
- compressed in the APK: 45,613,057 bytes (43.5 MiB);
- extracted plus compressed installed increment: approximately 145.2 MiB;
- `libQnnHtpPrepare.so` alone: 85,539,184 bytes (81.58 MiB raw).

The complete debug test APK is 144,328,777 bytes, and `du` reported 280,151
KiB for its installed code directory because legacy JNI packaging retains both
the APK and extracted native libraries. This APK also contains ORT and research
code, so its total size is not a Booming SS estimate; the 145.2 MiB Qualcomm
increment is the relevant number.

QAIRT binaries and accepted-license state stay under ignored `.tmp/` storage.
They must not be committed or published as a standalone runtime bundle. Any
application distribution needs a fresh review of the pinned QAIRT license,
notices, SoC targeting, and object-code redistribution terms.

## Remaining gates

1. Repeat the full-song, sustained, and UI tests at normal ambient temperature.
2. Perform level-matched listening or ABX checks on complete vocals,
   instrumental, quiet-vocals, and amplified-difference excerpts.
3. Measure energy with an external power monitor or a device exposing a valid
   energy counter. Current battery current/level samples are too coarse.
4. Test additional Snapdragon generations with matching HTP runtimes. Do not
   generalize the v79 result to all Qualcomm devices.
5. Investigate persisted/AOT QNN context loading. Omitting the 81.58 MiB prepare
   library and avoiding 8-second JIT are the two changes most likely to alter
   the product decision, but context binaries may be SoC and QAIRT-version
   specific.
6. Repeat cancellation, background, process-isolation, and player interaction
   tests through the real Booming SS worker before considering product use.
7. Compare an optional Qualcomm-specific GitHub APK or feature delivery model
   against the maintenance and download cost. Bundling this runtime in every
   arm64 build is not justified by the current evidence.

## Reproduction

Prepare the pinned runtime only after reviewing and accepting its license:

```powershell
python -m pip install -r requirements-qnn.txt
python tools/prepare_litert_qnn_v79_runtime.py --accept-qairt-license
```

Build with the audited bounded LiteRT core when GPU comparisons are required:

```powershell
$revision = (git rev-parse HEAD).Trim()
$runtimeAar = "<path-to-litert-android-2.1.5-bss.2.aar>"
.\gradlew.bat :app:assembleQnnV79Debug `
  "-PliteRtAar=$runtimeAar" `
  "-PbenchmarkSourceRevision=$revision" `
  -PbenchmarkSourceDirty=false `
  -PbenchmarkRuntimeId=litert-android-2.1.5-bss.2
```

Run a tensor test with `run_android_inference_benchmark.ps1`. Add
`-ExportOutputTensor` to retain an exact NCHW output for reconstruction, and
use `evaluate_mdx_tensor_window.py` to generate or compare audio windows.

```powershell
$tensorBenchmark = @{
  Serial = "<adb-serial>"
  ModelId = "uvr_mdxnet_3_9662"
  ContractFile = "app/src/main/assets/benchmark-contracts/uvr_mdxnet_3_9662.json"
  OnnxModel = "models/uvr-mdx-candidates/trvlvr-all-public-uvr-models/UVR_MDXNET_3_9662.onnx"
  LiteRtModel = "<path-to-UVR_MDXNET_3_9662_static_float32.tflite>"
  InputFile = "<path-to-frozen-input-nchw-f32.bin>"
  AppApk = "app/build/outputs/apk/qnnV79/debug/app-qnnV79-debug.apk"
  RuntimeArtifact = $runtimeAar
  AcceleratorBundleManifest = ".tmp/litert-qnn-v79-runtime/runtime-manifest.json"
  Iterations = 20
  Warmups = 2
  Threads = 4
  KeepActivityForeground = $true
}

.\tools\run_android_inference_benchmark.ps1 @tensorBenchmark `
  -Backend ort -UploadModels -Tag s25-9662-ort-reference
.\tools\run_android_inference_benchmark.ps1 @tensorBenchmark `
  -Backend litert_qnn -Tag s25-9662-qnn-v79
```

Run the full-song path:

```powershell
.\tools\run_android_qnn_audio_benchmark.ps1 `
  -Serial <adb-serial> `
  -SourceAudio <path-to-canonical-44.1-kHz-stereo-pcm16-wav> `
  -ContractFile app/src/main/assets/benchmark-contracts/uvr_mdxnet_3_9662.json `
  -OnnxModel models/uvr-mdx-candidates/trvlvr-all-public-uvr-models/UVR_MDXNET_3_9662.onnx `
  -LiteRtModel <path-to-UVR_MDXNET_3_9662_static_float32.tflite> `
  -AppApk app/build/outputs/apk/qnnV79/debug/app-qnnV79-debug.apk `
  -RuntimeArtifact $runtimeAar `
  -AcceleratorBundleManifest .tmp/litert-qnn-v79-runtime/runtime-manifest.json `
  -Tag s25-qnn-full-song
```

This research path deliberately rejects compressed audio, noncanonical WAV
chunk layouts, non-PCM16 data, sample rates other than 44.1 kHz, non-stereo
input, and PCM payloads larger than 128 MiB before it creates a LiteRT session.
When `-ReuseDeviceFiles` is used, the host and device SHA-256 values of both the
model and source audio must match.

Raw models, QAIRT files, input audio, output stems, and benchmark captures stay
ignored. Stable conclusions belong in this document.
