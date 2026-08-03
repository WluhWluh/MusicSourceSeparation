# Android LiteRT Batch 0 calibration: UVR MDXNET 3 9662

Date: 2026-08-03

Status: completed research calibration, not a production backend decision

## Decision summary

Batch 0 establishes a reproducible model contract and a device baseline for
`uvr_mdxnet_3_9662`. It does not rank or qualify any other separation model.
Only the 29 completed `b0screen-*` reports are valid performance evidence.

| Device | ORT reference | LiteRT CPU | LiteRT bounded GPU FP32 | LiteRT QNN NPU | Full song |
| --- | --- | --- | --- | --- | --- |
| Galaxy S10, SM8150 | completed | completed | completed | unavailable in the current QNN runtime matrix | not run |
| Galaxy S25, SM8750 | completed | completed | completed | completed, delegation verified | QNN completed, 48/48 windows |

The bounded GPU path is the fastest measured LiteRT tensor path on the S10. On
the S25, QNN is the fastest measured steady-state tensor path at about 122 ms
per model window, followed by the bounded GPU at about 304 ms in the 100-window
run. QNN also has an approximately 8.6-second JIT setup cost and a materially
larger tensor error than CPU or GPU, so steady-state speed alone is not an
application-selection rule.

Using the median cold-session setup and inference values, QNN overtakes the
bounded GPU after about 50.8 model windows. This is approximately 294 seconds
of generated audio at 44.1 kHz. Because Batch 0 has no matching GPU full-song,
UI, energy, or release-build result, this defines a model-only routing
hypothesis for the next batch rather than an application policy:

- use the bounded GPU for short, one-shot work on the S25;
- consider QNN for audio longer than about five minutes or when a session can
  be reused; and
- do not enable QNN in an application until full-song SI-SDR and controlled
  listening gates are complete.

## Scope and result validity

This batch tested one model and one frozen real input tensor:

- model: `uvr_mdxnet_3_9662` only;
- contract: `uvr_mdxnet_3_9662@2` only;
- devices: one Galaxy S10 and one Galaxy S25;
- tensor runs: ORT, LiteRT CPU, bounded LiteRT GPU FP32, and QNN where the
  current runtime supports the device;
- audio run: one full-song QNN run on the S25 only.

Every accepted run kept the benchmark Activity foreground and the screen on.
All 29 reports have `status=complete` and share the frozen contract, model,
runtime artifact, and source revision
`2923b73c2b74a541e37d4c1fa113b9186fbf2cb9`. Each report separately matched
its expected APK/flavor, optional accelerator bundle, and tensor or audio input
identity; all record `sourceDirty=false`.

The earlier `b0final-*` runs are excluded from all rankings. Their host Activity
did not hold `FLAG_KEEP_SCREEN_ON`; after the normal display timeout, Android
changed the foreground UID scheduling conditions. The clearest symptom was the
S25 LiteRT CPU 100-window run: its first-ten mean was 976.69 ms, but its
last-ten mean reached 9178.45 ms. This was a screen-timeout scheduling artifact,
not a valid model thermal result. After commit `2923b73`, the equivalent
`b0screen-*` rerun measured 958.50 ms and 1229.35 ms respectively. The excluded
data remains useful only as framework diagnostics.

## Frozen identity

### Source and runtime

| Property | Frozen value |
| --- | --- |
| Branch | `experiment/model-matrix-batch0-9662` |
| Benchmark source revision | `2923b73c2b74a541e37d4c1fa113b9186fbf2cb9` |
| Source dirty state | `false` |
| Build type | Android debug benchmark APKs |
| LiteRT runtime | 2.1.5, bounded GPU artifact `2.1.5-bss.2` |
| LiteRT AAR SHA-256 | `88cd2f7eaf1443d1c570085b1c24f239db87eb24c788a590adf5158e17443d0e` |
| QNN bundle manifest SHA-256 | `8e20d10a6b27107fcb80460cc27892d5c2b797107c09ecbebeb095fbfe09c297` |
| Standard APK | 139,213,018 bytes; `e011ff6c90c33dab323c3c875e24f5fd40c501fae5d6c5929e6affb96eb96568` |
| QNN v79 APK | 240,376,374 bytes; `59ac8503838582deeae18d1c4f097a2414a88bac4c31ad074e9d6ece2984bda8` |

The runners hashed the installed `base.apk` before every accepted run. The S10
installed APK matched the frozen standard APK; the S25 installed APK matched
the frozen QNN v79 APK.

### Model and inputs

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Contract sidecar | 2,273 | `edf02de52bb45c842ad65a4be8f2118ed6d212ec4590a81ae0401e760bfb4fe9` |
| Source ONNX | 29,704,436 | `e02220e80d8253f4c2209f8924298b2b686bbdf2868b788ff5500fb9bd94aadc` |
| LiteRT TFLite | 29,700,464 | `f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378` |
| Tensor fixture, NCHW float32 LE | 8,388,608 | `831d084ae1ca69cc3f3103c965928b086f5c22dd5b45c0b19bc78c44fec66355` |
| Full-song PCM16 WAV | 48,280,564 | `e845e52aeeeb69be702d3a28d756eaf7a3137dbe5338ea50fb0ac3d8c4f9bd89` |

The sidecar records conversion repository revision
`695cf49db9bbe0d43ef0c8b52ae145a16c10cf28` from `bss-tflite`. The benchmark
runner rejects an ONNX or TFLite file whose name, size, or SHA-256 does not
match this sidecar.

## Device identity

| Property | Galaxy S10 | Galaxy S25 |
| --- | --- | --- |
| Model | SM-G9730 (`beyond1q`) | SM-S9310 (`pa1q`) |
| SoC | QTI SM8150 | QTI SM8750 |
| Android | 12, API 31 | 15, API 35 |
| Security patch | 2023-03-01 | 2025-08-01 |
| Process ABI | `arm64-v8a` | `arm64-v8a` |
| Build fingerprint | `samsung/beyond1qltezc/beyond1q:12/SP1A.210812.016/G9730ZCU8HWE2:user/release-keys` | `samsung/pa1qzcx/pa1q:15/AP3A.240905.015.A2/S9310ZCU5AYHA_CHC5AYHA:user/release-keys` |

The S10 standard runtime exposed `CPU,GPU`. There is no SM8150-compatible pack
in the current v69/v73/v75/v79 QNN runtime matrix, and no S10 `b0screen` QNN
report was produced. This result is `unsupported by the current path`, not a
claim that the hardware can never execute an NPU workload and not a measured
CPU fallback.

## Model contract

The versioned sidecar, rather than the historical generic MDX notes, is the
authority for this batch.

| Contract property | Value |
| --- | --- |
| Contract ID / schema | `uvr_mdxnet_3_9662@2` / 2 |
| Runtime input | named float32 NHWC `[1, 2048, 256, 4]` |
| Runtime output | named float32 NHWC `[1, 2048, 256, 4]` |
| Host tensor fixture | float32 NCHW `[1, 4, 2048, 256]`, little endian |
| Sample rate / channels | 44,100 Hz / stereo |
| FFT / hop / window | 6,144 / 1,024 / periodic Hann |
| Frequency bins / time frames | 2,048 / 256 |
| Trim / generated frames | 3,072 / 254,976 |
| Generated audio per window | 5.78177 seconds |
| Model output | vocals |
| Output scale | 1.035 |
| Residual | instrumental = mixture - scaled model output |

The app loader also validates contract keys, tensor and DSP cross-fields,
pipeline compatibility, source and artifact identities, and stem semantics. It
fails closed on missing, unknown, or inconsistent fields.

## Method

Each cold tag starts with `am force-stop`, wakes and unlocks the device, then
launches the benchmark Activity with its benchmark-only screen-on flag.
The contract, source/artifact pair, and tensor fixture are identical across
backends on a device. ORT consumes the pinned ONNX; LiteRT consumes the pinned
TFLite. ORT runs first and writes the element-wise reference used by the
LiteRT paths.

The cold-session result for each backend consists of three independent
process-cold sessions. There was no device reboot, OS cache drop, driver cache
reset, or controlled cooldown between them, so setup is not a hardware-cold
measurement. Each session performs two warmups followed by 20 measured
iterations with four CPU threads configured for the CPU backends. Setup is
reported separately. LiteRT per-window inference includes dispatch and output
readback.
ORT `inferenceWallMs` covers `session.run`; its output copy is accumulated in
`outputReadWallMs` instead. The accepted cold ORT runs measured 5.4-8.8 ms per
iteration, so this asymmetry does not change the observed ranking but must be
included in a strictly symmetric comparison. Setup is also backend-defined:
the LiteRT timer includes model/buffer creation, NCHW-to-NHWC conversion, and
the initial input write, while the ORT timer ends after session creation.
Neither path's per-window timing includes STFT, ISTFT, audio decode, or WAV
output.

The sustained profile uses one new session, two warmups, and 100 measured
iterations of the same real tensor. `first 10` and `last 10` are arithmetic
means. This measures repeated-window stability; it is not a substitute for a
full-song run with changing windows. Peak PSS comes from periodic
`dumpsys meminfo` samples and can miss a shorter transient peak.

Backend acceptance is fail-closed:

- ORT, CPU, and GPU must complete with the expected shape and finite output;
- bounded GPU reports must identify profile `gpu-opencl-bounded-fp32-v1`, a
  queue window and kernel batch size of one, and nonzero dispatch/wait counts;
- QNN must pass device and library preflight, complete with finite output, and
  emit at least one non-empty Qualcomm IR partition; and
- an NPU label or provider-ready state without non-empty IR is not delegation.

## Cold-session tensor performance

The table aggregates three independent cold sessions. `P50 range` is the range
of the three session medians; the other timing columns are medians across the
three sessions.

| Device | Backend | Setup median (ms) | Inference P50 median (ms) | P50 range (ms) | P95 median (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| S10 | ORT | 100.40 | 3386.95 | 3283.30-3434.23 | 3550.59 |
| S10 | LiteRT CPU | 419.25 | 3100.14 | 3039.44-3193.29 | 3183.36 |
| S10 | LiteRT bounded GPU | 1141.85 | 2288.61 | 2287.29-2290.83 | 2300.05 |
| S25 | ORT | 32.08 | 1382.40 | 1366.53-1410.91 | 2163.94 |
| S25 | LiteRT CPU | 208.04 | 1097.47 | 1028.34-1542.84 | 1550.32 |
| S25 | LiteRT bounded GPU | 581.82 | 279.00 | 278.32-300.37 | 306.81 |
| S25 | LiteRT QNN v79 | 8558.13 | 122.12 | 121.14-123.07 | 124.77 |

The S25 QNN setup range was 8.438-11.078 seconds despite its stable inference
time. S25 ORT and CPU also show substantially more cold-session tail variance
than GPU or QNN. Setup and inference must remain separate in any application
estimate.

## Sustained 100-window performance

| Device | Backend | P50 (ms) | P95 (ms) | First 10 mean (ms) | Last 10 mean (ms) | Change | Peak PSS (MiB) | Thermal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S10 | ORT | 3292.45 | 3500.83 | 3018.57 | 3353.64 | +11.1% | 1096.8 | 0 -> 0 |
| S10 | LiteRT CPU | 3105.01 | 3263.75 | 2880.51 | 3142.55 | +9.1% | 788.7 | 0 -> 0 |
| S10 | LiteRT bounded GPU | 2280.41 | 2298.44 | 2278.75 | 2279.65 | +0.0% | 558.3 | 0 -> 0 |
| S25 | ORT | 1510.23 | 2347.78 | 1336.30 | 1619.17 | +21.2% | 1111.6 | 0 -> 1 |
| S25 | LiteRT CPU | 1098.19 | 1505.02 | 958.50 | 1229.35 | +28.3% | 799.8 | 0 -> 1 |
| S25 | LiteRT bounded GPU | 304.18 | 315.15 | 289.14 | 309.88 | +7.2% | 567.8 | 0 -> 0 |
| S25 | LiteRT QNN v79 | 121.88 | 127.76 | 122.18 | 123.33 | +0.9% | 659.3 | 0 -> 0 |

Android thermal status is coarse. The S25 CPU paths reached light or moderate
thermal status in some cold sessions as well as status 1 in the sustained run,
but these measurements do not isolate temperature from scheduler and power
policy effects. No backend power or energy conclusion is made.

### Model-only throughput

One window generates 254,976 frames, or 5.78177 seconds at 44.1 kHz. Dividing
that duration by the sustained median gives the following model-only rate. It
excludes setup, DSP, decode, layout conversion outside the timed invocation,
and file I/O.

| Device | Backend | Audio seconds / inference second |
| --- | --- | ---: |
| S10 | ORT | 1.76x |
| S10 | LiteRT CPU | 1.86x |
| S10 | LiteRT bounded GPU | 2.54x |
| S25 | ORT | 3.83x |
| S25 | LiteRT CPU | 5.26x |
| S25 | LiteRT bounded GPU | 19.01x |
| S25 | LiteRT QNN v79 | 47.44x |

## Numerical consistency

All accepted tensor outputs have the expected 2,097,152 float elements and zero
non-finite values. CPU and bounded GPU are very close to the device-side ORT
reference. QNN is directionally similar but not FP32-equivalent.

| Device | Backend | Max absolute error | RMSE | SNR vs ORT | Cosine similarity |
| --- | --- | ---: | ---: | ---: | ---: |
| S10 | LiteRT CPU | 5.6267e-5 | 4.0025e-6 | 97.8008 dB | 0.999999999917 |
| S10 | LiteRT bounded GPU | 6.2793e-5 | 4.0181e-6 | 97.7670 dB | 0.999999999916 |
| S25 | LiteRT CPU | 5.4076e-5 | 4.0024e-6 | 97.8010 dB | 0.999999999916 |
| S25 | LiteRT bounded GPU | 5.7697e-5 | 4.0644e-6 | 97.6674 dB | 0.999999999914 |
| S25 | LiteRT QNN v79 | 0.1023027 | 0.0054882 | 35.0587 dB | 0.999848946530 |

The high QNN cosine similarity does not cancel its approximately 62.6 dB SNR
gap from the CPU/GPU paths or its 0.1023 maximum error. Batch 0 therefore marks
QNN execution as technically successful but leaves audio quality unqualified.

## Accelerator execution evidence

### Bounded GPU

Every accepted GPU report identifies the audited FP32 profile with
`kernelBatchSize=1` and `commandQueueWindowSize=1`. Each 20-iteration cold run
records 2,794 dispatches and 2,794 event waits; each 100-iteration run records
12,954 of each. The identical nonzero counts on both devices establish that the
GPU path was exercised rather than silently replaced with CPU execution.

### Qualcomm HTP v79

All three S25 cold sessions, the sustained session, and the full-song session
report `delegationStatus=delegated` and `delegationVerified=true`. Each emitted
one non-empty 396,223-byte QNN IR partition. The full-song IR SHA-256 is
`870607347e6b76592d57efa6ddb447b855c6724e14320771ed482f0c2450bbb5`.
IR JSON hashes can differ between JIT sessions, so the acceptance criterion is
verified non-empty output under the frozen runtime identity, not byte identity
between compilations.

The full-song log records 183 of 183 LiteRT operations selected into one
partition, a 20,873,216-byte QNN context, a loaded graph, and 48 successful
`QnnGraph_execute` calls. Profiling was off; HTP mode was
`SUSTAINED_HIGH_PERFORMANCE`, with `HTP_OPTIMIZE_FOR_INFERENCE`.

The report hashes the actual native libraries present at runtime:

| Native library | Bytes | SHA-256 |
| --- | ---: | --- |
| `libLiteRt.so` | 5,328,296 | `ae2b996fde27021b070e88b56eebc9626a5261feb72f09791bdac38b2f09abd2` |
| `libLiteRtClGlAccelerator.so` | 2,728,920 | `83f2be273fdc0391ad977c8889d65ba6b947c3ebabc432bf51829e5c9e91f93c` |
| `libLiteRtCompilerPlugin_Qualcomm.so` | 691,744 | `c7fe5ee3ac5b89b9e903989a90d1584158b3db889f6c85330495b338951f735d` |
| `libLiteRtDispatch_Qualcomm.so` | 462,528 | `f8ee14eb9cad99a1fc8522478d1db6712417dffeced287e180a0f84d17b6acfb` |
| `libQnnHtp.so` | 2,778,176 | `090e993822564851eab1405aff171643b21e644e3f696c95c96f2732aaed813a` |
| `libQnnHtpPrepare.so` | 85,539,184 | `09b1c15c62b6875af49ffd3d841961c098b85c367f584fee370f986c62511298` |
| `libQnnHtpV79Skel.so` | 10,975,268 | `41f83395ed4b1bcfc43417a1b82f3f137c825747711c9ab4c9d50034ed198f98` |
| `libQnnHtpV79Stub.so` | 679,168 | `005bd3de462851ce3dde55260d7d8560d6d07dbc309f554780b1f6412e6d9df1` |
| `libQnnIr.so` | 1,741,288 | `982d7e403eec3de800219bf8de7039e7fa020749618e0505f831fcecfd2bd85d` |
| `libQnnSaver.so` | 788,048 | `5dbe2eb7f17c217d035ce288b75b6cc9445df551739dba787ccb55097af48b0e` |
| `libQnnSystem.so` | 2,983,560 | `7e69258e1278cc9b2bb62dbc6e2a52c227a100d6505a13fd6324a87993d0bba8` |

## S25 QNN full-song result

The canonical input is a 273.699093-second, 44.1 kHz stereo PCM16 WAV with
12,070,130 frames. One QNN session completed all 48 windows; the last window
contained 86,258 source frames.

Unlike the tensor LiteRT timer, full-song `inferenceWallMs` covers `model.run`
only. Output readback is the separate 194.503 ms stage, or 4.052 ms per window;
adding it gives a same-scope mean of about 122.226 ms per window. The reported
end-to-end timer is the in-process QNN audio pipeline: it includes decode,
session setup, DSP, output commit, and output hashing, but excludes APK/process
startup, service prevalidation and staging, ADB upload, and host artifact pull.

| Metric | Result |
| --- | ---: |
| Completed windows | 48 / 48 |
| Session setup wall / CPU | 8360.269 / 8337 ms |
| Inference mean | 118.173 ms |
| Inference P50 / P95 | 118.184 / 120.022 ms |
| Inference min / max | 114.243 / 120.540 ms |
| Inference CPU / wall ratio | 0.1729 |
| Processing wall time | 26,347.805 ms |
| In-process pipeline end-to-end | 26,437.667 ms |
| In-process pipeline RTF | 0.096594 |
| In-process pipeline audio rate | 10.35x real time |
| Android thermal status | 0 -> 0; all 21 samples were 0 |
| Sampled peak PSS | 706.7 MiB |

The complete stage timers are retained separately because model acceleration
does not remove the host DSP cost:

| Stage | Wall time (ms) | Stage | Wall time (ms) |
| --- | ---: | --- | ---: |
| Model hash | 18.983 | Source hash | 30.761 |
| Decode | 488.463 | Window input | 126.086 |
| STFT | 2747.387 | Input layout | 172.946 |
| Input write | 20.565 | Inference, 48-window sum | 5672.323 |
| Output read | 194.503 | Output layout | 240.193 |
| ISTFT | 4506.924 | Scale and subtract | 79.642 |
| PCM conversion | 3585.935 | WAV write | 24.214 |
| Output commit | 0.466 | Output hash | 89.826 |

Output files were pulled and re-hashed on the host:

| Stem | Bytes | SHA-256 | Pre-quantization float peak | Saturated samples | Non-finite |
| --- | ---: | --- | ---: | ---: | ---: |
| Vocals | 48,280,564 | `f7e5a030f4d3be64da73a4890d9f5e91de3b0f690e74b2e9e35fe82846d0ac10` | 0.98447 | 0 | 0 |
| Instrumental | 48,280,564 | `ba7858db0311f1a3067b19c1309eeed933dfd708c7cd16f1c050f5f43ff401ca` | 1.34834 | 7,801 positive / 5,374 negative | 0 |

The instrumental clipping is a quality gate. Because this batch did not run
CPU or GPU through the same full-song driver, Batch 0 alone cannot attribute it
to QNN or compare complete output quality across backends.

## Setup amortization

For the S25 cold-session medians:

```text
GPU total(N) = 581.82 ms + N * 279.00 ms
QNN total(N) = 8558.13 ms + N * 122.12 ms
break-even N = 50.84 windows
```

At 48 windows, these model-plus-setup estimates are 13.974 seconds for GPU and
14.420 seconds for QNN. At 51 windows they are 14.811 and 14.786 seconds
respectively. This is a model-only estimate, not an end-to-end comparison: the
batch has a QNN full-song run but no corresponding GPU full-song run.

## Batch 0 closure and next gates

Batch 0 closes the infrastructure and calibration objective for 9662:

- the strict versioned contract, full host/device identity chain,
  overwrite-protected retained host output directories, installed-APK check,
  and fail-closed accelerator evidence worked on all accepted reports;
- LiteRT CPU and bounded GPU can execute the model on both devices;
- QNN v79 can execute and fully delegate the model on the S25;
- the current QNN runtime matrix does not provide an S10 path; and
- no result from this batch is a production support declaration.

Before using this evidence for application adaptation, the next gates should
be performed in this order:

1. Run the same full-song fixture through CPU, bounded GPU, and QNN, then
   compute SI-SDR versus ORT, waveform error, clipping, and residual
   consistency as backend-fidelity measures. Separation-quality SI-SDR needs a
   different dataset with ground-truth isolated stems.
2. Complete controlled listening or ABX qualification for QNN output, focusing
   on low-energy vocals and transient-heavy passages where tensor error can be
   perceptually misleading.
3. Measure end-to-end GPU versus QNN break-even with real files, session reuse,
   app lifecycle interruption, and any available compiled-context cache.
4. Add energy instrumentation before making battery claims; Android thermal
   status and instantaneous current are insufficient.
5. Onboard the next candidate only after it has its own versioned sidecar,
   exact ONNX/TFLite identity, fixed real tensor, stem semantics, and desktop
   reference. Do not reuse the 9662 contract by shape alone.

## Verification

Before the device batch, the standard and QNN v79 unit-test variants passed,
both debug APK flavors built from the frozen clean revision, and both benchmark
PowerShell runners parsed successfully. The final APK hashes above were checked
again against the installed packages after the runs.

Relevant verification targets are:

```text
:app:testStandardDebugUnitTest
:app:testQnnV79DebugUnitTest
:app:assembleStandardDebug
:app:assembleQnnV79Debug
PowerShell AST parse of both benchmark runners
```

## Raw evidence

Raw reports, device samples, logcat, QNN IR, tensors, and WAV files are local
test material under the ignored `outputs/` tree. They are intentionally not
committed.

- S10 accepted tags:
  `outputs/android-benchmark/192.168.8.181_5555/b0screen-9662-20260803-s10-*`
- S25 accepted tags:
  `outputs/android-benchmark/192.168.8.197_5555/b0screen-9662-20260803-s25-*`
- S25 full-song report:
  `outputs/android-benchmark/192.168.8.197_5555/b0screen-9662-20260803-s25-qnn-audio-full/report.json`
- excluded diagnostic tags:
  `outputs/android-benchmark/*/b0final-9662-20260731-*`

Each accepted tag contains `host-identity.json`, `report.json`, and
`device-samples.jsonl`. QNN tags retain runtime logcat and the verified IR; GPU
dispatch evidence is embedded in `report.json`. The full-song directory
additionally contains the two pulled WAV outputs.

The frozen build and runner workflow is documented in the repository README.
Retained host output directories are overwrite-protected, and the runners
reject report identities that do not match the contract, source revision,
runtime artifact, APK, model, or input supplied by the host. Removing a local
directory permits deliberate tag reuse and replacement of remote artifacts.
