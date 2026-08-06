# LiteRT 2.1.5 Official HTDemucs 4-Stem S25 Experiment

Date: 2026-08-05

Status: host export and S25 CPU/GPU+CPU execution complete; host candidate
admitted; strict per-stem device numerical gate failed; not a product
qualification.

## Decision

The official four-stem HTDemucs base checkpoint `955717e8` was exported as a
project-owned 7.8-second FP32 neural-core LiteRT artifact and executed on the
Galaxy S25 with LiteRT 2.1.5.

The result supports one narrowly scoped continuation:

- continue a CPU-only research prototype for offline separation or production
  ahead of playback;
- keep `readyWindowCount=2` for the first playback integration;
- reject the tested GPU+CPU FP32 profile; and
- do not call this artifact a product candidate while the frozen per-stem
  device parity gate remains failed.

The CPU completed three 30-second songs at end-to-end RTF `0.670-0.686` and a
180-second sustained run at RTF `0.617`. The GPU path did not reproduce an
execution failure and all outputs were finite, but it delegated only 158 of
3458 nodes. Its median, mean, P95, prepare time, and memory were all worse than
CPU.

This experiment intentionally excludes QNN.

The model choice follows Batch 4A: Psytrance ONNX had clearly worse listening
quality and more cross-talk, while official base, the complete fine-tuned bag,
and the two base/specialist hybrids were not stably distinguishable in the
multi-song blind comparison. Official base is therefore the simplest general
four-stem mobile export target from that batch. This does not establish
perceptual equivalence among the four non-Psytrance variants.

## Identity

Device:

```text
Samsung SM-S9310 (Galaxy S25)
SoC: QTI SM8750
Android API: 35
ABI: arm64-v8a
ADB serial: private device endpoint; pass as <device-serial>
```

Runtime and APK files:

```text
runtime: com.google.ai.edge.litert:litert:2.1.5

app-standard-debug.apk
SHA-256 2a66d77f2c2add4057007e424c6157640c47d3e7ca9c5c003741399768659230

app-standard-debug-androidTest.apk
SHA-256 af314e19741eacd5eb28ac30f4aa886d35b8a2a3ac6d93b6cc01cb39f04a20af
```

Those APK hashes were computed from the current local build files after the
run. They are not embedded in the device report or batch evidence and must be
treated as post-run local identities, not as self-bound APK provenance.

The runtime report pins the native libraries actually loaded:

```text
libLiteRt.so
SHA-256 366e3e040b00692158f9f8f9105870672c93348a3d8e9024120b40045a074b0b

libLiteRtClGlAccelerator.so
SHA-256 d22d9490c43a9428a6047564560dae83ce32a616658baa324b43843bfb066e89
```

### APK provenance limitation

Every four-stem probe and E2E report records `sourceRevision` as the literal
string `$rev` and records the Maven runtime artifact SHA as
`maven-unresolved`. The repository HEAD observed around this experiment was
`3510d5c263ab8236cd96cfeb07feef1c1a5c0dff`, but the APK does not self-attest
that revision. The APK files, native libraries, model, manifest, checkpoint,
and tools are hash-pinned, but this evidence must not be described as a fully
self-bound source build. A future qualification run must fix the build-info
substitution and runtime artifact resolution first.

## Canonical Artifact

```text
modelId: htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0
file: htdemucs_4s.core.canonical_7p8s.fp32.tflite
bytes: 178,042,000
SHA-256:
9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81

candidate-manifest.json
bytes: 38,207
SHA-256:
134642ea71cfbb8174cb8f27a6696f3620b03d6556afcfade1457a4245b8a879
```

ABI:

```text
args_0    [1,2,343980]          FP32 waveform
args_1    [1,4,2048,336]       FP32 complex feature channels
output_0  [1,4,4,2048,336]     FP32 frequency branch
output_1  [1,4,2,343980]       FP32 time branch
stem order: drums,bass,other,vocals
sample rate: 44100 Hz
window: 343980 samples = 7.8 seconds
```

FlatBuffer inspection found one subgraph, 3458 operators, 4221 tensors, zero
custom operators, and minimum runtime version `2.16.0`. The signature is
`serving_default`; Android resolves the logical names `args_0`, `args_1`,
`output_0`, and `output_1` through that signature instead of depending on
internal tensor indices.

Source identity:

```text
official checkpoint: models/demucs/official-hf/htdemucs/955717e8.safetensors
bytes: 84,025,440
SHA-256 d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd

metadata: 955717e8.json
SHA-256 12540373de858920b60002ebfbe17738860dbf2e89df625dc3b7875af2d28491

bag manifest: htdemucs.yaml
SHA-256 239c445d0b14454d541ad8bd9bb271c9e536d267e8a4625208744cbb2e7bb66c

Demucs loader revision:
eeac1d15891af95b1288d2884b95baa3e5baa96c
```

The public `demucs-lite` neural-core boundary remains reference-only. This
FlatBuffer was generated from the official safetensors checkpoint and does not
copy third-party node IDs.

## Host Admission

The export used Python 3.12.3, Torch 2.11.0+cpu, litert-torch 0.9.1, and
ai-edge-litert 2.1.5. The layered host pipeline gate passed:

| Comparison | Aggregate SNR | Maximum absolute error |
| --- | ---: | ---: |
| single-window combined LiteRT vs Torch | 121.256 dB | `9.574e-7` |
| full two-window OLA LiteRT vs Torch | 90.781 dB | `2.009e-5` |
| overlap region | 118.803 dB | `4.992e-7` |
| EOF region | 125.551 dB | `2.384e-7` |

Torch neural-core OLA was bitwise equal to official `apply_model`, including
the official `TensorChunk` padding behavior. The old uniform tensor gate is
still reported as false; admission uses the frozen layered policy, including
the low-signal absolute-error rules. That host admission permits device
testing but does not predict a device pass.

## CPU Neural Core

The CPU profile used one LiteRT `CompiledModel`, four CPU threads, two warmup
runs, and ten measured invocations:

```text
profile: compiled-cpu-xnnpack
XNNPACK: 3297 / 3458 nodes, 75 partitions
prepare: 343.77 ms
outputs finite: yes
```

| Metric | Wall time |
| --- | ---: |
| minimum | 2450.19 ms |
| median | 2497.37 ms |
| mean | 2506.55 ms |
| P95 / maximum | 2572.36 ms |

Neural-core mean RTF is approximately `2.507 / 7.8 = 0.321`. The largest
in-invocation process snapshot was 1,126,095 KiB PSS, and native allocation
peaked at 1,013,296,736 bytes.

## GPU+CPU FP32

The GPU test was a real OpenCL/CPU hybrid, not strict GPU execution:

```text
profile: compiled-gpu-cpu-opencl-fp32
OpenCL: 158 / 3458 nodes, 3 partitions
XNNPACK remainder: 3141 / 3301 nodes, 77 partitions
prepare: 1614.48 ms
outputs finite: yes
```

The delegate log explicitly rejects the rank-5 output path and related
`ADD`/`MUL`/`RESHAPE` tensors, and also reports `BROADCAST_TO` and `GATHER_ND`
as unsupported. The widened Transformer body therefore remains predominantly
on CPU.

| Metric | CPU | GPU+CPU FP32 | GPU change |
| --- | ---: | ---: | ---: |
| median | 2497.37 ms | 2860.59 ms | +14.54% |
| mean | 2506.55 ms | 3161.41 ms | +26.13% |
| P95 / maximum | 2572.36 ms | 4010.33 ms | +55.90% |
| prepare | 343.77 ms | 1614.48 ms | +369.64% |

The final three measured GPU+CPU calls were 3979.29, 3984.10, and 4010.33 ms,
after the earlier measured calls stayed between 2694.37 and 2891.69 ms.
Thermal severity remained 0, so the two-stage latency behavior is recorded but
not attributed to thermal throttling.

The same in-invocation process snapshots used by the CPU result reached
1,816,625 KiB PSS on GPU+CPU, versus 1,126,095 KiB on CPU. Separate external
sampling observed peak total PSS of 1,850,625 KiB and GL/graphics PSS of
451,768 KiB. The external monitor is not directly interchangeable with the
internal snapshot series, but both show the GPU path's substantially larger
memory footprint. It moves only a small fraction of the graph to GPU; do not
advance this profile.

## Strict Device Numerical Gate

CPU and GPU+CPU both completed, produced finite tensors, and failed the same
frozen per-stem gate:

| Layer aggregate | CPU | GPU+CPU FP32 |
| --- | ---: | ---: |
| raw frequency SNR | 90.446 dB | 90.478 dB |
| frequency iSTFT SNR | 103.415 dB | 103.485 dB |
| time waveform SNR | 107.671 dB | 107.318 dB |
| combined waveform SNR | 109.070 dB | 109.378 dB |

Failed stems on both profiles:

```text
raw frequency: bass, vocals
frequency iSTFT: vocals
time waveform: vocals
combined waveform: bass, drums, vocals
```

For CPU, raw-frequency bass and vocals were 78.02 and 72.44 dB. Most waveform
failures were low-signal fixtures that exceeded the strict `1e-6` absolute
limit, including combined bass at `1.015e-6`, drums at `2.207e-6`, and vocals
at `2.877e-6`. Aggregate SNR cannot override a failed per-stem rule.

CPU and GPU+CPU were nevertheless close to one another: direct output
comparison measured 113.294 dB for `output_0` and 121.098 dB for `output_1`,
with maximum absolute differences `3.099e-6` and `1.341e-6`. This makes the
shared failure pattern unsurprising, but it does not convert either profile
into a pass.

## Three 30-Second CPU Runs

The device processed the same canonical 30-second PCM selected for the Batch
4A official-Torch renders. Each run completed six windows, wrote four complete
44.1 kHz stereo PCM16 stems, produced no non-finite values, and reported
thermal status 0.

| Track | Wall time | E2E RTF | Core median | Peak window PSS |
| --- | ---: | ---: | ---: | ---: |
| Athletics II | 20.107 s | 0.6702 | 2508.91 ms | 1,186,917 KiB |
| John Lennon - Imagine | 20.334 s | 0.6778 | 2561.61 ms | 1,185,092 KiB |
| Josiah James - Chasing The Wind | 20.592 s | 0.6864 | 2593.42 ms | 1,184,287 KiB |

Across the three songs, mean wall time was 20.344 seconds and mean RTF was
0.6781. Mean stage totals per 30 seconds were:

```text
neural-core inference: 15.289 s
STFT:                   0.621 s
iSTFT:                  1.894 s
PCM conversion:         1.446 s
```

Josiah's drums had 82 pre-quantization clipping samples. The official Torch
base render has exactly the same 82 samples, so this 30-second clipping event
belongs to the checkpoint/content pair rather than the LiteRT conversion.
All other 30-second stems had zero clipped samples.

### Device PCM16 vs official Torch

| Track | Aggregate SNR | Maximum difference | Correlation |
| --- | ---: | ---: | ---: |
| Athletics II | 89.144 dB | 1 LSB | 0.9999999994 |
| John Lennon - Imagine | 90.462 dB | 1 LSB | 0.9999999996 |
| Josiah James - Chasing The Wind | 87.543 dB | 2 LSB | 0.9999999991 |

Low-energy stems can have low per-stem SNR despite one-LSB differences. These
whole-render PCM16 results are strong practical rendering-parity evidence, but
they do not waive the separate frozen FP32 device gate.

## 180-Second Sustained CPU Run

Athletics II was run for 180 seconds and completed all 31 windows:

| Metric | Result |
| --- | ---: |
| E2E wall time | 111.140 s |
| E2E RTF | 0.61744 |
| neural-core total | 85.282 s |
| core minimum | 2444.39 ms |
| core median | 2606.93 ms |
| core mean | 2751.04 ms |
| core P95 nearest-rank | 3661.74 ms |
| core maximum | 3945.72 ms |
| peak window PSS | 1,186,415 KiB |
| peak native allocation | 1,013,319,872 bytes |
| peak Java heap used | 92,215,424 bytes |

All outputs were finite and all four WAVs contained exactly 7,938,000 frames.
Drums contained 120 pre-quantization clipping samples; unlike the 30-second
Josiah result, no 180-second official-Torch control was produced, so this is
recorded without assigning cause.

| Stem | WAV SHA-256 | Clipped samples |
| --- | --- | ---: |
| drums | `abf9b697f1e9c5ffc68548e6bd986b24c17098d650fe30b381eef13355af94bc` | 120 |
| bass | `021b6d40f5f19282bf6f482bfdb73041bdc7c3626496ad7b5b8498e030ec505d` | 0 |
| other | `fcffb05a47cbe28c476c6143106b106ebeea76e57e9ae65bb1af59c842602bc4` | 0 |
| vocals | `73c22d07cdae9b78969e900d0634f87eb2fa67b9ee80ef43badd27dd39a701bd` | 0 |

Windows 19-22 rose materially:

```text
windows 0-18 mean: 2543.11 ms
windows 19-22 mean: 3699.62 ms
windows 23-30 mean: 2770.58 ms
```

Latency then recovered. Android thermal severity was 0 for every window, so
the transient cannot be identified as thermal throttling from this evidence.

## Seek, Seams, and Buffer Projection

The analyzer performed three random-access reads per stem for every song and
verified shape, length, and readable content. Each 30-second stem covered five
OLA boundaries; each 180-second stem covered 30 boundaries.

Maximum boundary-delta/local-derivative-P95 ratios for the 30-second songs:

| Track | Maximum ratio across stems |
| --- | ---: |
| Athletics II | 1.000 |
| John Lennon - Imagine | 1.383 |
| Josiah James - Chasing The Wind | 1.672 |

For the 180-second run the maximum ratios were drums 1.365, bass 2.000, other
1.219, and vocals 1.000. These are objective boundary checks, not listening
quality scores.

The producer/consumer projection used complete measured window wall times:

| Run | Ready windows | Audio at start | Minimum refill margin | Projected underruns |
| --- | ---: | ---: | ---: | ---: |
| Athletics 30 s | 1 | 5.85 s | 2.688 s | 0 |
| Imagine 30 s | 1 | 5.85 s | 2.642 s | 0 |
| Josiah 30 s | 1 | 5.85 s | 2.625 s | 0 |
| Athletics 180 s | 1 | 5.85 s | 2.726 s | 0 |
| Athletics 30 s | 2 | 11.70 s | 8.552 s | 0 |
| Imagine 30 s | 2 | 11.70 s | 8.477 s | 0 |
| Josiah 30 s | 2 | 11.70 s | 8.444 s | 0 |
| Athletics 180 s | 2 | 11.70 s | 8.574 s | 0 |

This is not an `AudioTrack` underrun observation. Keep two ready windows until
the real playback data plane is tested under contention, lifecycle changes,
and background execution.

## Four Stems Are Not the Lighter Model

The official four-stem checkpoint is heavier than official HTDemucs-6s. Stem
count controls the output heads and post-processing volume, but it is not a
proxy for the shared neural-core size.

| Item | Official 4s | Official 6s | Four-stem change |
| --- | ---: | ---: | ---: |
| parameters | 41,984,456 | 27,414,996 | +53.14% |
| source safetensors | 84,025,440 B | 54,885,744 B | +53.09% |
| FP32 LiteRT artifact | 178,042,000 B | 117,624,880 B | +51.36% |
| Transformer dimension | 512 | 384 | +33.33% |

Both use `channels=48`, growth 2, depth 4, and five Transformer layers. The
four-stem model sets `bottom_channels=512`, while the six-stem checkpoint uses
`bottom_channels=0` and stays at its natural 384 channels. The four-stem model
therefore adds projections and runs the shared Transformer at 512 dimensions.

On the same frozen S25 neural-core fixture, four-stem CPU was 17.59% slower at
the median, 18.13% slower at the mean, and 19.52% slower at P95 than the
previous six-stem CPU run. Its smaller output DSP partly offsets this in an
end-to-end render, but the available 30-second four- versus six-stem E2E runs
used different songs. The roughly 2% wall-time difference is only directional;
the controlled neural-core comparison is the reliable compute comparison.

## Evidence

Artifact directory:

```text
models/demucs/generated/htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0/
```

Device evidence root:

```text
outputs/htdemucs4-official-s25-20260805/
```

Pinned top-level evidence:

| Evidence | SHA-256 |
| --- | --- |
| CPU result | `9ad474e3e062afa9406caa246821573b916495fc264fda97518759801f33e58e` |
| CPU parity | `81d43b6e9ad25699f2b105bdc29f9c3eedb61f6233566fc044a3d387144609a4` |
| GPU+CPU result | `098d0235922bef4816a79f4e95f2f505aa42b1f15f19512ebcec2ab3c62bec56` |
| GPU+CPU parity | `84d91e6dc47953b26bc51afa7206b4f06a9e4e1ccb39dbdac8586006c0c3700d` |
| 30-second batch progress | `f4541de41ee2a107c1b5f44e1757370e08a6f37d596148f76d783635eda3cf4d` |
| 180-second batch progress | `005179775bee915e8643d992006dd4f03dbe8e9f1afeecf8ca3377c08ec671d1` |
| 180-second report | `e38f8045405eb07f14f7a986f5c9cc69844c258e219fb01cc1235f695b97cd2d` |
| 180-second analysis | `af0e3187d350440abddd981009cc32bfe7270bdf23893c67627ff443fc44a462` |

The per-track directories contain each 30-second device report,
`e2e-analysis.json`, and `s25-vs-batch4a-comparison.json`. The latter pins the
shared selected PCM SHA and both device and official-Torch reports.

The per-attempt `full/progress.json` files retain `status: running` even after
all expected windows and output frames were written. Completion is established
by the enclosing attempt, final report, batch progress, output validation, and
instrumentation result, all of which are complete/OK. The stale checkpoint
status is a harness defect and must not be used as the final run status.

The original Athletics 30-second `device-envelope.json` and instrumentation
stdout/stderr were overwritten by the first 180-second invocation before the
duration-specific naming fix. The 30-second report, attempt, WAVs, batch
progress, parity comparison, and analyzer result remain intact. The retained
duration-specific envelope and instrumentation files describe the 180-second
run.

Key tool identities:

| Tool | SHA-256 |
| --- | --- |
| `export_htdemucs_4s_litert_candidate.py` | `1ab1a42727ed036bb0fa5ca9c6950000d9d5b4ff3faae37410ef96ad889db3ad` |
| `freeze_htdemucs_4s_litert_candidate.py` | `1c267d328a7bda869a352cd6e827d8c60afdb788ee74126519442ac82bd654a4` |
| `validate_htdemucs_4s_litert_outputs.py` | `b93a3abdcb0ecc067c056b353bcffa345cbf92411f2bf1c12fd5a082aef4e086` |
| executed `tools/frozen-runners/cf82fd50.../run_htdemucs_4s_s25_batch.py` | `cf82fd50f5eee601913e5e1adb59c791c8893840c9484861f84fc3b31f1e3d8c` |
| current hardened `tools/run_htdemucs_4s_s25_batch.py` | `4eafe8abae340c981125dcb03a0c669b4c39007eebdf9aab7b28bd3de2614780` |
| `analyze_htdemucs_e2e_outputs.py` | `e2f60de8af3d6be0d5745bf902d38bae9f68d3d679f38f7ee5c30c2e7f2cdbb0` |
| `compare_htdemucs_stem_sets.py` | `d769de60e53e1abfb23e0d1e364a981ecba095b28dd27e447f5ab2ca3dad784b` |

## Remaining Boundary

The evidence supports CPU feasibility research, not application support. A
product-facing decision still requires:

1. fix APK source/runtime provenance and rerun the frozen probe;
2. investigate or explicitly revise the strict per-stem FP32 device gate with
   a justified fixture policy, rather than using aggregate SNR to hide it;
3. implement the roadmap's N-stem cache, hydration, and playback data plane;
4. measure real `AudioTrack` underruns, seek, cancellation, service restart,
   and background contention; and
5. add a controlled 180-second official-Torch comparison before attributing
   the sustained drums clipping samples.

No four-stem cancel/resume run or GPU end-to-end render was performed in this
batch. GPU neural-core results are already poor enough that a GPU E2E run is
not warranted unless a materially different delegate graph becomes available.
