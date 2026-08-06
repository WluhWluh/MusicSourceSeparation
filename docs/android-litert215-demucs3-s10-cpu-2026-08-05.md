# LiteRT 2.1.5 HTDemucs CPU Matrix on Galaxy S10

Date: 2026-08-05

Status: 5-second smoke and three-model by three-track 30-second CPU matrix
complete. This is a research performance experiment, not product
qualification.

## Decision

All three canonical FP32 neural-core artifacts execute successfully through
LiteRT 2.1.5 CPU on the Galaxy S10:

- official HTDemucs six-stem;
- official HTDemucs four-stem base; and
- HTDemucs six-stem guitar-ft.

Every formal run completed 6/6 windows with the requested CPU backend and four
threads. All 48 formal output WAVs are finite, have the expected 30-second
PCM16 contract, and match the byte size and SHA-256 recorded by the device.
Android thermal status remained 0 in every window.

The current end-to-end implementation is not fast enough for playback-time
production on S10. Mean 30-second RTF is `2.960` for official six-stem,
`2.401` for official four-stem, and `2.905` for guitar-ft. A ready-window
buffer would delay an underrun but cannot correct a sustained producer rate
slower than playback.

The result is more specific than "the model is too slow." The warm neural
core alone remains inside the 5.85-second stride deadline for all three
artifacts. Its aggregate median is `4.243 s` for official, `5.116 s` for
official four-stem, and `4.302 s` for guitar-ft. The dominant S10 bottleneck
is the current serial Java/JTransforms frequency-branch iSTFT: its warm median
is `9.937 s` per six-stem official window, `6.451 s` per four-stem window, and
`9.694 s` per guitar-ft window.

Therefore:

- the three artifacts are proven runnable on S10 CPU;
- the existing E2E path is suitable only for bounded offline investigation;
- DSP optimization must precede any S10 playback-buffer experiment; and
- this batch says nothing about S10 GPU or NPU.

The guitar-ft artifact remains `diagnosticOnly`, `researchOnly`, and
`hostAdmissionStatus=not-admitted`. Its previously observed listening-quality
changes are not an admission waiver. Its S10 performance is nevertheless
effectively the same class as official six-stem.

## Device and Runtime Identity

```text
device: Samsung SM-G9730 (Galaxy S10, beyond1q)
hardware serial SHA-256:
d3736be879f9280e55b2850a58604601b3b81955a97536e17b6f56d8c4627694
SoC: QTI SM8150 / board msmnile
Android: 12 / API 31
ABI reported by runtime: arm64-v8a, armeabi-v7a, armeabi
build fingerprint:
samsung/beyond1qltezc/beyond1q:12/SP1A.210812.016/G9730ZCU8HWE2:user/release-keys
ADB transport: supplied at runtime; intentionally omitted from this report
```

The runner gates the hardware-serial digest, product model, SoC, API, Android
release, manufacturer, and device codename before staging a model. Every ADB
command explicitly selects the runtime-supplied S10 transport; the
simultaneously connected S25 was not used by this batch.

Runtime and source identity embedded in every device report:

```text
runtime: com.google.ai.edge.litert:litert:2.1.5
LiteRT 2.1.5 AAR SHA-256:
a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37

source revision: 3510d5c263ab8236cd96cfeb07feef1c1a5c0dff
source dirty: true
branch: experiment/model-matrix-demucs-multistem
```

The APKs were rebuilt with those values before the run. The host runner then
hashed both local APKs and the installed `/data/app/.../base.apk` files and
required exact byte-size and SHA equality:

| APK | Bytes | SHA-256 |
| --- | ---: | --- |
| app-standard-debug.apk | 144,715,111 | `66cc8e91dba90370dfd2482e08bbdf963e407a788efab3095dc0e93c97850bcf` |
| app-standard-debug-androidTest.apk | 517,060 | `66fb65cbfd05a7c676fe6fa1b8f142c8d95bd65c0395fc40eafc0fc32ad36982` |

This closes the earlier S25 build-info substitution problem: the exact
installed APK files, source revision, dirty state, and Maven AAR hash were all
validated and recorded. The APK identity is held in the `device.json`
sidecar, however, and `matrix-30s.json` does not embed that sidecar's digest.
The evidence is therefore strongly cross-checked but not one cryptographically
self-contained object.

## Artifact and Contract Identity

| Variant | Stems | Bytes | SHA-256 | Status |
| --- | ---: | ---: | --- | --- |
| `official` | 6 | 117,624,880 | `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7` | host admitted; prior S25 device gate failed |
| `official-4s` | 4 | 178,042,000 | `9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81` | host admitted; prior S25 device gate failed |
| `guitar-ft` | 6 | 117,729,544 | `ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5` | diagnostic/research/not admitted |

Shared input contract:

```text
sample rate: 44,100 Hz
channels: stereo
window: 343,980 samples = 7.8 seconds
stride: 257,985 samples = 5.85 seconds
overlap: 25 percent
waveform input: [1,2,343980] FP32
spectrum input: [1,4,2048,336] FP32
CPU threads: 4
```

The output stem order is `drums,bass,other,vocals,guitar,piano` for the two
six-stem artifacts and `drums,bass,other,vocals` for official four-stem. The
Android harness validates the signature and tensor ABI. The host runner
independently verifies the model ID, file name, bytes, SHA, stem inventory,
and candidate-status extension in every pulled report. The older official
six-stem schema intentionally omits the candidate-status extension; that
omission is validated explicitly rather than interpreted as false values.

This E2E matrix does not contain the frozen single-window FP32 device fixtures.
Successful PCM16 rendering therefore does not convert either official
artifact's earlier S25 strict device-gate failure into a pass.

## Test Method

The batch used the first exactly 30 seconds (`1,323,000` frames) of three real
songs already used by the S25 and host experiments:

| Role | Track | Selected PCM SHA-256 |
| --- | --- | --- |
| guitar-heavy | Athletics - II | `c35318b7d17823bedebe755a65fe636ab3527ef964cbc85eb151717c087869f3` |
| piano-heavy | John Lennon - Imagine | `83ee55a04439ca44035f47599277c1f1a0bdfffebf2fff31777c13daadd22529` |
| vocals/drums-sensitive | Josiah James - Chasing The Wind | `10f8760bc48dc0a78ba6f595107191a38d24d768206f959dddec5f3794d0a374` |

The selected PCM SHA is identical across all three models for a track and
matches the previously frozen canonical decode. Each run used one session and
six windows. Model order was rotated to reduce fixed heat-order bias:

```text
Athletics: official, official-4s, guitar-ft
Imagine:   official-4s, guitar-ft, official
Josiah:    guitar-ft, official, official-4s
```

There was a 10-second cooldown between runs. A run could start only with
`MemAvailable >= 2,000,000 KiB`. The device was not charging; battery level
moved from 80 to 69 percent. Battery service reported 11.6 to 15.5 C, the
largest AP sensor value found by the independent evidence audit was 42.7 C,
and all per-window Android thermal statuses were 0. These are favorable
thermal conditions, not a warm-ambient stress test.

The performance summaries use:

- prepare time for compile, tensor allocation, and writer setup;
- first-window neural-core inference separately;
- 15 warm neural-core and complete-window observations per model;
- device-attempt E2E time, excluding the preceding full-file MP3
  decode/canonicalization, host instrumentation, and ADB transfer;
- the largest PSS/native/Java value among window-end process snapshots; and
- pre-quantization finite and clipping statistics.

Window-end PSS is not an externally sampled instantaneous peak.

## Five-Second Smoke

The one-window smoke established basic execution and output integrity before
the nine formal runs:

| Variant | Prepare | Core inference | E2E | RTF | Peak window-end PSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| official | 629.98 ms | 4219.95 ms | 15.634 s | 3.127 | 1,094,720 KiB |
| official-4s | 693.36 ms | 4905.69 ms | 13.106 s | 2.621 | 1,158,035 KiB |
| guitar-ft | 642.72 ms | 4147.24 ms | 15.581 s | 3.116 | 1,094,833 KiB |

All smoke outputs were finite, thermal status was 0, and every pulled WAV
matched its device-recorded digest. The smoke includes cold FFT/session costs
and is not used for sustained ranking.

## Aggregate 30-Second Performance

Warm P50 and P95 pool five post-first windows from each of three tracks.

| Variant | Artifact MiB | Prepare mean | Warm core P50 / P95 | Core / 5.85 s stride | Warm iSTFT P50 / P95 | Warm full-window P50 / P95 | Mean 30 s E2E | Mean RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official | 112.18 | 614.43 ms | 4242.55 / 4315.96 ms | 0.725 | 9937.21 / 10709.44 ms | 14424.23 / 15214.70 ms | 88.803 s | 2.960 |
| official-4s | 169.79 | 651.20 ms | 5116.31 / 5400.57 ms | 0.875 | 6451.43 / 7086.35 ms | 11781.53 / 12403.26 ms | 72.027 s | 2.401 |
| guitar-ft | 112.28 | 601.44 ms | 4301.64 / 4574.23 ms | 0.735 | 9693.57 / 10484.42 ms | 14309.39 / 15095.05 ms | 87.155 s | 2.905 |

The official and guitar-ft cores share the same architecture. Their pooled
warm medians differ by 1.39 percent; that is not a material device-performance
reason to choose between their weights. Official four-stem has a 51 percent
larger FlatBuffer and its neural core is about 20.6 percent slower than
official six-stem, but producing and reconstructing only four stems makes its
complete E2E run about 18.9 percent faster.

## Per-Run Results

| Track | Variant | Prepare | First core | Warm core P50 / P95 | E2E | RTF | Peak window-end PSS | Clipped samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Athletics | official | 619.76 ms | 4363.76 ms | 4225.51 / 4247.72 ms | 89.312 s | 2.977 | 1080.6 MiB | 0 |
| Athletics | official-4s | 666.52 ms | 4963.15 ms | 5096.70 / 5132.66 ms | 72.625 s | 2.421 | 1137.1 MiB | 0 |
| Athletics | guitar-ft | 589.36 ms | 4771.11 ms | 4320.37 / 4357.48 ms | 89.151 s | 2.972 | 1079.6 MiB | 0 |
| Imagine | official-4s | 640.60 ms | 5025.83 ms | 5166.18 / 5232.95 ms | 73.589 s | 2.453 | 1135.7 MiB | 0 |
| Imagine | guitar-ft | 601.67 ms | 4250.17 ms | 4476.84 / 4574.23 ms | 84.274 s | 2.809 | 1081.3 MiB | 0 |
| Imagine | official | 636.72 ms | 4251.03 ms | 4244.15 / 4315.96 ms | 88.745 s | 2.958 | 1079.9 MiB | 0 |
| Josiah | guitar-ft | 613.30 ms | 4249.45 ms | 4241.43 / 4258.24 ms | 88.041 s | 2.935 | 1080.3 MiB | 0 |
| Josiah | official | 586.80 ms | 4333.87 ms | 4246.14 / 4281.39 ms | 88.353 s | 2.945 | 1079.5 MiB | 74 |
| Josiah | official-4s | 646.48 ms | 5039.80 ms | 5098.65 / 5400.57 ms | 69.867 s | 2.329 | 1136.0 MiB | 82 |

The only clipping is in the Josiah drums stem. Official six-stem reports 74
samples, exactly matching the same-weight official Torch render. Official
four-stem reports 82, matching the existing S25/host evidence. Guitar-ft has
no clipped samples in these three clips. The clipping is therefore not a new
S10 conversion symptom.

## DSP Bottleneck

Warm per-window medians show the current frequency reconstruction cost:

| Variant | STFT | Core | iSTFT | Complete window | Mean iSTFT share |
| --- | ---: | ---: | ---: | ---: | ---: |
| official | 54.25 ms | 4242.55 ms | 9937.21 ms | 14424.23 ms | 68.9% |
| official-4s | 55.96 ms | 5116.31 ms | 6451.43 ms | 11781.53 ms | 54.3% |
| guitar-ft | 55.11 ms | 4301.64 ms | 9693.57 ms | 14309.39 ms | 67.5% |

`HtdemucsDsp.frequencyToWaveform()` currently loops stems, channels, and 336
frames serially, rebuilds the Hermitian spectrum, and invokes a 4096-point
complex inverse FFT for each frame. The six-stem path therefore performs 4032
inverse FFTs per window. Its near-linear four-versus-six stem timing confirms
that reconstruction work, rather than model output content, dominates.

Existing S25 evidence provides a useful scale check. The table uses pooled
warm medians for core and iSTFT and mean RTF:

| Variant | S10 / S25 warm core | S10 / S25 warm iSTFT | S10 / S25 E2E RTF |
| --- | ---: | ---: | ---: |
| official | 2.01x | 22.64x | 4.45x |
| official-4s | 2.01x | 22.67x | 3.54x |
| guitar-ft | 2.03x | 22.63x | 4.28x |

The official four-stem and guitar-ft comparisons use the same three tracks.
The official six-stem S25 reference is the previously pinned 30-second Coast
Town run because an exact same-song bounded official S25 batch does not exist.
The common shape and the consistent 22.6x iSTFT ratio across all three
variants still identify the implementation bottleneck clearly.

The old S25 reports do not have the complete build provenance achieved here:
official four-stem records `$rev`/`maven-unresolved`, and guitar-ft also lacks
a resolved source identity. The ratios are valid research measurements with
matched model/input contracts where stated, but they are not a new S25
qualification comparison.

Before repeating playback projections on S10, evaluate a parity-preserving
DSP change such as a real-IFFT path and/or parallel stem-channel reconstruction
with independent FFT state. A native NEON-capable FFT backend is also a valid
research branch. Any replacement must pass the frozen Torch/JTransforms DSP
fixtures and full OLA/EOF gates before device timing is compared. The current
serial result should remain the baseline.

## PCM16 Numerical Comparison

Each S10 output was compared with the same-weight, same-selected-PCM host
render using the repository's streaming comparator. Aggregate results:

| Track | Variant | S10 vs host SNR | Maximum error | Correlation |
| --- | --- | ---: | ---: | ---: |
| Athletics | official | 89.597 dB | 1 LSB | 0.999999999452 |
| Athletics | official-4s | 89.147 dB | 1 LSB | 0.999999999392 |
| Athletics | guitar-ft | 87.688 dB | 1 LSB | 0.999999999152 |
| Imagine | official | 83.673 dB | 1 LSB | 0.999999997854 |
| Imagine | official-4s | 90.465 dB | 1 LSB | 0.999999999551 |
| Imagine | guitar-ft | 82.860 dB | 1 LSB | 0.999999997415 |
| Josiah | official | 84.713 dB | 2 LSB | 0.999999998311 |
| Josiah | official-4s | 87.542 dB | 2 LSB | 0.999999999120 |
| Josiah | guitar-ft | 84.358 dB | 2 LSB | 0.999999998167 |

Official four-stem S10 versus S25 is `99.867`, `101.995`, and `102.043 dB`
for the three tracks, with maximum error 1 LSB. Guitar-ft S10 versus S25 is
`98.622`, `97.163`, and `99.158 dB`, also with maximum error 1 LSB. These
cross-device results independently show that S10 preserves the already
validated S25 PCM16 behavior.

Across all three tracks together, S10 versus S25 is `101.877 dB` for official
four-stem and `98.716 dB` for guitar-ft. Only 0.0504 and 0.0439 percent of
samples differ, respectively, and every non-zero difference is exactly one
PCM16 LSB.

The official six-stem full-song S25 files are intentionally not used for a
direct bounded comparison: whole-song and 30-second runs scan different
global-normalization ranges. Its exact same-weight host comparison above is
the valid reference.

Low-energy per-stem SNR can be much lower even when maximum error is one LSB.
The aggregate PCM16 result is evidence of stable device rendering; it does not
replace the frozen FP32 per-stem host/device gate and does not change any model
admission status.

## Memory and Resource Result

| Variant | Largest per-run peak window-end PSS | Native allocation | Java used |
| --- | ---: | ---: | ---: |
| official | 1080.6 MiB | 861.9 MiB | 128.4 MiB |
| official-4s | 1137.1 MiB | 901.7 MiB | 87.9 MiB |
| guitar-ft | 1081.3 MiB | 862.0 MiB | 128.4 MiB |

The larger four-stem graph costs about 56 MiB more window-end PSS and 40 MiB more
native allocation than either six-stem graph, despite producing fewer stems.
No run crossed the 2,000,000 KiB `MemAvailable` start threshold. After the
batch the app process had exited, `MemAvailable` was 3,677,496 KiB, and all
three HTDemucs model files had been removed from the device. The pre-existing
HQ4 ONNX/TFLite files were left untouched.

## Evidence

Primary evidence:

```text
outputs/htdemucs-s10-cpu-matrix-20260805/device.json
SHA-256 0032fa416f867d748a56c8189464f7d398dc255f03cfb5c213686533a958512a

outputs/htdemucs-s10-cpu-matrix-20260805/matrix-5s.json
SHA-256 6b1bbdaf12e72e15ba5a54d72d07310c0b8bebf69e2cacf7dce5b825051ece17

outputs/htdemucs-s10-cpu-matrix-20260805/matrix-30s.json
SHA-256 4026175c1ed0d3c4337d5943420befed69b05108aa0bec189acdcc7d23cacaac
```

The formal batch contains 9 reports and 48 WAVs. With smoke included, the
evidence root contains 12 reports and 64 WAVs, representing 134,064,000 PCM16
samples. There are no `.partial` or `.invalid-*` paths. Each run directory
contains the device report, six or four output WAVs, and for formal runs a
`same-weight-comparison.json`. The nine comparison files record every
input/report/file SHA, per-stem metrics, and aggregate metrics.

Two evidence-hygiene limitations remain:

- all 12 `full/progress.json` checkpoint files retain `status="running"` even
  though their completed-window/output-frame counts are final and every
  authoritative `report.json`, attempt, and matrix entry is complete; and
- the formal 30-second start overwrote the shared root `device.json` written by
  the smoke. The individual smoke reports still bind their runtime/device/model
  identities, but the root sidecar is the formal-batch snapshot and the smoke
  matrix does not bind a separate device snapshot.

Consumers must use `report.json` and the terminal matrix status, not the stale
checkpoint status, as run completion evidence.

Tool identity:

```text
tools/frozen-runners/
  c0c3706347f6acdd03341c9d7f8e00955b768c389e8ec38baba6308380eae807/
  run_htdemucs_cpu_matrix.py
SHA-256 c0c3706347f6acdd03341c9d7f8e00955b768c389e8ec38baba6308380eae807

tools/compare_htdemucs_stem_sets.py
SHA-256 d769de60e53e1abfb23e0d1e364a981ecba095b28dd27e447f5ab2ca3dad784b
```

The frozen path is the byte-identical serial runner used by this batch. The
tracked `tools/run_htdemucs_cpu_matrix.py` is a later parallel-capable,
privacy-hardened successor and must not be substituted for that historical
tool identity. Its current successor SHA-256 is
`4277970030200d81dbbb1e022c59ce37ea92158d8bc39a7141054c9a0fefe1a7`.

## Reproduction

The APK build used:

```text
.\gradlew.bat :app:assembleStandardDebug \
  :app:assembleStandardDebugAndroidTest --no-daemon \
  -PbenchmarkSourceRevision=3510d5c263ab8236cd96cfeb07feef1c1a5c0dff \
  -PbenchmarkSourceDirty=true \
  -PbenchmarkRuntimeArtifactSha256=a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37
```

The historical invocation used the frozen runner above. Its runtime ADB
transport and raw hardware-serial argument are intentionally not published.
The equivalent privacy-preserving invocation with the current successor is:

```powershell
python tools/run_htdemucs_cpu_matrix.py `
  --serial $env:S10_ADB_SERIAL `
  --source-root $env:MSS_SOURCE_ROOT `
  --device-label s10 `
  --expected-device-model SM-G9730 `
  --expected-hardware-serial-sha256 d3736be879f9280e55b2850a58604601b3b81955a97536e17b6f56d8c4627694 `
  --expected-soc-model SM8150 `
  --expected-sdk 31 `
  --expected-os-release 12 `
  --expected-manufacturer samsung `
  --expected-device-codename beyond1q `
  --duration-seconds 30 `
  --threads 4 `
  --cooldown-seconds 10 `
  --output-root outputs/htdemucs-s10-cpu-matrix-20260805
```

No GPU, QNN, NPU, AudioTrack, seek/cancel/resume, background-contention,
power, or warm-ambient sustained tests are implied by this batch.
