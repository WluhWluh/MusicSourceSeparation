# LiteRT 2.1.5 HTDemucs native DSP contracts

Date: 2026-08-08

## Decision

One constrained native packed-real DSP implementation is valid for the three
tested HTDemucs neural-core artifacts. Official 6-stem and guitar-ft share the
same six-stem DSP ABI. Official 4-stem uses an independent four-stem output ABI;
it is not produced by truncating six-stem output.

The native implementation passes raw-float and full-audio numerical gates on
S25 and S10. Compared with the same APK's Kotlin/JTransforms path, native DSP
reduces 30-second CPU E2E time by 7.9-10.5% on S25 and 16.6-28.4% on S10. The
official 4-stem model remains slower than official 6-stem because neural-core
inference dominates; fewer output stems do not make the model intrinsically
lighter.

This is experimental evidence, not a Booming SS integration qualification.
The S25 180-second runs reached thermal status 1 or 2, and the current runner
still copies LiteRT output through Java `FloatArray` values before JNI
postprocessing.

## Scope and identity

- Branch: `experiment/demucs-native-dsp-contracts`
- LiteRT: `2.1.5`, `com.google.ai.edge.litert:litert:2.1.5`
- CPU profile: four LiteRT threads
- DSP profile: native pocketfft packed-real `r2c/c2r`, four workers
- Postprocess: `fused-reuse`
- Window: 343,980 samples at 44.1 kHz, 25% overlap
- STFT: FFT 4096, hop 1024, 2048 retained bins, 336 frames, periodic Hann
- Input: prefix of `Athletics-II.mp3`, file SHA-256
  `b781907393b5b5a5b9ef91e6fb0c5216a902a75064a8a35875dc6a6b94935dee`
- Devices: S25 `SM-S9310` / `SM8750` / API 35 and S10 `SM-G9730` /
  `SM8150` / API 31

The authoritative published executable contracts remain in the separate
`BSSModels/bss-tflite` repository:

| Variant | Contract | Artifact SHA-256 | Output ABI | Status |
| --- | --- | --- | --- | --- |
| Official 6s | `htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0@1` | `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7` | `[1,6,4,2048,336]`, `[1,6,2,343980]` | MIT review complete |
| Official 4s | `htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0@1` | `9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81` | `[1,4,4,2048,336]`, `[1,4,2,343980]` | MIT review complete |
| Guitar FT 6s | `htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0@1` | `ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5` | `[1,6,4,2048,336]`, `[1,6,2,343980]` | Research only; license review required |

The frozen source order is `drums`, `bass`, `other`, `vocals` for four-stem and
adds `guitar`, `piano` for six-stem. The native plan rejects every other source
count, order, shape, FFT, padding, and normalization profile.

## Implementation

`HtdemucsNativeDspContract` mirrors the three published executable identities
and freezes their DSP dimensions. `HtdemucsDspSession` gives the benchmark one
interface for Kotlin/JTransforms and native packed DSP. The native plan uses
pocketfft, parameterizes output and iSTFT workspaces by a validated source
count of four or six, and supports both frequency-only reconstruction and
frequency plus time-branch combination.

The canonical E2E runner accepts `canonicalE2eDspMode` values
`kotlin-jtransforms` and `native-packed`. It records the executable contract ID,
DSP mode, FFT implementation, worker count, stage timings, finite-output
checks, output hashes, process samples, and thermal status in the same report
schema.

This batch does not implement a zero-copy LiteRT/native boundary. LiteRT
`TensorBuffer.readFloat()` still allocates frequency and waveform output
arrays, which are passed to JNI. It also does not add QNN, GPU, arbitrary
Demucs shapes, selective-stem rendering, or application playback/cache logic.

## Synthetic DSP parity

Each device compared native packed DSP with Kotlin/JTransforms for the same
deterministic waveform, frequency tensor, and time branch. The acceptance gate
was finite output, SNR at least 80 dB, and maximum absolute error at most
`1e-3`.

| Device | Contract | Native STFT median | Native postprocess median | STFT SNR | iSTFT SNR |
| --- | --- | ---: | ---: | ---: | ---: |
| S25 | Official 6s | 13.069 ms | 28.466 ms | 133.65 dB | 138.64 dB |
| S25 | Official 4s | 12.997 ms | 18.939 ms | 133.65 dB | 138.61 dB |
| S25 | Guitar FT 6s | 13.087 ms | 28.324 ms | 133.65 dB | 138.64 dB |
| S10 | Official 6s | 47.156 ms | 74.047 ms | 133.65 dB | 138.64 dB |
| S10 | Official 4s | 47.446 ms | 50.616 ms | 133.65 dB | 138.61 dB |
| S10 | Guitar FT 6s | 36.739 ms | 79.840 ms | 133.65 dB | 138.64 dB |

Maximum STFT error was `9.537e-7`. Maximum iSTFT error was `1.192e-7` for
six-stem and `1.043e-7` for four-stem. Frequency plus time-branch combination
also passed for all contracts. The S10 microbenchmark used only three measured
runs, so the per-variant STFT timing spread is scheduling noise rather than a
model-weight effect; the preprocessing contract is identical.

## 30-second E2E results

All runs completed six windows with correct output length, finite samples, and
thermal status 0. `Core` is LiteRT model invocation time. `DSP` is STFT plus
iSTFT only. PSS is the final process sample after cleanup, not peak working-set
memory.

| Device | Model | DSP path | E2E | RTF | Core | DSP | Final PSS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| S25 | Official 6s | Kotlin | 17.83 s | 0.594 | 12.92 s | 1.953 s | 198.6 MiB |
| S25 | Official 6s | Native | 16.14 s | 0.538 | 12.89 s | 0.295 s | 201.7 MiB |
| S25 | Official 4s | Kotlin | 19.32 s | 0.644 | 15.51 s | 1.576 s | 176.5 MiB |
| S25 | Official 4s | Native | 17.79 s | 0.593 | 15.45 s | 0.228 s | 167.8 MiB |
| S25 | Guitar FT 6s | Kotlin | 18.25 s | 0.608 | 13.26 s | 2.011 s | 198.6 MiB |
| S25 | Guitar FT 6s | Native | 16.34 s | 0.545 | 13.12 s | 0.288 s | 202.2 MiB |
| S10 | Official 6s | Kotlin | 41.11 s | 1.370 | 29.93 s | 9.657 s | 394.2 MiB |
| S10 | Official 6s | Native | 32.30 s | 1.077 | 29.88 s | 0.806 s | 395.7 MiB |
| S10 | Official 4s | Kotlin | 46.08 s | 1.536 | 38.31 s | 6.472 s | 409.0 MiB |
| S10 | Official 4s | Native | 38.45 s | 1.282 | 36.63 s | 0.706 s | 407.7 MiB |
| S10 | Guitar FT 6s | Kotlin | 46.77 s | 1.559 | 32.37 s | 12.302 s | 393.5 MiB |
| S10 | Guitar FT 6s | Native | 33.49 s | 1.116 | 31.17 s | 0.825 s | 397.4 MiB |

| Device | Official 6s E2E gain | Official 4s E2E gain | Guitar FT E2E gain |
| --- | ---: | ---: | ---: |
| S25 | 9.5% | 7.9% | 10.5% |
| S10 | 21.4% | 16.6% | 28.4% |

Official 6s core time changes by only 0.3% on S25 and 0.2% on S10 between the
two DSP paths. This makes the native DSP, rather than inference fluctuation,
the primary cause of the paired E2E improvement. Guitar-ft uses the same DSP
ABI as official 6s; its separate identity is required for weight parity,
quality evaluation, cache keys, and licensing, not for a second DSP
implementation.

Official 4s is slower than official 6s by 20.1% in native E2E on S25 and 19.1%
on S10. Its native postprocess is cheaper because it reconstructs eight rather
than twelve source/channel planes, but its larger and slower neural core more
than removes that saving.

## Full-audio numerical gate

The complete 30-second WAV outputs were compared stem by stem after OLA and
PCM16 encoding:

| Comparison | Official 6s | Official 4s | Guitar FT 6s | Maximum PCM difference |
| --- | ---: | ---: | ---: | ---: |
| S25 native vs Kotlin | 98.69 dB | 99.47 dB | 98.26 dB | 1 LSB |
| S10 native vs Kotlin | 98.90 dB | 99.69 dB | 98.47 dB | 1 LSB |
| S10 native vs S25 native | 98.94 dB | 99.94 dB | 98.85 dB | 1 LSB |

This passes the existing 80 dB threshold and the intended maximum one-LSB PCM
gate. It qualifies the packed-real arithmetic for further integration work;
it does not re-evaluate separation quality between different weights.
Guitar-ft quality remains a separate research profile and must compare
guitar-ft Torch with guitar-ft LiteRT rather than with official 6s.

## S25 180-second completion run

| Model | Windows | E2E | RTF | Core | STFT | iSTFT | Final thermal status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official 6s | 31 | 112.44 s | 0.625 | 87.27 s | 0.70 s | 1.37 s | 1 |
| Official 4s | 31 | 131.42 s | 0.730 | 112.29 s | 0.75 s | 1.06 s | 2 |
| Guitar FT 6s | 31 | 112.03 s | 0.622 | 87.45 s | 0.68 s | 1.34 s | 2 |

All three runs completed with finite output and the correct length. They are
completion and thermal-pressure evidence only. They are not clean stable-
throughput results because final thermal status was nonzero, runs were not
paired with a Kotlin 180-second control, and the device had returned to status
0 by the later observation.

## Product interpretation and next work

The experiment supports one constrained `DemucsDspPlan` implementation with
two validated output profiles: four-stem and six-stem. Official 6s and
guitar-ft should share the six-stem DSP path while retaining different model
IDs, hashes, quality gates, cache keys, and license states. Official 4s must
retain its independent tensor and stem contract.

Before product integration, the next engineering batch should expose or map
LiteRT output buffers directly to native postprocessing, removing the current
66/44 MiB frequency and 16/11 MiB time-output Java-array path per window. It
should then repeat 30- and 180-second tests with peak PSS/native heap/ART
allocation and thermal-controlled paired order. The existing 180-second result
does not justify claiming sustained RTF below 1 under product thermal policy.

Booming SS integration, dynamic stem playback/cache manifests, cancellation,
seek, process-death recovery, and model-switch lifecycle tests remain outside
this branch by design.

## Verification and evidence

Build and host checks:

```text
./gradlew :app:testStandardDebugUnitTest
./gradlew :app:assembleStandardDebug
./gradlew :app:assembleStandardDebugAndroidTest
```

Device evidence, including synthetic reports, E2E JSON, PCM comparisons, and
audio outputs:

```text
outputs/android-benchmark/htdemucs-native-dsp-contracts-20260808/
```
