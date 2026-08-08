# MDX Kotlin DSP short baseline (2026-08-08)

## Scope

This Batch 1 baseline freezes the current Kotlin/JTransforms MDX DSP path before
native DSP experiments. It uses three contract-v2 sentinels from the
`bss-tflite` publication repository:

| Sentinel | Model SHA-256 | Contract SHA-256 | DSP contract |
| --- | --- | --- | --- |
| UVR-MDX-NET 3 9662 | `f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378` | `edf02de52bb45c842ad65a4be8f2118ed6d212ec4590a81ae0401e760bfb4fe9` | FFT 6144, hop 1024, 2048 x 256 |
| Kim Inst | `fd3ca5bcb6568d893be5f049de8804278ee669303749d6fb3010bbeaeae4d288` | `d7f85d042c1e702c525dff055cce41496b2fcb5efc61b709535e74f288e616a6` | FFT 7680, hop 1024, 3072 x 256 |
| UVR-MDX-NET Inst HQ 4 | `5f091562bd0297ff2223015a219d12cc1136a9d54df975680ff4eb239629c742` | `aa441f7f801849929a1d02cd3adcc5bf121df0c978974d074dee50e3fca03ef6` | FFT 5120, hop 1024, 2560 x 256 |

Runtime: `litert-android-2.1.5-bss.2`, AAR SHA-256
`88cd2f7eaf1443d1c570085b1c24f239db87eb24c788a590adf5158e17443d0e`.
The installed benchmark APK SHA-256 was
`ecd98945d03d91eeb19cabe82d28d61d429dd24309afc456d89db2ccb909a1f6`.

Each row below aggregates three independent CompiledModel sessions. Each
session used two warmups followed by ten measured windows, four CPU threads,
the same deterministic stereo waveform, and a contract-derived periodic Hann
DSP plan. GPU means bounded OpenCL FP32 with N=1 queue/command preparation;
every GPU session reported positive dispatch and matching event-wait counts.

## Results

Times are milliseconds per window except setup. P50/P95 pool all 30 measured
windows. PSS is the largest end-of-session total PSS. Allocation and GC are ART
deltas over ten measured windows.

| Device | Model | Backend | Setup mean | Total P50/P95 | STFT | Layout in/out | Invoke | Read | iSTFT+OLA | Residual | PCM16 | DSP share | Max PSS | Alloc/session | GC/session |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S25 | 9662 | CPU | 299 | 1571 / 1618 | 98 | 5 / 6 | 1288 | 3 | 163 | 2 | 7 | 16.6% | 774 MiB | 826 MiB | 20 |
| S25 | 9662 | bounded GPU | 864 | 601 / 632 | 101 | 5 / 9 | 304 | 9 | 163 | 2 | 8 | 43.9% | 532 MiB | 826 MiB | 19 |
| S25 | Kim Inst | CPU | 509 | 4905 / 5019 | 145 | 7 / 9 | 4477 | 4 | 249 | 2 | 7 | 8.0% | 1384 MiB | 1067 MiB | 20 |
| S25 | Kim Inst | bounded GPU | 1425 | 1585 / 1627 | 146 | 7 / 13 | 1137 | 11 | 264 | 2 | 8 | 25.8% | 1019 MiB | 1067 MiB | 20 |
| S25 | HQ4 | CPU | 514 | 4185 / 4663 | 112 | 6 / 9 | 3915 | 4 | 190 | 2 | 8 | 7.1% | 1164 MiB | 846 MiB | 20 |
| S25 | HQ4 | bounded GPU | 1476 | 1084 / 1102 | 104 | 7 / 11 | 766 | 11 | 164 | 2 | 8 | 25.0% | 893 MiB | 846 MiB | 20 |
| S10 | 9662 | CPU | 398 | 3313 / 3809 | 112 | 9 / 8 | 2589 | 4 | 672 | 3 | 10 | 23.0% | 760 MiB | 826 MiB | 40 |
| S10 | 9662 | bounded GPU | 1095 | 2911 / 2971 | 107 | 9 / 14 | 2277 | 7 | 491 | 2 | 9 | 20.5% | 412 MiB | 826 MiB | 40 |
| S10 | Kim Inst | CPU | 964 | 10787 / 11165 | 186 | 16 / 14 | 9799 | 6 | 712 | 2 | 10 | 8.4% | 1360 MiB | 1066 MiB | 40 |
| S10 | Kim Inst | bounded GPU | 2020 | 6516 / 7056 | 153 | 14 / 18 | 5743 | 12 | 684 | 2 | 9 | 12.6% | 799 MiB | 1066 MiB | 40 |
| S10 | HQ4 | CPU | 725 | 8407 / 8629 | 121 | 12 / 11 | 7684 | 5 | 499 | 2 | 11 | 7.4% | 1152 MiB | 845 MiB | 40 |
| S10 | HQ4 | bounded GPU | 2038 | 4826 / 5141 | 110 | 11 / 23 | 3842 | 12 | 796 | 4 | 13 | 18.8% | 709 MiB | 846 MiB | 40 |

GPU dispatch counts were 1270 per 9662 session and 1390 per Kim/HQ4 session;
event-wait counts matched exactly. All 360 measured windows produced finite
output. No CPU fallback result is used as a GPU result.

## Interpretation

The current DSP is not the dominant cost on CPU for Kim or HQ4. It becomes a
material target on the faster S25 GPU path: STFT plus combined iSTFT/OLA is
43.9% of 9662, 25.8% of Kim, and 25.0% of HQ4 window time. On S10 GPU the same
share is 12.6-20.5%. A parity-preserving native/parallel DSP experiment should
therefore start with 9662 on S25 GPU, then use Kim and HQ4 as memory and FFT
shape guards.

The current implementation allocates roughly 0.8-1.1 GiB over ten windows and
causes 19-40 ART GCs. Workspace reuse is a separate high-value target even
where inference dominates, because it should reduce P95 and memory pressure.

## Limitations

`MdxSpectrogram.tensorToWaveform()` combines inverse FFT, overlap-add, window
normalization, and trim; this baseline reports them as `iSTFT+OLA` rather than
inventing a split. The synthetic fixture measures repeatable kernel and memory
cost, not full-song decode, joins, WAV I/O, or separation quality. Thermal
status was observed as 0 during device execution, but the first version of the
harness did not persist per-session thermal samples; strict thermal regression
runs must add that evidence before comparing small percentage changes.

Raw device reports are retained under the ignored path
`outputs/android-benchmark/mdx-native-dsp-baseline-2026-08-08-final/`.
