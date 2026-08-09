# MDX App Live DSP matrix v2 batch results

## Decision

The 81-run App Live campaign supports one ARM64 product policy for the three
qualified MDX DSP shapes: use native pocketfft packed-real on every device. Do
not add a full-complex/packed-real device selector, startup benchmark, or SoC
allowlist. Keep native full-complex in research and diagnostic builds as a
numerical reference. A product build should fall back to Kotlin/JTransforms
only when the native library or packed plan cannot be used.

This supersedes the earlier small-matrix recommendation to use native
full-complex on the Galaxy S10. The isolated, balanced v2 campaign is broader
and more directly measures the DSP policy than the earlier model/backend
matrix.

## Frozen evidence

| Field | Value |
| --- | --- |
| Campaign | `app-live-mdx-dsp-matrix-v2` |
| Contract | `2` |
| Bundle ID | `eeccfead0c30e7022afdecace644e82bb8bbb2dda8a8e27988d4bb676ec7ebc0` |
| Source commit | `4a26d2850f97f98fa6fc4338249dedd425a03bda` |
| Source dirty | `false` |
| APK SHA-256 | `56d6499307ea9b42f7ad25f12fa0e859adf098121a5c50e99ec8ebcea4278a65` |
| Native DSP SHA-256 | `e53313d906ee5e1bcd9dc215e133bd4c3cde367c773ba0c874d421a9429767a6` |
| pocketfft revision | `c90e55b3d529f8efa40ed01a20de22405f45fc65` |
| Worker policy | four workers |
| Sampling | two warmups, ten measured samples per profile |

All 81 terminal runs were downloaded before analysis. The relay manifests and
SHA-256 values verified 648 uploaded files totaling 8,529,841 bytes. The
strict batch merger accepted 81 runs and 729 rows. All rows use the identity
above, report nine shape/profile rows per run, and pass the contract-v2 schema
and sample-count checks.

The retained local evidence is under
`C:\Users\User\Documents\BSSUploadRelay\results\app-live-mdx-dsp-matrix-v2\`.
The principal merged artifacts are `dsp-matrix-batch-summary.csv` and
`dsp-matrix-batch-summary.json`.

## Coverage

| Dimension | Coverage |
| --- | ---: |
| Terminal complete runs | 81 |
| Unique firmware fingerprints | 77 |
| Unique device models | 72 |
| Manufacturers | 9 |
| Reported SoC models | 37 |
| ABI | `arm64-v8a` |
| Android SDK | 29, 30, 31, 33, 34, 35, 36, 37 |

The manufacturer run counts are Samsung 35, Google 13, vivo 7, Motorola 5,
OnePlus 5, OPPO 5, Xiaomi 5, realme 4, and Huawei 2. The SoC set includes
multiple generations of Qualcomm Snapdragon, Google Tensor, Samsung Exynos,
MediaTek, and Kirin devices. Eight Android 10/11 runs cannot report
`Build.SOC_MODEL`; their firmware and device identities remain available.

Each run tests the three frozen shapes below with Kotlin/JTransforms, native
full-complex, and native packed-real in balanced cross-over order. The raw
performance population is therefore 243 run-shape comparisons. Repeated
firmware fingerprints are collapsed by the median for a second, device-equal
population of 231 comparisons.

| Sentinel | FFT | Hop | dimF | Frames |
| --- | ---: | ---: | ---: | ---: |
| UVR MDXNET 3 9662 | 6144 | 1024 | 2048 | 256 |
| Kim Inst | 7680 | 1024 | 3072 | 256 |
| UVR MDXNET Inst HQ4 | 5120 | 1024 | 2560 | 256 |

## Numerical qualification

All 729 rows qualify and all outputs are finite. Native execution is not
raw-float bit-exact with JTransforms, which is expected for a different FFT
algorithm, but every result remains far inside the frozen `80 dB` SNR and
`1e-3` maximum-absolute-error gates.

| Metric | Worst observed result |
| --- | ---: |
| Native STFT SNR | 133.5897525 dB |
| Native iSTFT SNR | 136.8223158 dB |
| STFT maximum absolute error | `3.0517578125e-5` |
| iSTFT maximum absolute error | `7.45058059692383e-8` |

These synthetic checks establish the conversion-level numerical behavior for
the three shapes. They do not replace complete-song PCM and join qualification
for a new shape or product pipeline.

## Packed-real versus full-complex

Packed-real has the lower combined STFT+iSTFT median in 240 of 243 raw
run-shape comparisons. It wins all three shapes in 78 of 81 runs; three runs
are mixed and no run uniformly favors full-complex.

The percentiles below give `native-full / native-packed`, so values greater
than one favor packed. Percentiles use the 77-fingerprint device-equal set;
winner counts retain every raw run.

| Shape | Packed median wins | P10 | P50 | P90 |
| --- | ---: | ---: | ---: | ---: |
| 9662 | 80 / 81 | 1.119x | 1.330x | 1.495x |
| Kim Inst | 81 / 81 | 1.141x | 1.316x | 1.450x |
| HQ4 | 79 / 81 | 1.124x | 1.285x | 1.462x |
| All shapes | 240 / 243 | 1.132x | 1.316x | 1.479x |

The packed/full median speedup decomposes to 1.223x for STFT and 1.408x for
iSTFT. The inverse transform is the larger and more consistent source of the
packed-real gain.

Packed-real also has the lower combined P95 in 225 of 243 raw comparisons and
215 of 231 device-equal comparisons. Its device-equal P95 speedup is 1.264x at
the median, with P10/P90 of 1.033x/1.564x. Ten timed samples make an individual
P95 sensitive to scheduling outliers, so it is supporting rather than
device-policy evidence.

### Median exceptions

Only three raw comparisons favor full-complex by the median. All three favor
packed-real by P95, and none reproduces as a SoC-wide pattern.

| Device / shape | Full median | Packed median | Full/packed | Full P95 | Packed P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Galaxy S10, HQ4, repeat 2 | 60.894 ms | 61.957 ms | 0.983x | 70.313 ms | 69.206 ms |
| Galaxy S10, HQ4, repeat 3 | 92.623 ms | 100.114 ms | 0.925x | 136.502 ms | 113.721 ms |
| Pixel 9 Pro, 9662 | 61.882 ms | 62.546 ms | 0.989x | 85.141 ms | 69.471 ms |

After firmware-fingerprint deduplication, only Galaxy S10/HQ4 at 0.983x and
Pixel 9 Pro/9662 at 0.989x remain below one. A selector would add a permanent
runtime and qualification branch to recover approximately one percent on
these medians while worsening their observed tail latency. The evidence does
not justify that complexity.

Manufacturer-level device-equal medians all favor packed-real. They range from
1.178x for Huawei to 1.385x for OnePlus; Samsung is 1.366x and Google is
1.216x. Multi-device SoC groups also favor packed, including SM8550 at 1.405x,
SM8650 at 1.410x, SM8750 at 1.315x, Tensor G4 at 1.326x, Exynos s5e9925 at
1.414x, and MediaTek MT6878 at 1.299x.

## Native packed-real versus Kotlin

The main product gain comes from moving the shared MDX DSP out of the current
Kotlin/JTransforms implementation. Packed-real is 3.829x faster than Kotlin at
the device-equal combined-DSP median.

| Shape | Packed P10 | Packed P50 | Packed P90 | Median speedup vs Kotlin |
| --- | ---: | ---: | ---: | ---: |
| 9662 | 25.66 ms | 55.76 ms | 113.15 ms | 4.063x |
| Kim Inst | 30.89 ms | 70.16 ms | 138.91 ms | 3.659x |
| HQ4 | 23.89 ms | 55.34 ms | 111.45 ms | 3.611x |

Using 48 windows as the canonical complete-song scale gives the following
device-equal median DSP-only savings. These values exclude model invocation,
decode, PCM conversion, and writes.

| Shape | Packed vs Kotlin | Packed vs native-full |
| --- | ---: | ---: |
| 9662 | 7.016 s | 0.714 s |
| Kim Inst | 7.914 s | 0.949 s |
| HQ4 | 5.991 s | 0.617 s |

The approximately 3.8x DSP speedup must not be presented as a 3.8x product RTF
speedup. In the prior 9662 complete-song runs, model invocation accounts for
14.180 of 16.86 seconds on S25 and 137.979 of 144.67 seconds on S10. Faster
model backends receive a larger end-to-end benefit from native DSP; on slower
devices, model invocation remains dominant.

## Galaxy S10 reassessment

The earlier three-session model/backend matrix favored native full-complex for
9662 and Kim on S10. That experiment mixed DSP timing with separate model and
backend sessions and required cooling-controlled replacements. It was useful
for initial exploration but is not the appropriate basis for a universal DSP
algorithm policy.

The final v2 artifact ran three times on the same S10 firmware. Packed-real
speedup ranges are:

| Shape | Full/packed median range |
| --- | ---: |
| 9662 | 1.080x to 1.132x |
| Kim Inst | 1.048x to 1.091x |
| HQ4 | 0.925x to 1.020x |

9662 and Kim now consistently favor packed. HQ4 is noisy around parity and
does not compensate for a device-specific branch. The earlier S10 full-complex
recommendation is therefore retired. S10 still needs a packed-real complete
song and 100-window qualification because its existing long run used
native-full.

## Resource and stability evidence

| Metric | Result |
| --- | ---: |
| Runs with nonzero thermal status | 0 / 81 |
| Battery temperature delta, median / P90 / max | +0.1 / +1.0 / +2.6 C |
| Process PSS delta, median / P90 / max | +56,294 / +85,807 / +133,804 KiB |
| Native heap delta, median / P90 / max | +438,736 / +1,212,704 / +2,703,456 bytes |
| Shape-runs with blocking GC | 18 / 243 |

The PSS and GC values describe a diagnostic APK that runs Kotlin, native-full,
and native-packed for all three shapes in one process. They cannot be assigned
to packed-real alone and are not a product retained-memory estimate. The much
smaller native-heap delta provides no immediate native workspace leak signal.
The earlier S25 packed-real 100-window run remained finite with a +0.2 MiB PSS
delta, but this short App Live campaign does not independently qualify
sustained thermal behavior on every device.

## Product policy and remaining gates

1. Use one immutable contract-derived `NativeMdxDsp` packed-real plan for the
   supported ARM64 product path. Reuse its native workspaces for the session.
2. Do not ship a performance selector between native-full and native-packed.
   Keep native-full as a research/debug parity profile, not a product policy.
3. Retain Kotlin/JTransforms as the conservative fallback for native load,
   plan creation, or execution failure and for ABIs not qualified here.
4. Qualify every remaining MDX DSP contract shape independently. This campaign
   covers three 256-frame sentinels, not the matrix's 128/512-frame and other
   FFT profiles. Shared code compatibility does not transfer numerical or
   resource qualification to an untested shape.
5. Before product integration, run S10 packed-real through the canonical
   48-window song, all joins, PCM comparison, and 100-window stability gate.
6. After integration, measure the actual product pipeline and lifecycle. App
   Live establishes portable DSP behavior but does not cover playback,
   prebuffering, seek, cancellation, foreground/background transitions, or
   low-memory recovery.

Further App Live testing of these same three shapes has low expected value.
The next broad-device package should add the currently unqualified DSP shapes
or test a packed-only product-like retained-memory and sustained workload.
