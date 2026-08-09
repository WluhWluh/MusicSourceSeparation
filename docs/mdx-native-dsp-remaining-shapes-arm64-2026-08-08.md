# MDX native DSP remaining-shape ARM64 gate

## Scope and identity

Batch 1 qualifies the ten MDX DSP shapes not covered by the earlier 81-run
App Live campaign. The S25 and S10 ran the same ARM64 APK and contract-v3
bundle. Each shape used four workers, two warmups, and five measured rounds in
alternating Kotlin/native-packed order. No model invocation or native-full
profile was included.

| Field | Value |
| --- | --- |
| Campaign | `mdx-dsp-shape-remaining10-arm64-v3` |
| Bundle ID | `6db226ab6ef70cced6ebcffa061d60fec3cd9d99f00c791e90b0d7aece10feae` |
| Source commit | `0615b72e704aee1a08e528e888ee586bb4457e1d` |
| Source dirty | `false` |
| APK SHA-256 | `24bcd17d6e9c191a95973307293e9e18f6422a489f4d0ddcdde2d6e8b0ad77bb` |
| APK bytes | `97,889,227` |
| Native DSP SHA-256 | `e53313d906ee5e1bcd9dc215e133bd4c3cde367c773ba0c874d421a9429767a6` |
| ABI | `arm64-v8a` |

Both terminal runs were downloaded and verified from the relay without remote
cleanup. The retained merged CSV and JSON are under
`C:\Users\User\Documents\BSSUploadRelay\results\mdx-dsp-shape-remaining10-arm64-v3\`.
The strict contract-v3 merger accepted two runs and 40 rows; all 40 rows
qualified.

## Numerical gate

Every Kotlin and native-packed output was finite. The worst native-packed
results remain well inside the frozen 80 dB SNR and `1e-3` absolute-error
gates.

| Metric | Worst result |
| --- | ---: |
| STFT SNR | 133.7211836 dB |
| iSTFT SNR | 137.4371860 dB |
| STFT maximum absolute error | `9.1552734375e-5` |
| iSTFT maximum absolute error | `6.7055225372e-8` |

## Device performance

The table gives combined STFT+iSTFT medians and Kotlin/native-packed speedup.
All ten shapes favor native-packed on both devices.

| Shape | S25 packed | S25 speedup | S10 packed | S10 speedup |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 2048 / 128 | 7.31 ms | 4.29x | 16.00 ms | 4.47x |
| 4096 / 2048 / 512 | 27.78 ms | 5.67x | 105.75 ms | 2.98x |
| 5120 / 2048 / 256 | 14.91 ms | 7.86x | 60.37 ms | 4.52x |
| 6144 / 3072 / 256 | 22.91 ms | 7.77x | 74.83 ms | 4.51x |
| 6144 / 2048 / 512 | 36.87 ms | 7.13x | 126.54 ms | 4.35x |
| 6144 / 3072 / 512 | 50.44 ms | 4.65x | 170.75 ms | 3.93x |
| 8192 / 2048 / 256 | 33.17 ms | 3.60x | 69.19 ms | 9.47x |
| 8192 / 2048 / 512 | 83.70 ms | 4.93x | 136.63 ms | 8.91x |
| 16384 / 2048 / 256 | 76.78 ms | 3.97x | 109.88 ms | 5.13x |
| 16384 / 2048 / 512 | 99.40 ms | 4.06x | 226.28 ms | 9.52x |

S25 completed the matrix in 51.747 seconds and S10 in 138.416 seconds. Thermal
status remained `0/0`; battery temperature rose 2.9 C on S25 and 0.9 C on S10.
Process PSS rose 74.2 MiB and 77.0 MiB respectively, while native-heap growth
was only 0.60 MiB and 0.33 MiB. The diagnostic process runs Kotlin and native
for every shape, so its PSS delta is not a packed-only retained-memory result.
Three shape-runs reported blocking GC in total and cannot assign it to one
profile.

## Decision

The shared packed-real implementation is numerically and operationally valid
for all 13 known MDX DSP shapes on the two product ARM64 controls. The ten new
shapes do not require a separate FFT implementation or shape-specific runtime
selection. Model inference, complete-song joins, and product lifecycle remain
separate qualification layers.

The four risk sentinels for the next broad-device package remain
4096/2048/128, 4096/2048/512, 6144/3072/512, and 16384/2048/512. They cover
thread overhead, long time axes, the largest tensor, and the largest FFT.
