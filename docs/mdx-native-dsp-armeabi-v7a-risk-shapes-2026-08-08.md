# MDX native DSP armeabi-v7a risk-shape gate

## Scope and corrected identity

The Galaxy S10 supports both 64-bit and 32-bit zygotes. This experiment used an
armeabi-v7a-only APK to force the application into a 32-bit process, then ran
the four risk shapes with two and four workers. Each shape/worker pair compared
Kotlin/JTransforms and native packed-real with two warmups and five measured
rounds.

The first diagnostic run exposed a reporting bug: the summary used the first
device-supported ABI rather than the active process ABI. Commit `9ac916c`
records `processAbi` from process bitness and the matching supported-ABI list.
The corrected r2 campaign below is the canonical evidence; the earlier
`mdx-dsp-risk4-armeabi-v7a-v3` run is retained only as diagnostic history.

| Field | Value |
| --- | --- |
| Campaign | `mdx-dsp-risk4-armeabi-v7a-v3-r2` |
| Bundle ID | `758b25164c16105f17420ece54368f45a76464203668c4904d128c6a52991d88` |
| Source commit | `9ac916cd015b6957ed98e2824a59e291306adc8e` |
| Source dirty | `false` |
| Summary / identity ABI | `armeabi-v7a` / `armeabi-v7a` |
| Process bitness | 32-bit |
| APK bytes | `95,056,609` |
| APK SHA-256 | `b78fc275af3fae83107c58b807841fc350b6df7ece5d045e7be03adc658bcce3` |

The frozen APK and checksum are under
`C:\Users\User\Documents\BSSUploadRelay\app-live-apks\` with prefix
`BSS-MDX-DSP-Risk4-v3-armeabi-v7a-758b2516`. Relay-verified results are under
`C:\Users\User\Documents\BSSUploadRelay\results\mdx-dsp-risk4-armeabi-v7a-v3-r2\`.

## Numerical and performance results

The strict merger accepted one run and 16 rows; all 16 qualified. Lowest packed
STFT/iSTFT SNR was 133.611/137.582 dB. Largest STFT/iSTFT absolute error was
`9.1553e-5`/`7.4506e-8`. Every output was finite.

| Shape | Packed 2 workers | Packed 4 workers | 4w / 2w speedup | Packed vs Kotlin, 4w |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 2048 / 128 | 21.25 ms | 21.37 ms | 0.99x | 8.37x |
| 4096 / 2048 / 512 | 111.34 ms | 94.58 ms | 1.18x | 9.53x |
| 6144 / 3072 / 512 | 176.17 ms | 148.56 ms | 1.19x | 5.89x |
| 16384 / 2048 / 512 | 296.07 ms | 237.55 ms | 1.25x | 9.60x |

Four workers are effectively tied on the 128-frame shape and win on all three
longer or larger shapes. The existing four-worker product policy therefore
does not need an arm32-specific branch.

The matrix took 172.146 seconds and thermal status stayed `0/0`. Process PSS
rose 111.29 MiB. Native heap ended 9.33 MiB below its start snapshot after all
plans were released, so the run has no native-workspace retention signal. One
two-worker shape-run reported a blocking GC; no four-worker shape-run did.

## Decision

Native packed-real itself is compatible with armeabi-v7a, including the largest
24 MiB tensor and 16384-point FFT shapes. This is sufficient for a DSP fallback
or research profile, but not for admitting the full 30-model LiteRT matrix on
32-bit devices. Model allocation, invoke, complete-song memory, and low-memory
recovery remain mandatory because a 32-bit process has substantially less
address-space headroom. ARM64 remains the recommended product ABI.
