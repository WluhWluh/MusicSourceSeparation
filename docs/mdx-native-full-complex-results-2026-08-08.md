# NativeMdxDsp full-complex results

## Implementation

`NativeMdxDsp` builds an immutable contract-derived plan for FFT size, hop,
frequency bins, frames, chunk length, Hann window, normalization envelope, and
worker count. It uses pinned Java float arrays, persistent native FFT/OLA
workspaces, four contiguous STFT frame lanes, and one iSTFT lane per channel.
Input and output tensors remain NHWC.

The mixed-radix FFT implementation is pocketfft at fixed revision
`c90e55b3d529f8efa40ed01a20de22405f45fc65`, included as a git submodule under
its BSD-3-Clause license. BandBuddy's native radix-2 FFT was not reused because
MDX requires FFT 6144, 7680, and 5120.

## Numerical gate

Native pocketfft and Kotlin JTransforms do not produce bit-exact transforms.
The strict bit-exact goal therefore failed and remains explicitly recorded.
The frozen numerical gate passed:

| Model | STFT SNR | STFT max abs | iSTFT SNR | iSTFT max abs | Bit-exact |
| --- | ---: | ---: | ---: | ---: | --- |
| 9662 | 134.990 dB | `3.05e-5` | 137.372 dB | `5.22e-8` | no |
| Kim Inst | 133.904 dB | `4.58e-5` | 136.794 dB | `5.96e-8` | no |
| HQ4 | 133.539 dB | `3.05e-5` | 137.297 dB | `4.47e-8` | no |

All values exceed 80 dB SNR and remain below `1e-3` max absolute error. This is
a numerical-pass/bit-exact-fail research profile, not a silent parity claim.

## Device results

Each row contains three thermal-0 sessions, two warmups, and ten measured
windows. DSP is STFT plus combined iSTFT/OLA. The reference is the Kotlin
four-worker reusable NHWC profile.

| Device | Model | Backend | Kotlin DSP | Native DSP | DSP speedup | Total change |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| S25 | 9662 | CPU | 127.3 ms | 31.5 ms | 4.05x | -2.9% |
| S25 | 9662 | bounded GPU | 130.9 ms | 40.5 ms | 3.23x | -22.8% |
| S25 | Kim Inst | CPU | 198.3 ms | 43.2 ms | 4.59x | -1.3% |
| S25 | Kim Inst | bounded GPU | 207.0 ms | 57.1 ms | 3.62x | -7.6% |
| S25 | HQ4 | CPU | 132.6 ms | 29.2 ms | 4.54x | -6.9% |
| S25 | HQ4 | bounded GPU | 138.8 ms | 40.3 ms | 3.44x | -11.9% |
| S10 | 9662 | CPU | 338.1 ms | 67.5 ms | 5.01x | +1.1% |
| S10 | 9662 | bounded GPU | 353.1 ms | 79.2 ms | 4.46x | -23.3% |
| S10 | Kim Inst | CPU | 449.6 ms | 106.0 ms | 4.24x | +8.1% |
| S10 | Kim Inst | bounded GPU | 474.9 ms | 115.8 ms | 4.10x | -18.8% |
| S10 | HQ4 | CPU | 361.0 ms | 109.4 ms | 3.30x | +7.6% |
| S10 | HQ4 | bounded GPU | 322.5 ms | 103.9 ms | 3.10x | -18.5% |

Positive total change denotes a slower full window. Those CPU regressions are
fully dominated by LiteRT invoke drift; the native DSP stage itself improved
in every group. Initial S25 HQ4 runs reached thermal 1 and were rejected; the
reported replacement batch contains six thermal-0 sessions.

## Decision

The native performance and numerical gates pass, while strict bit-exact parity
does not. Continue to a separate native packed-real experiment and retain
full-complex as its direct same-library reference. Full-song PCM and join gates
remain required before any broader qualification.

Raw reports remain under ignored
`outputs/android-benchmark/mdx-native-full-2026-08-08/`.
