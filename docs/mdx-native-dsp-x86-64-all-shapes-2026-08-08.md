# MDX native DSP x86_64 all-shape gate

## Scope and identity

The contract-v3 runner executed all 13 MDX DSP shapes in a 64-bit x86 Android
emulator. Each shape compared four-worker Kotlin/JTransforms and native
packed-real using two warmups and five measured rounds. This is a correctness,
memory, and lifecycle gate; emulator latency is not mobile-device product
performance evidence.

| Field | Value |
| --- | --- |
| Campaign | `mdx-dsp-all13-x86_64-v3` |
| Bundle ID | `60f0e7ecf94609b58ff37b8cca3269947081716c94f74e9b6f5cf0b3060ef7cf` |
| Source commit | `cd6b420fc1cb895564ec878f49afb3654af9c01b` |
| Source dirty | `false` |
| ABI / process | `x86_64` / 64-bit |
| APK bytes | `101,345,657` |
| APK SHA-256 | `aae97e3d75fd41561a1fb610945bff290a51dfe5365f110c5f7ad95e3cb90eaf` |

The frozen APK and checksum are retained under
`C:\Users\User\Documents\BSSUploadRelay\app-live-apks\` with prefix
`BSS-MDX-DSP-All13-v3-x86_64-60f0e7ec`. The terminal run was downloaded and
SHA-verified from the relay without cleanup. Merged evidence is under
`C:\Users\User\Documents\BSSUploadRelay\results\mdx-dsp-all13-x86_64-v3\`.

## Results

The strict merger accepted one run and 26 rows; all 26 qualified. Every output
was finite.

| Metric | Result |
| --- | ---: |
| Lowest packed STFT SNR | 133.6109179 dB |
| Lowest packed iSTFT SNR | 136.8495527 dB |
| Largest STFT absolute error | `9.1552734375e-5` |
| Largest iSTFT absolute error | `7.4505805969e-8` |
| Matrix elapsed | 211.236 s |
| Process PSS delta | +73.61 MiB |
| Native heap delta | +0.27 MiB |
| Thermal status | 0 / 0 |
| Blocking-GC shape-runs | 2 / 13 |

Native-packed was faster than Kotlin for every shape. The within-emulator
speedup ranged from 15.48x to 36.80x with a 23.64x median. These large ratios
reflect x86_64 AVD/JTransforms scheduling and cold-state behavior and must not
be compared with ARM device speedups.

## Decision

Native packed-real is suitable as the default MDX DSP path in x86_64 emulator
CI. All FFT, dimF, and 128/256/512-frame combinations load, execute, close, and
meet the shared numerical gate. No x86_64-specific algorithm or shape fallback
is needed. Product RTF claims remain restricted to physical ARM devices.
