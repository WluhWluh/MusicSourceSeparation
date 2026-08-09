# MDX native DSP x86 risk-shape gate

## Scope and API 26 correction

The pure 32-bit x86 API 26 emulator ran three DSP sentinels: the 9662 control,
the largest tensor, and the largest FFT. Each compared four-worker
Kotlin/JTransforms with native packed-real using two warmups and five measured
rounds.

The first attempt stopped before DSP execution because identity collection
called the API 28 `PackageInfo.longVersionCode` method. Commit `2c57145` adds
the API 26 `versionCode` fallback. The corrected r2 campaign is canonical; the
initial `mdx-dsp-x86-3-v3` attempt is retained as failure-stage evidence.

| Field | Value |
| --- | --- |
| Campaign | `mdx-dsp-x86-3-v3-r2` |
| Bundle ID | `a1952e5292a9628c2da93b63e713feba2cfe79ecb685b7c8a880f54c8d351d21` |
| Source commit | `2c57145ca2e4de4fa15f593baf342b4ba61e19dd` |
| Source dirty | `false` |
| Summary / identity ABI | `x86` / `x86` |
| Process bitness | 32-bit |
| APK bytes | `100,046,939` |
| APK SHA-256 | `f6689f47566a392740d9eeff9ce429267b225ad47d8736ed63b0b46dfc8af830` |

The frozen APK and checksum are under
`C:\Users\User\Documents\BSSUploadRelay\app-live-apks\` with prefix
`BSS-MDX-DSP-X86-3-v3-a1952e52`. Relay-verified results are under
`C:\Users\User\Documents\BSSUploadRelay\results\mdx-dsp-x86-3-v3-r2\`.

## Results

The strict merger accepted one run and six rows; all six qualified. Every
output was finite.

| Shape | Packed median | Packed P95 | Speedup vs Kotlin |
| --- | ---: | ---: | ---: |
| 6144 / 2048 / 256 | 23.45 ms | 64.72 ms | 38.89x |
| 6144 / 3072 / 512 | 53.49 ms | 61.20 ms | 35.93x |
| 16384 / 2048 / 512 | 97.60 ms | 105.35 ms | 16.04x |

Lowest packed STFT/iSTFT SNR was 134.429/137.569 dB. Largest STFT/iSTFT
absolute error was `9.1553e-5`/`7.4506e-8`. The matrix took 49.815 seconds,
process PSS rose 71.49 MiB, and native heap rose 1.10 MiB. API 26 reports
thermal status as unavailable (`-1`). One shape-run recorded a blocking GC.

The large Kotlin/native ratios are emulator-specific and are not mobile-device
performance evidence.

## Decision

Native packed-real is functionally suitable for legacy x86 emulator
compatibility, including the largest DSP tensor and FFT shapes. This does not
qualify the corresponding LiteRT models: the existing 2 GiB x86 model test
still fails HQ4 during delegated tensor allocation. Keep x86 as a test-only
compatibility ABI, prefer x86_64 for CI, and do not expose high-memory model
support solely because its DSP stage passes.
