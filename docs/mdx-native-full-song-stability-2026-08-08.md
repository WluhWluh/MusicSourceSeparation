# MDX native DSP full-song and stability gate

## Full-song scope

The canonical Coast Town input is 273.699093 seconds, 44.1 kHz stereo PCM16,
12,070,130 frames, and 48 MDX windows with 47 joins. This stage uses the
device-preferred research profiles established by the short matrix:

- S25: bounded OpenCL FP32 plus `native-packed`;
- S10: bounded OpenCL FP32 plus `native-full`.

Both runs used the same `uvr_mdxnet_3_9662@2` artifact and contract. The
instrumentation performs decode, context-window extraction, native STFT,
LiteRT invoke, native iSTFT/OLA, contract scale and residual, center crop, PCM16
quantization, WAV output, hashing, memory sampling, thermal evidence, and
bounded-runtime dispatch evidence. No Booming SS code is involved.

## Performance

| Device | Profile | Windows | Processing | RTF | STFT total | iSTFT/OLA total | Invoke total | PSS | Thermal | Dispatch/wait |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| S25 | native-packed | 48 | 16.86 s | 0.0616 | 0.502 s | 0.903 s | 14.180 s | 229 MiB | 0 / 0 | 6096 / 6096 |
| S10 | native-full | 48 | 144.67 s | 0.5286 | 2.081 s | 2.929 s | 137.979 s | 181 MiB | 0 / 0 | 6096 / 6096 |

The reported processing RTF excludes instrumentation process startup but
includes all 48 window pipeline work and WAV writes.

## Numerical and join gate

S25 native-packed is compared with the prior same-device Kotlin packed-real
full-song output. S10 native-full is compared with the prior same-device Kotlin
parallel full-complex output. PCM16 quantization uses the same `roundToInt`
semantics in both pipelines.

| Device | Stem | Full-song SNR | Join-neighborhood SNR | Max PCM delta | Max join-step delta |
| --- | --- | ---: | ---: | ---: | ---: |
| S25 | vocals | 95.44 dB | 95.42 dB | 1 LSB | 0 LSB |
| S25 | instrumental | 104.87 dB | 104.99 dB | 1 LSB | 0 LSB |
| S10 | vocals | 95.56 dB | 95.49 dB | 1 LSB | 0 LSB |
| S10 | instrumental | 104.99 dB | 105.26 dB | 1 LSB | 0 LSB |

Both device reports pass the 80 dB / `1e-3` gate. All 47 joins are covered by
2,048-frame neighborhoods. No boundary step changes.

## One-hundred-window stability

The 9662 bounded-GPU profiles then processed 100 deterministic rotating
waveforms. PSS was sampled every ten windows.

| Device | Profile | Finite failures | Total P50 / P95 | First / last PSS | PSS delta | Thermal | Dispatch/wait |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| S25 | native-packed | 0 | 348.6 / 370.1 ms | 521.4 / 521.6 MiB | +0.2 MiB | 0 / 0 | 12,700 / 12,700 |
| S10 | native-full | 0 | 2566.6 / 2583.0 ms | 509.1 / 508.9 MiB | -0.2 MiB | 0 / 0 | 12,700 / 12,700 |

There is no sustained PSS growth, non-finite output, thermal escalation, or
missing GPU dispatch in this stability batch.

## Final research decision

The experiment repository qualifies `native-packed` as the S25 performance
profile and `native-full` as the S10 performance profile for the tested MDX
contracts. Native execution remains a numerical-parity profile, not a
bit-exact replacement for JTransforms. This work intentionally stops before
Booming SS application lifecycle, playback, seek, cancellation, and UI tests.

Raw reports and WAVs remain under ignored paths:

- `outputs/android-benchmark/mdx-native-full-song-2026-08-08/`
- `outputs/android-benchmark/mdx-native-stability-2026-08-08/`
- `outputs/mdx-native-full-song-s25-compare.json`
- `outputs/mdx-native-full-song-s10-compare.json`
