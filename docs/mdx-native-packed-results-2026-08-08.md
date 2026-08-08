# NativeMdxDsp packed-real results

## Numerical gate

The packed profile uses pocketfft `r2c/c2r` with the same native plan,
workspaces, NHWC contract, and worker policy as native full-complex. Against
native full-complex, all three sentinels passed the numerical gate:

| Model | STFT SNR | STFT max abs | iSTFT SNR | iSTFT max abs |
| --- | ---: | ---: | ---: | ---: |
| 9662 | 136.645 dB | `3.05e-5` | 143.044 dB | `2.98e-8` |
| Kim Inst | 135.603 dB | `3.05e-5` | 142.326 dB | `3.73e-8` |
| HQ4 | 135.577 dB | `1.53e-5` | 142.895 dB | `4.47e-8` |

The 80 dB SNR and `1e-3` maximum-absolute-error gates pass. Packed and full
remain non-bit-exact, as expected for distinct FFT algorithms.

## Device performance

The table reports packed speedup relative to native full-complex. Values above
1.0 are faster. The matrix used three sessions, two warmups, and ten measured
windows per model/backend. Four replacement groups required cooling-controlled
reruns; S10 HQ4 CPU still ended at thermal 1 in each ten-window session and is
marked thermal-limited.

| Device | Model | Backend | STFT speedup | iSTFT speedup | Combined DSP |
| --- | --- | --- | ---: | ---: | ---: |
| S25 | 9662 | CPU / GPU | 1.32x / 1.22x | 1.52x / 1.41x | 1.44x / 1.34x |
| S25 | Kim Inst | CPU / GPU | 1.22x / 1.33x | 1.46x / 1.34x | 1.36x / 1.33x |
| S25 | HQ4 | CPU / GPU | 1.27x / 1.15x | 1.43x / 1.29x | 1.36x / 1.24x |
| S10 | 9662 | CPU / GPU | 1.04x / 0.72x | 0.87x / 0.78x | 0.93x / 0.75x |
| S10 | Kim Inst | CPU / GPU | 0.89x / 0.96x | 0.77x / 0.88x | 0.81x / 0.91x |
| S10 | HQ4 | CPU / GPU | 1.24x / 1.13x | 1.02x / 0.95x | 1.10x / 1.02x |

## Decision

Packed real is a valid S25 optimization, improving every S25 group by
1.24-1.44x over native full-complex. It is not a universal native policy.
On S10, retain native full-complex for 9662 and Kim; HQ4's small packed gain is
not sufficient to justify a separate product policy in this research stage.

Full-song and join comparisons must use device-specific preferred profiles:
S25 native-packed and S10 native-full. The thermal-limited S10 HQ4 CPU packed
results are retained as resource evidence rather than a clean performance
qualification.

Raw reports remain under ignored
`outputs/android-benchmark/mdx-native-packed-2026-08-08/`.
