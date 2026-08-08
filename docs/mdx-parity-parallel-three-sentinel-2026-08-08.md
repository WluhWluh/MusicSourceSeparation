# MDX parity-preserving parallel DSP: three-sentinel gate

## Scope

This stage applies the previously qualified full-complex parallel policy to the
three native-DSP sentinels: 9662 (FFT 6144), Kim Inst (FFT 7680), and HQ4
(FFT 5120). STFT uses four contiguous frame lanes. iSTFT uses one lane per
stereo channel and preserves frame accumulation order within each channel.

The device matrix used S25 and S10, LiteRT `2.1.5-bss.2`, CPU4 and bounded
OpenCL FP32, three independent sessions, two warmups, and ten measured windows.
All 36 parallel reports completed, produced finite output, and all GPU reports
contained positive bounded-runtime dispatch evidence.

## Numerical gate

The JVM gate runs the complete 256-frame shape for all three FFT contracts.
Serial and four-worker STFT tensors were bit-exact. Serial and four-worker raw
iSTFT waveforms were also bit-exact for both channels. This stage therefore
introduces no numerical tolerance or PCM difference.

## Device DSP results

The original three-sentinel batch is the broad reference. Values are mean
STFT plus combined iSTFT/OLA milliseconds per window.

| Device | Model | Backend | Serial DSP | Parallel DSP | DSP speedup |
| --- | --- | --- | ---: | ---: | ---: |
| S25 | 9662 | CPU | 313 | 188 | 1.66x |
| S25 | 9662 | bounded GPU | 311 | 182 | 1.71x |
| S25 | Kim Inst | CPU | 394 | 222 | 1.78x |
| S25 | Kim Inst | bounded GPU | 410 | 217 | 1.89x |
| S25 | HQ4 | CPU | 302 | 166 | 1.82x |
| S25 | HQ4 | bounded GPU | 269 | 156 | 1.72x |
| S10 | 9662 | CPU | 1054 | 604 | 1.75x |
| S10 | 9662 | bounded GPU | 1074 | 630 | 1.71x |
| S10 | Kim Inst | CPU | 898 | 451 | 1.99x |
| S10 | Kim Inst | bounded GPU | 837 | 441 | 1.90x |
| S10 | HQ4 | CPU | 619 | 384 | 1.61x |
| S10 | HQ4 | bounded GPU | 906 | 352 | 2.58x |

The 9662 values come from an additional same-APK worker=1/worker=4 control.
That control was required because cross-batch LiteRT invoke time varied enough
to obscure total-window comparisons. Kim and HQ4 still showed positive DSP
speedup in every device/backend group. Model inference is sequential with DSP
and is not changed by this patch; invoke drift is not attributed to DSP.

## Decision

The stage passes. Carry the four-worker full-complex policy into the workspace
reuse and native experiments. Continue reporting DSP and invoke separately.
Do not use cross-session total-window changes as the primary DSP metric.

Raw reports remain under ignored
`outputs/android-benchmark/mdx-parallel-dsp-2026-08-08/`.
