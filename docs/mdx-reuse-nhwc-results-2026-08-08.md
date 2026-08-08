# MDX reusable NHWC DSP results

## Change

The explicit `reuse-nhwc` profile keeps the qualified four-worker full-complex
algorithm while adding persistent padding, input tensor, inverse overlap-add,
trimmed waveform, residual, and PCM16 workspaces. STFT writes NHWC directly and
iSTFT consumes NHWC directly. The legacy profile remains available.

The three complete sentinel shapes passed zero-tolerance JVM comparisons:

- direct-NHWC STFT equals legacy NCHW followed by layout conversion;
- reusable NHWC iSTFT equals the legacy raw-float waveform;
- serial/parallel parity from the preceding stage remains unchanged.

## Device gate

S25 and S10 each completed 18 sessions: three models, CPU4 and bounded OpenCL
FP32, three cold sessions, two warmups, and ten measured windows. All reports
completed, all windows were finite, GPU dispatch evidence was positive, and
all session start/end thermal statuses were 0.

| Device | Model | Backend | Allocation before | Reuse | Reduction | GC before/reuse | Max PSS before/reuse |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| S25 | 9662 | CPU / GPU | 827 MiB | 484 MiB | 41.4% | 17 / 10 | 774/743, 520/521 MiB |
| S25 | Kim Inst | CPU / GPU | 1067 MiB | 604-605 MiB | 43.4% | 20 / 10-11 | 1386/1362, 1020/995 MiB |
| S25 | HQ4 | CPU / GPU | 846 MiB | 444 MiB | 47.5% | 15 / 9 | 1166/1131, 895/874 MiB |
| S10 | 9662 | CPU / GPU | 827 MiB | 484 MiB | 41.5% | 40 / 20 | 765/742, 422/415 MiB |
| S10 | Kim Inst | CPU / GPU | 1067 MiB | 604 MiB | 43.4% | 40 / 20 | 1359/1333, 798/774 MiB |
| S10 | HQ4 | CPU / GPU | 846 MiB | 444 MiB | 47.5% | 39-41 / 20 | 1158/1128, 697/688 MiB |

Tensor layout time fell from approximately 12-57 ms per window to below the
reporting resolution. Mean STFT plus iSTFT/OLA improved in eleven of twelve
device/model/backend groups. The S10 Kim bounded-GPU group regressed 7.8% in
DSP time while remaining bit-exact and allocation-improved; it is retained as
variance evidence rather than used to reject the memory/layout change.

## Decision

The stage passes. Carry the direct NHWC contract and reusable workspace policy
into `NativeMdxDsp`. Keep model invoke and DSP timing separate. Native work
must improve on this profile, not on the original allocating serial baseline.

Raw reports remain under ignored
`outputs/android-benchmark/mdx-reuse-nhwc-2026-08-08/`.
