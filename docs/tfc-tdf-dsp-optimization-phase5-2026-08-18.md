# TFC-TDF DSP optimization Phase 5

Status: native packed-real FFT, fixed worker lanes, and fused iSTFT residual
output are implemented in the isolated Android streaming prototype. The
existing Kotlin/JTransforms path remains available as a regression baseline.

## Implementation

`NativeTfcTdfDsp` uses the frozen TFC-TDF contract:

- 2,048-point Hann FFT, 1,024 hop, 128 frames;
- 1,025 positive-frequency bins;
- NHWC complex feature order `L.real, R.real, L.imag, R.imag`;
- interleaved stereo input and output;
- 5,120-frame model-context trim remains outside the DSP transform;
- persistent native worker threads and per-worker FFT/overlap workspaces;
- pocketfft `r2c/c2r` or full-complex `c2c` selected per plan.

The native residual operation reconstructs, normalizes, subtracts the source,
and writes only the requested valid region. It does not change the model or
audio contract.

## Isolated S25 results

Device: Samsung SM-S9310 / `pa1q` / Snapdragon `SM8750`, API 35. Each profile
used two warmups and twelve rotated measurements in one instrumented run.

| Profile | STFT median | iSTFT median | fused residual | combined STFT+iSTFT |
| --- | ---: | ---: | ---: | ---: |
| Kotlin JTransforms | 8.29 ms | 10.92 ms | 11.03 ms | 19.32 ms |
| native full-complex, 1 lane | 5.49 ms | 5.06 ms | 4.98 ms | 10.52 ms |
| native packed-real, 1 lane | 3.77 ms | 3.27 ms | 3.22 ms | 7.06 ms |
| native packed-real, 2 lanes | 1.94 ms | 1.82 ms | 1.75 ms | 3.77 ms |
| native packed-real, 4 lanes | 1.06 ms | 1.28 ms | 1.05 ms | 2.35 ms |

The native packed 4-lane result is the fastest observed S25 configuration.
The full-chain run processed 17 neural windows with no late or discarded
outputs. Its aggregate STFT/iSTFT time was `89.2/86.1 ms` across the run;
the separate residual operation is a small wall-time optimization after the
FFT rewrite.

## Isolated S10 results

Device: Samsung SM-G9730 / `beyond1q` / Snapdragon `SM8150`, API 31. Profiles
were run in separate application processes to avoid persistent worker
contention between profiles.

| Profile | STFT median | iSTFT median | fused residual | combined STFT+iSTFT |
| --- | ---: | ---: | ---: | ---: |
| native full-complex, 1 lane | 8.09 ms | 8.71 ms | 8.60 ms | 16.81 ms |
| native packed-real, 1 lane | 5.73 ms | 5.77 ms | 5.38 ms | 11.45 ms |
| native packed-real, 2 lanes | 3.64 ms | 3.67 ms | 3.86 ms | 7.39 ms |
| native packed-real, 4 lanes | 7.50 ms | 15.17 ms | 14.98 ms | 22.29 ms |

Two lanes are the S10 choice. Four lanes have a worse tail and median because
the extra worker contention outweighs the additional parallelism.

## Full-chain checks

The dry/wet engine continued to run at approximately real-time speed with no
missed windows:

| Device/configuration | First wet | Seek to wet | Wall RTF |
| --- | ---: | ---: | ---: |
| S25 bounded GPU, packed 4, separate | 836 ms | 281 ms | 1.0013 |
| S25 bounded GPU, packed 4, fused | 698 ms | 164 ms | 1.0015 |
| S25 CPU, packed 4, separate | 744 ms | 532 ms | 1.0012 |
| S10 CPU, packed 2, separate | 1,466 ms | 1,125 ms | 1.0099 |

These are individual runs and are not a controlled latency claim. The S25
GPU fused run's lower seek value includes normal decoder/runtime scheduling
variance. The stable conclusion is that DSP is no longer the dominant part of
the S25 GPU first-window critical path; decoder fill and LiteRT invocation
remain larger.

## Remaining work

The next experiment is a two-slot host/inference pipeline that can prepare the
next input while the current GPU invocation is in flight. It must use separate
TensorBuffer slots and preserve output ordering. The LiteRT `readFloat()` copy
is still an explicit allocation; it will be investigated separately rather
than hidden in the DSP timing.
