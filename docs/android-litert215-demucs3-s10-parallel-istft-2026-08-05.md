# S10 LiteRT 2.1.5 HTDemucs parallel iSTFT experiment

Date: 2026-08-05

## Decision

The parity-preserving `parallel-lanes` experiment succeeded. Four outer iSTFT
workers are the clear S10 configuration for this implementation. They preserve
the serial output bit-for-bit while moving the sustained bottleneck back to the
LiteRT neural core.

This is not yet a real-time playback path. Across three 30-second tracks, the
mean E2E RTF remains 1.292 for official 6s, 1.382 for official-4s, and 1.310 for
guitar-ft. It is suitable for offline separation experiments. Product adoption
still needs long-song battery/thermal tests and application cancellation/seek
integration.

Real-IFFT and native NEON FFT were not implemented in this batch. The
parity-preserving lane implementation already reduced warm iSTFT to roughly
1.1-1.5 seconds per window. The neural core now takes roughly 4.5-5.5 seconds,
so a new FFT implementation is no longer the first bottleneck.

## Fixed identity

- Branch: `codex/s10-parallel-istft`
- Git revision: `3510d5c263ab8236cd96cfeb07feef1c1a5c0dff`
- Source dirty: `true`
- Device: `SM-G9730`, SoC `SM8150`, Android 12 / API 31
- Hardware serial SHA-256:
  `d3736be879f9280e55b2850a58604601b3b81955a97536e17b6f56d8c4627694`
- LiteRT: `2.1.5`
- LiteRT AAR SHA-256:
  `a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37`
- App APK: 144,715,111 bytes,
  `fea25bb84ac81f0a85bac2a095a8d20738c5615960dd74adba7f247a38832a1d`
- Test APK: 630,813 bytes,
  `9f678a09388dbf5abfae20d466d31824643a70061008e9c9fada82780bcc5a5b`
- Local and installed APK byte sizes and SHA-256 values match.
- LiteRT CPU threads: 4
- JTransforms global threads / processors: 8 / 8
- JTransforms 1D 2-thread / 4-thread thresholds: 8,192 / 65,536
- Inverse transform: `FloatFFT_1D.complexInverse`, complex array length 8,192
- JTransforms global settings were observed and recorded, never modified.

Models:

| Variant | Stems | Artifact SHA-256 |
| --- | ---: | --- |
| official | 6 | `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7` |
| official-4s | 4 | `9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81` |
| guitar-ft | 6 | `ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5` |

## Implementation

The serial path retains the original loop and shared FFT instance. The new
parallel mode treats every `stem x channel` plane as an independent lane. A DSP
instance owns a fixed, prestarted pool; each worker owns its own FFT instance,
FFT buffer, and overlap buffer. A worker processes complete planes, so frame
order, window multiplication, overlap accumulation, normalization, and output
write order within each plane remain unchanged.

The DSP is single-caller synchronized, cannot race with `close()`, responds to
cancellation between frames, and performs bounded executor shutdown while
preserving the caller interrupt flag. No JTransforms process-global state is
changed.

The benchmark records LiteRT threads separately from iSTFT mode/workers. Mode,
worker count, JTransforms settings, and parity-check mode are included in each
report. The per-run reuse validator checks those fields before accepting an
existing output directory. This is not a complete resume identity: it does not
bind an existing run to the current APK digests, hardware serial or fingerprint,
exact dirty-tree contents, or runner SHA; the root `device.json` is overwritten,
and the matrix is recreated. It also accepts any positive reported JTransforms
global thread/processor counts rather than requiring the observed 8/8. A
`skipped-valid-existing` result therefore proves contract/output reuse, not full
execution-provenance identity. Parallel pool construction is included in
session prepare, not in per-window iSTFT timing.

## Host parity gate

`HtdemucsDspFixtureTest` passed 4/4 tests:

- Four- and six-stem serial versus 2-worker and 4-worker outputs compare equal
  with `Float.toRawBits()` for every element.
- The same 4-worker DSP is reused for three six-stem reconstructions to check
  determinism and pool reuse.
- No live `htdemucs-istft-*` thread remains after close.
- Canonical Torch iSTFT parity remains 133.54 dB SNR with maximum absolute
  error `1.1920929e-7`.
- Canonical Torch STFT parity remains 133.61 dB SNR with maximum absolute
  error `4.7683716e-7`.

## S10 raw-float gate

The five-second gate executes the selected parallel implementation, then uses
an independent serial DSP to reconstruct the same neural-core frequency tensor
on S10/ART. It compares every raw FP32 bit and hashes the little-endian raw
bytes. The serial reference is outside the measured parallel iSTFT stage.

| Model | Outer workers | iSTFT, one window | Raw FP32 mismatches | Raw FP32 SHA-256 |
| --- | ---: | ---: | ---: | --- |
| official | 1 serial | 13,007.6 ms | n/a | n/a |
| official | 2 | 5,968.8 ms | 0 / 4,127,760 | `9cab552a9a6ed13a80238fac9891420c8e4aa4e463ac225e5de1f5c5400df189` |
| official | 4 | 994.4 ms | 0 / 4,127,760 | `9cab552a9a6ed13a80238fac9891420c8e4aa4e463ac225e5de1f5c5400df189` |
| official-4s | 1 serial | 9,750.9 ms | n/a | n/a |
| official-4s | 2 | 4,049.3 ms | 0 / 2,751,840 | `5b8b9764138007b34f9d395f77361e601a1233878f4b7a218b5c07c51c58353e` |
| official-4s | 4 | 1,091.7 ms | 0 / 2,751,840 | `5b8b9764138007b34f9d395f77361e601a1233878f4b7a218b5c07c51c58353e` |

All ten output WAVs per execution matrix are also SHA-identical across serial,
2-worker, and 4-worker modes. Single-window timings select the candidate only;
the sustained result below is authoritative.

The nonlinear 2-to-4-worker improvement is specific to the observed nested
execution policy. Each 4,096-point complex inverse reaches JTransforms' 8,192
element 2-thread threshold. Four outer workers can therefore feed the eight
global JTransforms workers on this eight-core device. These are not total
thread-count 1/2/4 tests.

## Sustained 30-second result

Inputs are the same first 1,323,000 decoded frames used by the previous S10
matrix: Athletics II, John Lennon - Imagine, and Josiah James - Chasing The
Wind. Every run has six windows. The table pools windows 1-5 from all three
tracks, giving 15 warm observations per model. P95 is nearest-rank and equals
the maximum for 15 observations.

| Model | Warm iSTFT P50 / mean / P95 | Warm core mean | Warm window P50 / mean / P95 | E2E RTF mean [range] | Mean per-run peak window-end PSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| official | 1,505.9 / 1,494.5 / 1,737.1 ms | 4,546.7 ms | 6,312.7 / 6,272.8 / 6,508.1 ms | 1.292 [1.258-1.315] | 1,089.7 MiB |
| official-4s | 1,107.1 / 1,075.8 / 1,223.6 ms | 5,488.2 ms | 6,733.7 / 6,748.7 / 7,053.6 ms | 1.382 [1.368-1.405] | 1,143.6 MiB |
| guitar-ft | 1,585.0 / 1,548.8 / 1,833.7 ms | 4,582.3 ms | 6,374.3 / 6,362.8 / 6,719.3 ms | 1.310 [1.303-1.316] | 1,090.3 MiB |

Relative to the prior three-track serial matrix:

| Model | Prior warm iSTFT mean | 4-worker mean | iSTFT speedup | Prior RTF | 4-worker RTF | RTF reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official | 9,949.7 ms | 1,494.5 ms | 6.66x | 2.960 | 1.292 | 56.4% |
| official-4s | 6,338.9 ms | 1,075.8 ms | 5.89x | 2.401 | 1.382 | 42.5% |
| guitar-ft | 9,536.0 ms | 1,548.8 ms | 6.16x | 2.905 | 1.310 | 54.9% |

The prior matrix used app APK `66cc8e91...0bcf`; the final experiment uses
`fea25bb8...32a1d`. A same-APK Athletics control was therefore also run. Its
serial-to-parallel warm iSTFT speedups are 8.50x, 8.07x, and 8.77x; E2E RTF
reductions are 61.7%, 53.3%, and 65.8%. The same-APK serial measurements were
much more variable than the prior serial matrix even with thermal status 0.
This unresolved Java/JTransforms scheduling variance is why the report retains
both comparison scopes instead of presenting one as universally exact.

The 4-worker run adds only about 7-10 MiB to the mean of per-run peak
window-end PSS relative to the prior serial matrix. Each run-level peak is the
largest post-window sample, not an independently sampled instantaneous peak.

## Output and safety audit

- Final 4-worker inventory: 9 reports, 48 WAVs, all complete.
- Actual report SHA and WAV SHA were recomputed from disk and match each matrix.
- All WAVs are 44.1 kHz, stereo, PCM16, and exactly 1,323,000 frames.
- The 48 final WAV SHA values match the prior serial three-track matrix exactly.
- All nine selected PCM SHA values match the prior serial matrix.
- Non-finite output samples: 0.
- Thermal status values: only 0.
- Clipping is unchanged from the prior matrix: Josiah drums has 74 samples for
  official and 82 for official-4s; all other outputs have zero.
- Five-second 2/4-worker parity inventory: 4 reports, 20 WAVs, all complete;
  raw-float mismatch count is zero for both four- and six-stem shapes.

## Interpretation and next step

Use `parallel-lanes`, four outer workers, and the unchanged JTransforms global
policy for the next S10 application experiment. Do not expose worker count as
an unconstrained product tuning knob; the result depends on the nested 4 x 2
execution observed on SM8150.

The optimized warm window remains longer than the 257,985-sample stride
(approximately 5.85 seconds), and all model RTF values remain above 1.0. A
finite prebuffer cannot make this sustain real-time playback. Keep it offline.

If near-real-time six-stem separation becomes the next research target,
real-IFFT or a pinned native FFT remains worth a separate mode and its own ARM
raw-float/tolerance gate. Eliminating iSTFT entirely would only reduce the
observed mean RTF to about 0.988 for official, 1.163 for official-4s, and 0.995
for guitar-ft. In practice a real-IFFT will remove only part of that time, so
neural-core optimization is now at least as important. Do not mix such a
rounding-changing experiment into this bit-exact lane result.

## Evidence

The byte-identical parallel runner is frozen at:

```text
tools/frozen-runners/
  76943655082cb26265d7c9b5f720533cdf99cd94d0109efd2b3d5bab7deafa6c/
  run_htdemucs_cpu_matrix.py
SHA-256 76943655082cb26265d7c9b5f720533cdf99cd94d0109efd2b3d5bab7deafa6c
```

The current tracked entry point is a privacy-hardened successor with SHA-256
`4277970030200d81dbbb1e022c59ce37ea92158d8bc39a7141054c9a0fefe1a7`. The
matrix and device reports do not embed the runner SHA, so the frozen identity
is supported by the preserved execution-session history rather than by a
self-contained report field.

- `outputs/htdemucs-s10-istft-parallel-20260805-v3/device.json`
- `outputs/htdemucs-s10-istft-parallel-20260805-v3/matrix-5s-istft-serial-w1.json`
- `outputs/htdemucs-s10-istft-parallel-20260805-v3/matrix-5s-istft-parallel-lanes-w2-float-parity.json`
- `outputs/htdemucs-s10-istft-parallel-20260805-v3/matrix-5s-istft-parallel-lanes-w4-float-parity.json`
- `outputs/htdemucs-s10-istft-parallel-20260805-v3/matrix-30s-istft-serial-w1.json`
- `outputs/htdemucs-s10-istft-parallel-20260805-v3/matrix-30s-istft-parallel-lanes-w4.json`
- Prior baseline: `outputs/htdemucs-s10-cpu-matrix-20260805/matrix-30s.json`
