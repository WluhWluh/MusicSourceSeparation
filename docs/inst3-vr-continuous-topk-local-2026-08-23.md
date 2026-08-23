# Continuous H50 Top-K Local Training

Status: completed locally on 2026-08-23. This is a non-commercial MUSDB18
research experiment. Checkpoints, caches, teacher-derived audio, and private
listening files remain under the ignored `data/` tree.

## Question

Can the most severe remaining Inst 3-directed events be improved by training
only their audio-domain loss while anchoring the rest of each window to H50?
The comparison is continuous-context throughout, so isolated zero-padding is
not part of the training or listening path.

## Contract

- Source checkpoint: continuous H50 `V-R-H50@step-8000`.
- Student semantic: residual vocals, with instrumental derived as
  `mixture - residual`.
- Teacher target: `V_T = mixtureGt - Inst3Instrumental`.
- Student/teacher input assembly: 128-frame continuous-context overlap-save;
  2,048 FFT, 1,024 hop, five context hops per side.
- Data: 80 train songs; 10 calibration and 10 internal-test songs; official
  final-test songs unused.
- Event pool: Stage 1 50/100 ms top events, rescored with the actual
  continuous H50 output and native Inst 3 output.
- Excluded from training: first/last 100 ms, internal join neighborhoods
  within 100 ms, and useful-span edge regions outside the central 80%.
- Selected records: eight per train song, 640 records per pass, five passes,
  800 updates per arm, batch size 4, seed 891, learning rate `1e-6`, frozen
  BatchNorm statistics.
- `H50-continuation`: full useful-span audio Charbonnier loss toward `V_T`.
- `H50-topk-local`: top-2 event-record Charbonnier loss toward `V_T` plus H50
  anchor loss outside the event mask, `anchorBeta=1.0`.

Machine-readable report:

`data/musdb18-inst3-vr-continuous-topk-local/reports/continuous-topk-local-report.json`

Runner:

`tools/run_inst3_vr_continuous_topk_local.py`

## Event selection

The 80-song sparse rescan examined 3,840 Stage 1 top-event rows:

| Quantity | Count |
| --- | ---: |
| Active source events | 3,840 |
| Song-edge events excluded | 2 |
| Join/useful-edge events excluded | 2,017 |
| Short-tail events excluded | 22 |
| Continuous central candidates | 1,799 |
| Unique selected windows | 490 |
| Repeated records for short songs | 150 |

The repeated records preserve the fixed eight-record budget for very short
MUSDB18 songs. They do not create new audio samples.

## Holdout result

Values are relative to the matched continuous H50 checkpoint. Negative is
better for projection/error metrics; positive instrumental SDR is better.

### Whole-song per-song mean

| Arm | Instrumental SDR | Vocal projection | Inst 3 positive projection | Raw miss RMS |
| --- | ---: | ---: | ---: | ---: |
| H50-continuation | `-0.049 dB` | `-0.589 dB` | `-1.679 dB` | `+0.038 dB` |
| H50-topk-local | `-0.034 dB` | `-0.241 dB` | `-0.607 dB` | `+0.032 dB` |

The continuation arm is substantially stronger than top-k-local on the
aggressive-removal direction. Neither arm improves raw miss RMS overall.

### Short-event directions

| Arm | Block | Positive projection p95 | Positive projection max | Raw miss p95 | Raw miss max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continuation | 50 ms | `-0.926 dB` | `-0.374 dB` | `-0.023 dB` | `-0.015 dB` |
| Continuation | 100 ms | `-0.996 dB` | `-0.476 dB` | `-0.000 dB` | `+0.035 dB` |
| Continuation | 200 ms | `-1.034 dB` | `-0.736 dB` | `+0.024 dB` | `+0.017 dB` |
| Top-k-local | 50 ms | `-0.309 dB` | `-0.236 dB` | `+0.020 dB` | `+0.067 dB` |
| Top-k-local | 100 ms | `-0.321 dB` | `-0.308 dB` | `+0.007 dB` | `+0.061 dB` |
| Top-k-local | 200 ms | `-0.321 dB` | `-0.300 dB` | `+0.038 dB` | `+0.032 dB` |

The continuation positive-projection p95 improved on all 20 holdout songs at
all three durations. Its maximum improved on 18/20 songs at 50 ms and 20/20
at 100/200 ms. Top-k-local p95 also improved on all 20 songs, but by only about
0.31 dB; its maximum improved on 18/20, 20/20, and 19/20 songs.

The predeclared 0.5 dB maximum-hotspot gate is not met consistently by
top-k-local. Continuation reaches the gate at 100/200 ms but not 50 ms, and
its raw-miss maxima remain slightly worse. Neither arm is a clear listening
replacement without the private A/B review.

## Region behavior

Both arms changed the central and join regions, so the result is not explained
by song or window boundaries. The top-k-local arm's mean continuous-region
metrics remained close to continuation, but its Inst 3 projection reduction
was consistently smaller. The song-edge region is sparse and excluded from
training; it is not used for model selection.

## GPU profile

The run was completed on the reduced-clock RTX 4060 Laptop GPU. Sampling during
training showed low average utilization, generally `0-25%`, with approximately
6.5 GB allocated. A separate synchronized fixed-cache profile measured:

| Batch | Forward | iSTFT | Backward | Step | Total/update |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.27 ms | 0.91 ms | 44.70 ms | 6.82 ms | 70.94 ms |
| 2 | 39.99 ms | 1.10 ms | 94.14 ms | 5.13 ms | 140.62 ms |
| 4 | 82.81 ms | 1.32 ms | 192.37 ms | 5.62 ms | 282.40 ms |
| 8 | 667.82 ms | 57.11 ms | 8,201.42 ms | 7.03 ms | 8,938.13 ms |

Batch 8 causes memory/scheduling thrashing and is not a viable optimization.
Batch 4 has approximately the same per-sample throughput as batch 1/2. The
remaining utilization gap is primarily CPU-side cache stacking, host-to-device
copy, small-kernel launch overhead, and the per-update synchronization used to
surface CUDA errors promptly. The next performance experiment should test
pinned host buffers with asynchronous prefetch and less frequent explicit
`torch.cuda.synchronize()`, while retaining periodic correctness fences.

## Artifacts and validation

- `H50-continuation@step-800`: SHA-256
  `d02c9629bc1c25039f6dbb17a04bca5b151ffb001d2a740d55be9511adde121a`.
- `H50-topk-local@step-800`: SHA-256
  `dc5817b7902cfc53a4e8cf7f36ea8fdfcc11e210ee94c36e309c091e8dc5b531`.
- 20-song continuous evaluation completed for both arms.
- 12-song listening set: 72 finite 44.1 kHz stereo PCM16 FLAC files under
  `data/musdb18-inst3-vr-continuous-topk-local/listening-12/`.
- 15 focused unit tests, Python compilation, checkpoint finiteness, and audio
  frame/sample-rate validation passed.

## User listening review

The user compared the continuous H50 baseline with the two trained arms on
the 12-song set. `H50-continuation` has audibly less vocal residue than the
original H50 in some passages and is suitable as the starting point for a
further controlled training round. This is a partial listening preference,
not a claim that every short hotspot improved. No final product selection was
made between continuation and top-k-local in this review.

## Decision

Do not retain `H50-topk-local` as the next model candidate. The local anchor
made the update more conservative but also removed most of the measurable
aggressive gain. `H50-continuation` is the stronger experimental arm and is
the appropriate starting checkpoint for the next controlled round. Its
raw-miss maximum warning remains active, so the next round must continue to use
the 12-song continuous listening set and per-hotspot maximum metrics.

Further loss-mask variations on the same 128-frame model are lower priority.
If continuation is not audibly preferred, the remaining limitation is more
likely model temporal capacity or target ambiguity than event selection.
