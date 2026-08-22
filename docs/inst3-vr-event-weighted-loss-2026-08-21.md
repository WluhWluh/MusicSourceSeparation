# Inst 3 V-R-H50 Event-Weighted Loss Experiment

Status: completed locally on 2026-08-21. This is a non-commercial MUSDB18
research experiment. Checkpoints, teacher-derived targets, MUSDB18-derived
caches, and listening files remain in the ignored `data/` tree and are not
publishable artifacts.

## Purpose

The preceding V-R-H50 run was the best current listening result. This run
keeps its sampling schedule completely fixed and tests whether giving selected
short-removal events more loss weight produces a further Inst 3-like vocal
removal tendency.

The experiment arms are:

| arm | hard-event loss weight | meaning |
| --- | ---: | --- |
| `lambda-0.00` | 0.00 | exact reuse of the completed V-R-H50 control checkpoints |
| `lambda-0.25` | 0.25 | selected event-overlapping frames receive `1.25x` weight |
| `lambda-0.50` | 0.50 | selected event-overlapping frames receive `1.50x` weight |

The student remains the original residual-vocals model initialized from
`vocals_epoch=891.ckpt`. The target remains `mixtureGt - Inst3Instrumental`.
The H50 schedule is four uniform plus four song-local hard-event windows per
song and pass, over 80 training songs. Evaluation uses the 20
song-disjoint calibration/internal-test songs; the official final-test split
was not used.

## Implementation

Runner: `tools/run_inst3_vr_event_weighted.py`

Tests: `tests/test_inst3_vr_event_weighted.py`

The event mask is built from the frozen Stage 1 100 ms event arrays. A TFC
frame is marked when its centered `n_fft` support overlaps a selected event
block. The implemented loss is a normalized temporal-frame weighted spectral
L1:

```text
weight(frame) = 1 + lambda  if the frame overlaps a selected event
               1            otherwise
```

This is an explicit, deterministic spectral-frame approximation to the
proposed audio-domain weighting. It is not equivalent to a sample-exact
audio-domain mask at the FFT boundary, so later work should not describe it as
the final audio-domain loss.

Frozen contract:

- 128-frame TFC-TDF model, 44.1 kHz, original FFT/DSP contract
- 80 training songs and 20 evaluation songs, song-disjoint
- 8 fixed windows per song per pass, H50 selection
- 50 passes, 640 records/pass, 8,000 updates total
- batch size 4, AdamW, learning rate `1e-5`, weight decay 0
- seed `891`, frozen BatchNorm running statistics, gradient clip `1.0`
- milestones at 0, 25, and 50 passes

The full run used the repository CUDA environment (PyTorch 2.11.0+cu128) on
an NVIDIA GeForce RTX 4060 Laptop GPU. Both nonzero arms completed without a
CUDA error, non-finite training loss, or failed checkpoint.

## Objective Results

The values below are the 50-pass checkpoints on the 20-song evaluation set.
For projection and error values, more negative dBFS is better. Deltas are
relative to the exact V-R-H50 control (`lambda-0.00`).

| arm | accompaniment vocal projection | delta | instrumental SDR | delta | low-vocal instrumental SDR | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H50 control | -16.632 dB | - | 13.777 dB | - | 19.272 dB | - |
| `lambda-0.25` | -16.792 dB | -0.161 dB | 13.788 dB | +0.010 dB | 19.224 dB | -0.048 dB |
| `lambda-0.50` | -16.922 dB | -0.290 dB | 13.795 dB | +0.017 dB | 19.183 dB | -0.089 dB |

### Short-event metrics

`positiveProjectionRmsP95Dbfs` measures the upper tail of the portion of the
teacher-removed content still projected into the student's accompaniment.

| window | H50 control | `lambda-0.25` | delta | `lambda-0.50` | delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 ms | -33.607 | -33.826 | -0.219 | -33.956 | -0.349 |
| 100 ms | -34.094 | -34.261 | -0.167 | -34.443 | -0.349 |
| 200 ms | -34.394 | -34.655 | -0.261 | -34.788 | -0.394 |

At the per-song level, `lambda-0.25` improved the 50 ms and 100 ms projection
p95 on 20/20 songs and the 200 ms value on 19/20 songs. `lambda-0.50` improved
all three p95 values on 20/20 songs.

The result does not eliminate the worst individual leak: the maximum
positive-projection values changed by only about 0.003--0.006 dB relative to
H50. The aggregate miss-RMS maximum also changed by approximately +0.008 to
+0.012 dB. Thus the weighted loss improves the broad upper tail, but does not
solve the single most severe residual event.

The counts of active positive events above -30 dBFS / -35 dBFS were:

| window | H50 | `lambda-0.25` | `lambda-0.50` |
| ---: | ---: | ---: | ---: |
| 50 ms | 270 / 776 | 258 / 741 | 251 / 723 |
| 100 ms | 133 / 377 | 128 / 352 | 125 / 337 |
| 200 ms | 65 / 162 | 62 / 160 | 59 / 156 |

### Artifact checks

At 50 pass, both weighted arms had zero clipped samples and zero non-finite
windows. The first-difference mechanical-artifact proxy was slightly lower
than H50, not higher:

| arm | derivative excess p95 | derivative excess max | clipped | non-finite |
| --- | ---: | ---: | ---: | ---: |
| H50 | 0.870 dB | 3.281 dB | 0 | 0 |
| `lambda-0.25` | 0.867 dB | 3.229 dB | 0 | 0 |
| `lambda-0.50` | 0.858 dB | 3.216 dB | 0 | 0 |

This proxy is diagnostic only and cannot replace listening.

## Artifacts

Report:

`data/musdb18-inst3-vr-event-weighted/reports/inst3-vr-event-weighted-report.json`

12-song full-length listening outputs:

`data/musdb18-inst3-vr-event-weighted/listening-12/lambda-0.25/`

`data/musdb18-inst3-vr-event-weighted/listening-12/lambda-0.50/`

The exact H50 control outputs remain at:

`data/musdb18-inst3-vr-hard-sampling-h50/listening-12/V-R-H50/`

The listening report confirms 12 outputs for each weighted arm, with matching
sample rate, channel count, frame count, finite PCM, and FLAC decode checks.
The user has not yet supplied the subjective comparison for this run, so no
perceptual winner is declared here.

## Decision

The objective result supports retaining event weighting as a useful refinement
of H50. `lambda-0.50` is the strongest measured setting: it improves the
projection p95 monotonically without an objective artifact or overall
instrumental-SDR regression. The small low-vocal safety-metric decline and
unchanged worst hotspot mean it should remain an experimental listening
candidate, not a product replacement.

The next decision should be made from level-matched listening of H50,
`lambda-0.25`, and `lambda-0.50`, with special attention to the existing
short-syllable problem in `yoru-ni-kakeru`, harmony/reverb tails, and any new
scrape or tonal damage. If `lambda-0.50` is audibly preferred, the next
technical experiment should be a sample-accurate audio-domain event loss (or
an event-local anchor outside hard regions), rather than immediately raising
the multiplier further. If the two weighted arms are indistinguishable, keep
H50 as the simpler baseline.
