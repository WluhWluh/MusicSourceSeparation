# Inst 3 V-R-H50 hard-window sampling experiment

Status: complete local, non-commercial research experiment.  The experiment
was run after the GPU core and memory overclock had been reduced.  No model
was exported, integrated into Booming SS, or published.

Machine-readable report:

`data/musdb18-inst3-vr-hard-sampling-h50/reports/inst3-vr-hard-sampling-report.json`

The 12-song listening set is under:

`data/musdb18-inst3-vr-hard-sampling-h50/listening-12/`

All checkpoints, caches, teacher-derived audio, and private listening files
remain local ignored research artifacts.

## Contract

The run keeps the corrected V-R contract from the H25 experiment:

- Base student: `vocals_epoch=891.ckpt`, residual-vocals output semantic.
- Teacher target: `teacherResidual = mixtureGt - Inst3Instrumental` in the
  audio domain, then the student STFT contract.
- Data: all 80 MUSDB18 train songs; 20 song-disjoint calibration/internal-test
  songs; official final-test songs were not used.
- Student: 128 frames, frozen BatchNorm running statistics, AdamW with zero
  weight decay, learning rate `1e-5`, batch size 4, gradient clip `1.0`, seed
  `891`.
- Each pass: eight fixed windows per train song, 640 records, 160 updates.
- Milestones: 0, 25, and 50 passes; 8,000 updates at pass 50.
- `V-R-U`: eight uniform candidate draws.
- `V-R-H50`: four shared uniform draws plus four draws from the song-local
  top-25-percent hard-event pool.

The H50 run uses a separate experiment root and cache contract from H25.  The
same run also retrained `V-R-U`, so the U/H50 comparison uses the same run
environment and schedule budget.

Selected identities:

| Item | SHA-256 or value |
| --- | --- |
| Report | `376ddc6e443f408ef4b1b6c52c7c61bcd5d929d2d4dcb2dd071b7d41ad3991cf` |
| Runner | `52936ea48df99dae11128b007aff25669e6f9cedd111f2564f71ca54dd5d1671` |
| Manifest | `3a580e39c8754abb902784f257fb251f1ba52e6be678f5022343a1846252ab2b` |
| Selection | `4df1e5c9c29303282670dea1cd41531c780fdc3ab9235c79d134d4c945d31d4a` |
| Event report | `3baf35352edca46d5cc7ec2b3c69956500d8a55bae27b0642f83be14bf2534b8` |

## GPU execution

The run completed on the reduced-clock NVIDIA GeForce RTX 4060 Laptop GPU
with PyTorch `2.11.0+cu128`.  The 80-song teacher cache stage and both 8,000
update training arms completed without illegal-memory-access errors,
cuDNN failures, NaN/Inf values, or non-finite evaluation windows.

Training wall time was approximately 2,345 seconds for H50 and 2,412 seconds
for the U control, excluding the shared cache preparation.

## Evaluation result

Lower dB values for `accompanimentVocalProjection` and `teacherResidualMiss`
mean less content retained in the signal that Inst 3 removes.  The event
metrics use the deterministic 20-song coverage evaluation.

| Checkpoint | Instrumental SDR | Accompaniment vocal projection | 50 ms miss p95 | 100 ms miss p95 | 200 ms miss p95 | 50 ms > -30 dB count | Low-vocal instrumental SDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 13.687 | -15.908 | -29.000 | -29.163 | -29.232 | 381 | 19.525 |
| V-R-U pass 50 | 13.757 | -16.104 | -29.251 | -29.482 | -29.770 | 313 | 19.489 |
| V-R-H50 pass 25 | 13.734 | -16.367 | -29.303 | -29.561 | -29.786 | 299 | 19.357 |
| V-R-H50 pass 50 | 13.777 | -16.632 | -29.455 | -29.725 | -29.882 | 270 | 19.272 |

Relative to the same-run U pass 50, H50 improves:

- accompaniment vocal projection by `-0.528 dB`;
- 50/100/200 ms miss p95 by `-0.204/-0.243/-0.112 dB`;
- 50 ms positive-projection p95 by `-0.888 dB`;
- the 50 ms positive-projection count above `-30 dBFS` by 43 events.

All 20 evaluation songs improved in accompaniment vocal projection relative to
the U control.  This is a stronger aggregate and per-song result than the
H25 comparison, but it remains an objective diagnostic rather than a
perceptual judgment.

Relative to H25 pass 50, H50 pass 50 continues the aggressive trend:

- accompaniment vocal projection improves by about `-0.271 dB`;
- 50/100/200 ms miss p95 improves by about `-0.088/-0.178/-0.047 dB`;
- the 50 ms hotspot count falls from 285 to 270;
- standard instrumental SDR is about `0.020 dB` lower;
- low-vocal instrumental SDR is about `0.103 dB` lower.

The first-difference artifact proxy p95 is `0.870 dB`, versus `0.867 dB` for
the same-run U and `0.858 dB` for H25.  Clipped samples and non-finite
windows are both zero.  The small proxy increase is not a perceptual artifact
score and must be checked by listening.

## Listening set

The renderer generated 24 valid full-song PCM16 FLAC files:

- `V-R-U/`: 12 control songs;
- `V-R-H50/`: 12 H50 songs;
- `render-report.json`: source identities, checkpoint identities, output
  hashes, frame counts, and render timing.

Every output is 44.1 kHz stereo and preserves its source frame count.  The
files are local listening artifacts only and must not be uploaded to a model
release.

## Decision

The full-song A/B review found a small but repeatable improvement over H25 and
the same-run U control.  H50 left less overall vocal residue and handled some
isolated short leaks slightly better.  No accompaniment-quality regression or
new mechanical artifact was audible in the twelve-song private set.

H50 pass 50 is therefore the current best result by human listening and
becomes the preferred baseline for the next controlled experiment.  This is a
local listening decision, not product qualification: the private set is small,
the official MUSDB18 final-test split remains untouched, and the checkpoint is
still subject to the local non-commercial research restrictions above.

The next experiment may move more aggressively toward Inst 3, but it must
change only one axis at a time.  The preferred first follow-up is a fixed H50
sampling schedule with event-weighted teacher loss; H75 should remain a
separate sampling-dose experiment rather than being combined with the new
loss.
