# Inst 3 V-R hard-window sampling experiment

Status: complete local research experiment.  The corrected run completed on
2026-08-21 after the GPU core and memory overclock were reduced.  No model was
exported, integrated into Booming SS, or published.

Machine-readable report:

`data/musdb18-inst3-vr-hard-sampling/reports/inst3-vr-hard-sampling-report.json`

All MUSDB18-derived caches, checkpoints, and teacher outputs remain under the
ignored local `data/` tree.

## Contract

- Base student: `vocals_epoch=891.ckpt`, residual-vocals output semantic.
- Teacher: `uvr_mdxnet_inst_3@2`; target is
  `mixtureGt - Inst3Instrumental` in the audio domain, then the student STFT.
- Data: all 80 MUSDB18 train songs; calibration and internal-test contain 20
  song-disjoint evaluation songs.  The official final-test split was not used.
- Per song: eight selected windows per pass, 640 records per pass, batch size
  4, AdamW at `1e-5`, gradient clip `1.0`, 50 passes, checkpoints at 0/25/50.
- `V-R-U`: eight uniform draws per song.
- `V-R-H25`: six shared uniform draws plus two draws from that song's local
  top-25-percent hard-event pool.
- Both variants use the same pass permutation and total update budget.

The corrected cache contract is
`local-inst3-vr-hard-sampling-cache@2`, with
`targetAlignment=segment-origin-0`.  This records that a teacher residual
already sliced to a candidate window must be transformed with local origin 0.

Selected identity values from the corrected report:

| Item | SHA-256 or value |
| --- | --- |
| Report | `e301ba5bd69ca05b238200cf646ffcb1cac51c97485779a55ca04bd1f753a26d` |
| Runner | `bba660c1e410649228288675c88708208479ca390b531dc9e6c51f77c113eca0` |
| Manifest | `3a580e39c8754abb902784f257fb251f1ba52e6be678f5022343a1846252ab2b` |
| Selection | `6c029ebee2217a180317a9a7010d26796a7620b55a21f859f40719bfceb549b5` |
| Student checkpoint | `101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d` |
| Inst 3 teacher | `2b7834e2972158d8c9864e7376e3a7d084079c80a23f38dc31c4b0a4e901a1cb` |

## Correctness repair

The first long run after the underclock change was numerically stable, but its
evaluation was invalid.  `teacher_residual` was a local candidate slice, while
the cache builder passed the original song-level `candidate.start` to
`student_window_spec`.  Most target windows therefore became effectively
zero, and all trained checkpoints collapsed toward the same zero-residual
behavior.

The runner now builds the target with a dedicated local-origin helper and
increments the cache contract.  A regression test compares the helper with the
local-origin reference and proves that the old song-level offset produces the
wrong near-zero result.  The old report is not used for any conclusion; the
report referenced above is from a forced cache rebuild with the corrected
contract.

An audit after rebuilding found 80/80 training caches with non-zero target
arrays.  Target RMS ranged from approximately `0.583` to `2.182` (float32
packed student spectra).

## GPU stability

The host used an NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch `2.11.0+cu128`,
and CUDA `12.8` on Windows 11.  After the user reduced the GPU core and memory
overclock:

- A five-song debug run passed 1,000 updates for each variant with CUDA
  launch checking enabled.
- The corrected formal run passed 8,000 updates for `V-R-U` and 8,000 for
  `V-R-H25`.
- Both variants passed their pass-25 and pass-50 checkpoints.
- No illegal-memory-access, cuDNN execution failure, NaN/Inf training value,
  or evaluation non-finite window was observed.
- The formal run used the normal cuDNN path and non-fused gradient clipping;
  it did not switch to the slower CPU/native-convolution fallback.

The earlier failures under the higher overclock were not reproduced after the
clock reduction.  This is strong evidence that the clock margin contributed
to the previous CUDA instability, although it is not a hardware warranty.

## Evaluation result

The fixed evaluation covers 20 song-disjoint calibration/internal-test songs
and 37,447,680 evaluated samples.  Event metrics are computed over active
teacher-removed blocks; lower dB values mean less retained content projected
onto the Inst 3 removed signal.

| Checkpoint | Instrumental SDR (dB) | Accompaniment vocal projection (dB) | 50 ms miss p95 (dBFS) | 100 ms miss p95 (dBFS) | 200 ms miss p95 (dBFS) | 50 ms > -30 dB count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 13.687 | -15.908 | -29.000 | -29.163 | -29.232 | 381 |
| V-R-U pass 25 | 13.733 | -15.952 | -29.227 | -29.418 | -29.530 | 339 |
| V-R-U pass 50 | 13.755 | -16.100 | -29.251 | -29.486 | -29.759 | 313 |
| V-R-H25 pass 25 | 13.759 | -16.340 | -29.318 | -29.527 | -29.765 | 295 |
| V-R-H25 pass 50 | 13.797 | -16.361 | -29.367 | -29.547 | -29.835 | 285 |

Relative to the initial checkpoint, H25 pass 50 gives:

- `-0.453 dB` accompaniment vocal projection;
- `-0.367 dB`, `-0.384 dB`, and `-0.602 dB` for the 50/100/200 ms miss p95;
- `-1.342 dB` for the 50 ms positive-projection p95 (the corresponding
  maximum changed by `-0.099 dB`);
- `+0.110 dB` instrumental SDR.

Relative to the uniform pass-50 control, H25 pass 50 improves aggregate
accompaniment vocal projection by `0.261 dB` and improves the 50/100/200 ms
miss p95 by `0.116/0.061/0.076 dB`.  On the 20 individual songs, H25 pass 50
improves the accompaniment vocal projection on 18 songs; the median
per-song change is `-0.165 dB`.

Safety checks for H25 pass 50 found zero clipped samples and zero non-finite
windows.  The first-difference artifact proxy p95 excess was `0.858 dB`,
below the initial `1.066 dB`; this proxy is diagnostic and is not a perceptual
artifact score.  Low-vocal instrumental SDR changed from `19.525` to
`19.375 dB`, a small safety tradeoff that should be checked by listening
before any follow-up training.

## Decision

The hard-window sampler is promising but the aggregate gain is modest.  H25
is preferable to U for the next controlled experiment because it improves the
short-event metrics without a numerical artifact regression.  The result is
not sufficient for product integration or weight publication.  The next
useful step is a listening comparison of the corrected U/H25 pass-50 outputs,
followed by a separately budgeted H50 or adaptive-refresh experiment only if
the audible improvement agrees with these metrics.  No official final-test
claim should be made from this run.

## Private listening set

The corrected pass-50 checkpoints were rendered on the same twelve full-song
private sources used by the earlier scale-10 experiments:

`data/musdb18-inst3-vr-hard-sampling/listening-12/`

- `V-R-U/`: 12 PCM16 FLAC files.
- `V-R-H25/`: 12 PCM16 FLAC files.
- `render-report.json`: source identities, checkpoint identities, output
  hashes, frame counts, and render timing.

All 24 files are 44.1 kHz stereo and preserve the source frame count.  The
corresponding U/H25 files have distinct hashes.  They are local listening
artifacts only and must not be uploaded to a release.
