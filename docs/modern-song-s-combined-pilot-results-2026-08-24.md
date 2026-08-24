# Combined Modern-S Pilot

Status: completed locally on 2026-08-24. This is a local non-commercial
research experiment. Source audio, derived caches, and teacher-derived
checkpoints remain local artifacts.

## Objective

Test whether enlarging the reviewed `S` event pool with a second batch of
modern songs produces a stronger and safer Inst 3-oriented student than an
equal-budget MUSDB-only continuation. `S` is the manually marked subset of
`R` whose residual vocal is especially conspicuous because of level or clear
consonants. The target and loss semantics were unchanged; only the external
sampling pool changed.

## Frozen contract

- Source: `H50-continuation+5/step-1600.pt`.
- Previous pool: 32 events from 16 songs.
- Current two-pass-consensus pool: 54 events from 22 songs.
- C1 external schedule: 24 previous and 40 current records per pass.
- C0 extra slots: deterministic MUSDB records with the same budget.
- 640 MUSDB records plus 64 extra records per pass (`704` total).
- Five passes, `880` updates, batch size `4`, AdamW, learning rate `1e-6`.
- BatchNorm running statistics frozen, gradient clipping `1.0`, seed `891`.
- 128-frame continuous-context overlap-save assembly.
- External event loss: 100 ms core plus 25 ms soft guard; non-event H50
  anchor weight `0.25`.
- Calibration and internal-test songs were evaluated; the official final
  test was not used.

The exact selection and schedule hashes are recorded in
`data/modern-song-s-combined-pilot/reports/combined-s-report.json`.

## Training artifacts

| Arm | Final checkpoint | SHA-256 |
| --- | --- | --- |
| C0-combined-control | `runs/C0-combined-control/step-2480.pt` | `70ea1a34207ac6c47ac556978db1d0671ae449175e113cd3ce479af4b991844c` |
| C1-combined-stable-S | `runs/C1-combined-stable-S/step-2480.pt` | `dd9570c4ecbfad22e529d5fcf3751a938c946b52abd2723558cfe40f3f523a15` |

Milestones `step-1776`, `step-2128`, and `step-2480` were saved for both
arms. The CUDA run used an NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch
`2.11.0+cu128`; observed GPU utilization was approximately 88-98 percent and
GPU memory use was approximately 6.5-6.8 GB.

## Continuous MUSDB evaluation

Relative to C0, C1 changed the 20-song calibration/internal-test mean as
follows:

- whole-song positive vocal projection: about `-0.59 dB` further toward
  Inst 3;
- whole-song instrumental SDR: about `-0.16 dB`;
- teacher-removed positive projection: about `-2.04 dB`;
- teacher residual-miss RMS: about `+0.17 dB`.

At short windows, C1 reduced positive vocal projection consistently, but the
raw residual-miss metrics did not improve consistently. Relative to the H50
baseline at 100 ms, C1 improved positive-projection p95 on all 20 songs,
while raw miss p95 improved on only 9 songs and had a worst-song regression of
about `4.19 dB`. This indicates more aggressive removal, not reliable
correction of the specific worst leaks.

## Private 12-song analysis

Continuous renders are under:

`data/modern-song-s-combined-pilot/listening-12/`

Variants are `Source-H50-continuation+5`, `C0-combined-control`, and
`C1-combined-stable-S`. Relative to C0, C1 showed:

- coherent retained content: `-1.93 dB`;
- 50 ms positive-projection p95: `-1.08 dB`;
- 100 ms positive-projection p95: `-1.17 dB`;
- 200 ms positive-projection p95: `-1.18 dB`;
- Inst 3 waveform-match SNR: `-0.19 dB`.

The retained-content and projection changes confirm a stronger aggressive
direction. The lower teacher-match SNR and higher target error mean that C1
cannot be described as a uniformly better approximation of Inst 3.

## Interpretation and next step

The expanded pool produces a measurable local change, but it does not yet
establish better full-song quality or safer removal. The experiment should be
kept as a useful S-focused candidate and not promoted over H50 solely from
the aggregate projection numbers. The next bounded experiment is a five-pass
continuation from C1 with its AdamW state restored, alongside an equal-budget
C0 continuation. Keep the same song split, event representatives, 9.09%
external share, and continuous rendering. Compare worst held-out S events and
the 12-song blind listening set; stop if the extra pass budget only increases
aggressiveness without reducing the known audible hotspots. Do not increase
the S fraction or add another song pool in that continuation, so optimization
headroom remains separable from data-distribution changes.

## Reports and listening paths

- `data/modern-song-s-combined-pilot/reports/combined-s-report.json`
- `data/modern-song-s-combined-pilot/reports/private-inst3-analysis.json`
- `data/modern-song-s-combined-pilot/reports/musdb-evaluation.json`
- `data/modern-song-s-combined-pilot/listening-12/C0-combined-control/`
- `data/modern-song-s-combined-pilot/listening-12/C1-combined-stable-S/`
- `data/modern-song-s-combined-pilot/listening-12/Source-H50-continuation+5/`
