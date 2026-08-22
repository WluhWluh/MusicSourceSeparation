# Inst 3 V-R Miss-Driven Sampling Experiment

Date: 2026-08-22

This is a local, non-commercial MUSDB18 experiment. It continues the best
current V-R-H50 residual-vocals checkpoint and changes only the four hard
draws per training song. No checkpoint, teacher-derived audio, or MUSDB18
cache from this experiment is intended for publication.

## Question

Can the current H50 model's actual remaining Inst 3-directed miss identify
better training windows than the frozen Stage 1 hard-event score?

For each Stage 1 candidate, the scan computes:

```text
teacherResidualMiss = (mixtureGt - H50Instrumental) - H50Residual
                    = H50Instrumental - Inst3Instrumental
```

The candidate score is the largest positive projection of this miss onto the
Inst 3 removed signal over 50, 100, and 200 ms blocks. Raw miss RMS is used as
the tie-break. The selection is frozen before training.

## Fixed Contract

| Item | Value |
| --- | --- |
| Train split | 80 MUSDB18 train songs |
| Evaluation split | 10 calibration + 10 internal-test songs |
| Official final test | Not used |
| Source checkpoint | V-R-H50 `step-8000.pt`, source pass 50 |
| Student semantic | Residual vocals |
| Teacher target | `mixtureGt - Inst3Instrumental` |
| Windows per song | 8: 4 uniform + 4 hard/miss-driven |
| Passes | 5 |
| Updates | 800 per arm, 160 per pass |
| Batch size | 4 |
| Learning rate | `1e-5`, inherited from H50 |
| Optimizer | AdamW state inherited from H50 |
| BatchNorm | Running statistics frozen |
| Seed | 891 |
| Device | CUDA, NVIDIA GeForce RTX 4060 Laptop GPU |
| Teacher contract | `uvr_mdxnet_inst_3@2` |

The control arm is `H50-control`: the existing four uniform plus four Stage 1
hard windows. The experiment arm is `H50-miss-driven`: the same four uniform
draws plus four windows ranked from the current H50 miss.

Four short songs had fewer than eight distinct candidate windows. To preserve
the fixed eight-draw budget, the runner repeats available non-uniform
miss-driven candidates deterministically. This affects selection multiplicity
only; it does not create additional audio or cache data.

## Results

Aggregate values below are from the 20 song-disjoint calibration/internal-test
songs. Deltas are `H50-miss-driven@pass-5` minus `H50-control@pass-5`; for dB
projection values, a negative delta means less residual vocal projection.

| Metric | H50 control | Miss-driven | Delta |
| --- | ---: | ---: | ---: |
| Instrumental SDR | 13.786 dB | 13.699 dB | -0.078 dB |
| Low-vocal instrumental SDR | 19.272 dB | 18.795 dB | -0.477 dB |
| Accompaniment vocal projection | -16.606 dB | -16.875 dB | -0.243 dB |
| 50 ms positive projection p95 | -33.608 dBFS | -34.102 dBFS | -0.494 dB |
| 100 ms positive projection p95 | -34.093 dBFS | -34.492 dBFS | -0.398 dB |
| 200 ms positive projection p95 | -34.439 dBFS | -35.124 dBFS | -0.685 dB |
| 50 ms positive projection maximum | -18.532 dBFS | -18.537 dBFS | -0.006 dB |
| 50 ms raw miss RMS maximum | -14.723 dBFS | -13.804 dBFS | +0.919 dB |

The exact report records slightly more precise values. Across individual
held-out songs:

| Comparison | Songs improved |
| --- | ---: |
| Accompaniment vocal projection | 19/20 |
| 50 ms positive-projection p95 | 18/20 |
| 100 ms positive-projection p95 | 20/20 |
| 200 ms positive-projection p95 | 18/20 |
| 50 ms positive-projection maximum | 19/20 |
| 50 ms raw miss maximum | 10/20 |
| Instrumental SDR | 5/20 |

Positive-projection hotspots above -30 dBFS also fell from 266 to 234 at 50
ms, 132 to 115 at 100 ms, and 63 to 54 at 200 ms. Hotspots above -35 dBFS
fell from 764 to 701, 371 to 325, and 161 to 147 respectively.

The result supports the intended direction: miss-driven sampling teaches the
student to remove more of the Inst 3-directed residual in the ordinary tail
of events. It does not yet solve the worst single short miss, and it costs
some low-vocal instrumental safety margin. The raw miss maximum becoming
larger is the main reason this should remain an experiment rather than replace
V-R-H50 based on metrics alone.

The objective improvement is not sufficient evidence of a useful listening
improvement. In particular, the positive-projection p95 reductions do not
guarantee that the few most salient abrupt syllable leaks have been reduced.

There were no non-finite evaluation windows or clipped samples in the
aggregate results. The first-difference mechanical-artifact proxy did not
show a training failure, but it is only a diagnostic and cannot replace
listening.

## Listening

The runner generated 36 PCM16 FLAC files for 12 private songs under:

```text
data/musdb18-inst3-vr-miss-driven/listening-12/
```

They include `H50-pass-50`, `H50-control@pass-5`, and
`H50-miss-driven@pass-5`. This directory is local generated data and remains
outside Git. The report's listening section contains the file hashes and
source-song identities.

### User Listening Review

The 12-song full-length comparison found no clearly audible overall
improvement over H50. Medium-strength residual-vocal details traded wins and
losses between the two models. The most abrupt and obvious residual-vocal
hotspots were not improved, and some songs made those leaks more noticeable in
the miss-driven result. No consistent perceptual benefit was established.

This review outweighs the aggregate p95 improvement for model selection. It
shows that the current miss score is better at describing the ordinary tail of
the Inst 3-directed error than the salient worst-case leak that determines the
listening preference.

## Reproduction

The experiment runner and its unit tests are:

- `tools/run_inst3_vr_miss_driven.py`
- `tests/test_inst3_vr_miss_driven.py`

The CUDA run was:

```powershell
.\\.tmp\\vr-hard-env\\Scripts\\python.exe tools\\run_inst3_vr_miss_driven.py `
  --device cuda --threads 8 --passes 5 --milestones 0,1,2,5 `
  --output-root data\\musdb18-inst3-vr-miss-driven --no-resume
```

The machine-readable result is:

```text
data/musdb18-inst3-vr-miss-driven/reports/inst3-vr-miss-driven-report.json
```

## Decision

Keep V-R-H50 as the current best listening baseline. Do not promote
`H50-miss-driven@pass-5` or spend more runs increasing its sampling fraction.
The experiment is useful as a negative result: it improved aggregate
projection-tail metrics without producing a reliable perceptual improvement,
and some private songs regressed audibly.

The next direction should target salient worst-case events rather than the
global miss tail. First freeze a small, independently selected hard-event
evaluation set using the actual 12-song listening review and objective
worst-hotspot ranking, then test an event-local objective that preserves the
H50 output outside the event. The event target should include a short audio
context around each hotspot and use a sample-aligned loss, not only a window
selection score. Compare three arms from the same H50 checkpoint:

1. H50 continuation control.
2. H50 output anchor outside the event plus Inst 3 residual target inside the
   event context.
3. The same event-local target with an explicit short-block loss on the
   center 50-100 ms.

Use a very small budget first, such as 1-5 passes, and require both a
held-out-hotspot improvement and a level-matched 12-song listening win. If
the event-local model still cannot change the salient leaks, the limitation is
likely the 128-frame TFC-TDF representation or its spectral/iSTFT contract,
not the sampling schedule. At that point the next meaningful route is a
separately trained shorter-context or higher-capacity model, rather than more
fine-tuning of the current checkpoint.
