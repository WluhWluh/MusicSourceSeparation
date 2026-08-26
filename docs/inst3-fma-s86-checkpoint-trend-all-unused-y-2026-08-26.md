# S86 FMA Checkpoint Trend on Unused Y Songs

Date: 2026-08-26

## Scope

The source population is the all-S86 FMA review list. It contains 158 songs,
all marked `Y`, with five frozen event centres per song. The recorded S86
training selections exclude 48 of those songs, leaving 110 song-disjoint
unused songs and 550 event centres for this evaluation.

The exclusion audit includes the S86 base pool, the S/R continuation pool, the
first FMA S/R continuation selection, and the later FMA leakage-survey
selection. The 30 survey songs outside the 158-song source list are recorded
in the provenance but do not change the 110-song result.

Each metric uses the same global 128-frame, 5+5-hop continuous
overlap-save assembly as inference. Only stride windows intersecting a fixed
50/100/200 ms event block were inferred; an independent check against a full
continuous render differed by at most about `1e-5` residual amplitude in the
tested event regions. No isolated zero-filled event windows were used.

## Checkpoint Trend

The primary value is pooled 100 ms positive projection in dBFS. More negative
is better. `vs source` and `vs prior` are signed differences, so a negative
value is an improvement.

| node | stage / pass | step | 100 ms pooled p95 | vs source | vs prior | mean per-song p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| first-p00-step-3360 | first / 0 | 3360 | -18.717 | +0.000 | -- | -26.645 |
| first-p01-step-3536 | first / 1 | 3536 | -18.619 | +0.098 | +0.098 | -26.291 |
| first-p03-step-3888 | first / 3 | 3888 | -18.337 | +0.380 | +0.282 | -26.225 |
| first-p05-step-4240 | first / 5 | 4240 | -18.174 | +0.543 | +0.163 | -26.197 |
| first-p08-step-4768 | first / 8 | 4768 | -18.289 | +0.428 | -0.115 | -26.191 |
| first-p10-step-5120 | first / 10 | 5120 | -18.383 | +0.334 | -0.094 | -26.212 |
| first-p11-step-5296 | first / 11 | 5296 | -18.420 | +0.297 | -0.037 | -26.230 |
| first-p13-step-5648 | first / 13 | 5648 | -18.562 | +0.155 | -0.141 | -26.260 |
| second-p01-step-5824 | second / 14 | 5824 | -18.802 | -0.085 | -0.241 | -26.738 |
| second-p05-step-6528 | second / 18 | 6528 | -20.777 | -2.060 | -1.975 | -28.948 |
| second-p08-step-7056 | second / 21 | 7056 | -21.136 | -2.419 | -0.359 | -29.802 |
| second-p10-step-7408 | second / 23 | 7408 | -21.222 | -2.504 | -0.085 | -30.056 |

## Interpretation

- The first continuation does not generalize as an improvement on the unused
  songs. Its 100 ms p95 worsens by `0.543 dB` at pass 5 and is still `0.155
  dB` above the source at pass 13. The 50 ms endpoint is essentially flat
  (`+0.021 dB`), while the 200 ms endpoint is also slightly worse (`+0.249
  dB`).
- The second continuation changes the direction. From first-pass-13 to
  second-pass-10, the 100 ms p95 improves by `2.660 dB`; relative to the
  original step-3360 source it improves by `2.504 dB`. The largest single
  100 ms value improves from `-12.605` to `-13.968 dBFS`.
- The improvement is broad but not universal at the song level: 100 of 110
  per-song 100 ms p95 values improve, 9 worsen, and 1 is unchanged at
  second-p10. The gain tapers substantially between second-p08 and
  second-p10 (`0.085 dB` pooled).
- The 50 and 200 ms p95 values move in the same direction at the second-stage
  endpoint: `-17.564 -> -19.936 dBFS` and `-20.212 -> -23.644 dBFS`.

## Limits

These are five fixed, previously screened candidate events per song rather
than an unbiased full-song benchmark. The projection is a teacher-directed
leakage proxy relative to Inst 3, not an accompaniment-preservation score.
Because this run infers only event-intersecting windows, it intentionally does
not report whole-song SDR, seam quality, or listening quality. Full continuous
renders and accompaniment-safety checks remain necessary before selecting a
checkpoint for release or further restoration training.

## Artifacts

- JSON: `data/modern-song-fma-s86-checkpoint-trend-all-unused-y/checkpoint-trend-report.json`
- Evaluator: `tools/evaluate_fma_s86_checkpoint_trend.py`
- Checkpoint metadata and all per-song/per-event rows are stored in the JSON.
