# Inst 3 W-D hard-window sampling plan

Status: planned research experiment. No training was started by this plan.
The goal is to reduce short, abrupt vocal leaks while retaining the pretrained
TFC-TDF initialization. It does not authorize product integration or weight
publication.

## Motivation

The W-D continuation moved the student slightly toward the aggressive Inst 3
style. The `seed-891` listening review found softer residual vocals than the
initial and scale-10 students, but also intermittent short scratch-like
mechanical artifacts and overall usability below the initial model. Uniformly
repeating the same 320 windows did not reliably improve the target across
seeds. The next controlled variable should therefore be window selection,
with emphasis on short events rather than another unrestricted continuation.

## Frozen base

- Student: the pretrained `vocals_epoch=891.ckpt` TFC-TDF model.
- Output semantic: direct instrumental, as in W-D; reset only the final output
  head using the existing W-D initialization contract.
- Target: direct `uvr_mdxnet_inst_3@2` instrumental target in the existing
  audio-domain teacher contract.
- Student window: the existing 128-frame FP32 TFC-TDF contract.
- Data: the same ten selected MUSDB18 training songs and 320 candidate
  windows, with the same song-disjoint holdout and calibration/internal-test
  sets. Official final-test songs remain excluded from training and tuning.
- Optimizer and numerics: AdamW, learning rate `1e-4`, zero weight decay,
  batch size `4`, gradient clip `1.0`, FP32, CUDA-capable host.
- BatchNorm: keep the W-D policy, running statistics updated in train mode.
- Seeds: run `891` first, then `1891` and `2891` only if the primary run is
  numerically stable and shows a meaningful hard-event improvement.

The W-D 100-pass checkpoint is the preferred starting state for the first
sampling ablation because it avoids conflating output-head warm-up with the
sampling question. A separate fresh-reset-head control must be retained for
the final comparison.

## Event score

For every candidate training window, compute a deterministic hard-event score
from already cached teacher and student data. The score must be independent of
the private external listening songs:

1. Reconstruct the teacher instrumental and the current student instrumental
   in the audio domain using the existing overlap/trim contract.
2. Define teacher removed content as `mixture - teacherInstrumental`.
3. Partition each window into 50 ms, 100 ms, and 200 ms blocks.
4. Mark blocks active when teacher-removed RMS exceeds the existing `-60 dBFS`
   diagnostic floor.
5. Rank by a deterministic combination of:
   - positive projection of student error onto teacher-removed content;
   - teacher-removed RMS, so silent regions do not dominate;
   - a short-event multiplier that favors 50-200 ms peaks over broad average
     energy.

The initial score should be calculated from the W-D 100-pass checkpoint and
then frozen for the main ablation. A later adaptive-refresh variant may
recompute scores at a documented milestone, but it must be a separate cell.

## Sampling matrix

Keep the total optimizer updates and the 320-window candidate pool constant.
Compare the following cells from the same W-D 100-pass state, with identical
schedule seeds and evaluation:

| Cell | Uniform share | Hard-event share | Purpose |
| --- | ---: | ---: | --- |
| `U` | 100% | 0% | continuation control |
| `H25` | 75% | 25% | low-risk hard-event emphasis |
| `H50` | 50% | 50% | primary candidate |
| `H75` | 25% | 75% | aggressive stress case |

Hard-event draws must be sampled with replacement from a frozen top-ranked
pool, while uniform draws use the same seeded permutation stream. Record the
actual per-update window IDs and schedule hash so a run can be resumed and
audited. Do not silently replace the whole dataset with the top windows.

For the first pilot, train each cell for 25 and 50 additional passes, with
checkpoints at 0/25/50. This is enough to expose whether hard sampling reduces
short-event error before spending another 100-pass budget. Use the same
optimizer state and W-D 100-pass initialization for each cell; do not share
optimizer state between cells.

## Evaluation

Report all existing aggregate metrics, but add event-specific metrics on the
song-disjoint calibration/internal-test set:

- 50/100/200 ms retained teacher-residual p95 and maximum;
- count and energy-weighted mean of the top 12 residual hotspots per song;
- vocal projection in active hard-event blocks;
- instrumental SDR and low-vocal instrumental SDR as safety metrics;
- teacher spectrum L1 and output peak/clipping count;
- block-to-block discontinuity and mechanical-artifact proxy.

The private 12-song listening set remains evaluation-only and must be rendered
only after a candidate passes the numerical smoke checks. Compare against:

- `initial-vocals`;
- the four-song `alpha=1.00-step-2048` listening reference;
- the ten-song scale-10 `alpha=1.00-step-8192` reference;
- `seed-891` W-D 200-pass output.

Do not use known external listening timestamps to choose training windows or
set thresholds. They may be recorded after the run as a blind evaluation
annotation only.

## Go/no-go gates

Advance a hard-sampling cell only when, on both calibration and internal-test
aggregates:

1. 100 ms retained teacher-residual p95 improves by at least `0.5 dB` versus
   the uniform control, with the same sign on the 50 ms and 200 ms metrics.
2. The top-event maximum does not worsen by more than `0.5 dB`.
3. Instrumental SDR and low-vocal instrumental SDR do not decline by more
   than `0.2 dB`.
4. No new sustained clipping or non-finite output occurs.
5. Blind listening confirms fewer abrupt syllable leaks without introducing
   more scratch-like mechanical artifacts.

If H25/H50 improves short-event metrics without a listening regression, extend
only the best cell to the three seeds and a longer 100-pass continuation. If
all hard ratios fail, stop this sampling direction and investigate the W-D
direct-head artifact mechanism instead of increasing training time.

## Required artifacts

- frozen hard-window score manifest with source/model/teacher hashes;
- per-cell schedule and contract JSON;
- resumable model/optimizer/RNG checkpoints;
- event-level numerical report;
- private listening render report and files, kept under ignored `data`;
- no ONNX/TFLite export and no `bss-tflite` upload.
