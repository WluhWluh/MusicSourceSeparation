# Inst 3 W-D continuation

Status: complete as a local, non-commercial research experiment. The
continuation does not establish a releasable model. MUSDB18-derived audio,
Inst 3 outputs, checkpoints, and generated files remain under the ignored
`data` tree and must not be uploaded to `bss-tflite` or integrated into
Booming SS.

Runner: [run_inst3_wd_continuation.py](../tools/run_inst3_wd_continuation.py)

Tests: [test_inst3_wd_continuation.py](../tests/test_inst3_wd_continuation.py)

Listening renderer: [render_inst3_wd_listening.py](../tools/render_inst3_wd_listening.py)

Listening renderer tests: [test_inst3_wd_listening.py](../tests/test_inst3_wd_listening.py)

Machine-readable result (ignored local artifact):

`data/musdb18-inst3-wd-continuation/reports/inst3-wd-continuation-report.json`

## Question

The initialization/output matrix showed that `W-D` can learn a direct
instrumental output, but it remained behind the pretrained residual-output
cell at 100 passes. This bounded experiment tests whether continuing the same
`W-D` state from 100 to 200 passes produces enough improvement to justify more
seeds and later experiments.

Only `seed-891` was required initially. The other two seeds were continued
only after the primary seed passed the predefined gate.

## Frozen contract

- Parent: completed `W-D` state at 100 passes from the initialization/output
  matrix; the parent state was read-only input.
- Variant: pretrained TFC-TDF body with a reset direct-instrumental output
  head.
- Training selection: the same 10 fixed songs and 320 windows as the parent.
- Holdout: 16 windows per selected training song.
- Evaluation: the same 10 calibration and 10 internal-test songs, 16 windows
  per song; official MUSDB18 final-test songs were not used.
- Student: 128-frame FP32 TFC-TDF.
- Target: direct `uvr_mdxnet_inst_3@2` instrumental output under the frozen
  audio-domain teacher contract.
- Optimizer: the parent AdamW state, learning rate `1e-4`, weight decay `0`.
- Batch size: `4`; gradient clipping: norm `1.0`.
- BatchNorm: continued in train mode, preserving the parent W-D policy.
- Schedule: the deterministic 200-pass schedule preserves the exact first
  100-pass parent prefix; no new sampling policy was introduced.
- Milestones: 100, 125, 150, and 200 passes; 80 updates per pass.
- Execution: CUDA on an NVIDIA GeForce RTX 4060 Laptop GPU, with model,
  AdamW, and RNG state persisted for exact resume.

The runner also reconstructs the original direct-head initialization before
loading the parent state, so parameter-drift and output-change diagnostics
remain relative to the real W-D origin.

## Gate

The primary seed was considered a pass only when all three conditions held
between 100 and 200 passes:

1. Calibration/internal aggressive-target SDR gain is at least `+0.5 dB`.
2. Calibration/internal instrumental SDR does not decline by more than
   `0.1 dB`.
3. Training-holdout aggressive-target SDR does not decline by more than
   `0.1 dB`.

## Results

Values are calibration plus internal-test aggregates. Deltas are 200-pass
value minus 100-pass value.

| Seed | Target SDR at 100 | Target SDR at 200 | Target delta | Instrumental SDR delta | Holdout target delta | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 891 | 13.285 | 13.795 | +0.509 | +0.325 | +0.750 | pass |
| 1891 | 13.619 | 13.787 | +0.169 | +0.126 | +0.328 | target gate failed |
| 2891 | 13.181 | 13.596 | +0.415 | +0.259 | +0.401 | target gate failed |

The primary-seed decision is therefore `passed`, so the two remaining seeds
were run. The three-seed continuation is not a passed gate: the target gain
is not consistent across initializations.

### Mean trend across seeds

| Pass | Aggressive-target SDR | Instrumental SDR | Teacher-residual SDR | Teacher spectrum L1 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 13.362 | 11.449 | 7.359 | 0.06485 |
| 125 | 13.466 | 11.515 | 7.463 | 0.06474 |
| 150 | 13.544 | 11.574 | 7.542 | 0.06257 |
| 200 | 13.726 | 11.685 | 7.724 | 0.05968 |

The mean target gain is about `+0.364 dB`. All three seeds improve the
instrumental safety metric and reduce teacher spectrum error, but only one
seed reaches the `+0.5 dB` target-gain threshold. The later checkpoints are
therefore useful research observations, not a stable direct-instrumental
model qualification.

## Interpretation

The bounded continuation answers the narrow question positively for one
initialization: `seed-891` still had a small useful learning curve after 100
passes. It does not justify assuming that longer W-D training will reliably
produce the desired Inst 3-like behavior. Seed variation remains large enough
that the three-seed gate fails.

The result also does not show a quality regression caused by continuation.
Instrumental SDR improved for every seed, and teacher spectrum L1 decreased
for every seed. Those are safety and optimization diagnostics, not a
replacement for level-matched listening on the 12-song private set.

## User listening review

The `seed-891` 200-pass direct-instrumental output was rendered against the
initial model and the 100/125/150/200-pass checkpoints on the same twelve-song
private full-song set. The renderer produced 72 PCM16 FLAC files with matching
44.1 kHz stereo frame counts. The files remain under the ignored local tree:

`data/musdb18-inst3-wd-continuation/listening-seed-891/`

The user's review was:

- On some songs, the W-D 200-pass accompaniment has intermittent short
  scratch-like mechanical artifacts.
- Most of the otherwise normally separated regions still contain noticeably
  more vocal residue than `initial-vocals`.
- The residue is softer and less abrupt than the frequent complete short
  syllable leaks heard in the earlier initial and scale-10 experiments.
- Overall usability remains below `initial-vocals`, but the separation style
  is perceptibly a little closer to Inst 3.

This is consistent with the numerical result: W-D moved toward the direct
Inst 3 target, but the output quality and artifact rate are not good enough to
qualify the checkpoint for product use. The listening result is a product
development observation, not a release or generalization claim.

## Disposition

- Do not publish the checkpoints or teacher-derived audio.
- Do not export the 200-pass checkpoint to ONNX/TFLite or qualify it in
  Booming SS.
- Do not spend more compute on unrestricted W-D continuation under this
  contract.
- If direct instrumental output remains the goal, the next experiment should
  target the remaining short vocal-event errors with measured hard-window
  sampling or a staged output-head warm-up, while retaining the pretrained
  body and using the same held-out evaluation and listening protocol.

Reproduction command for the completed run:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe `
  tools\run_inst3_wd_continuation.py `
  --device cuda --threads 8 --seeds 891,1891,2891
```
