# Inst 3 distillation stability sweep

Status: complete as a local, non-commercial research experiment. The sweep
does not establish a releasable model and its generated audio/checkpoints stay
under the ignored `data` tree.

## Purpose

The first four-song pilot showed severe degradation after only 64 updates. It
used a very small highest-mixture-energy window set and left the model in
training mode, so BatchNorm population statistics were updated from one
window at a time. This sweep isolates those issues while preserving the same
frozen song and teacher identities.

It compares, at three learning rates:

- `S0-ground-truth`: instrumental ground-truth loss only;
- `S1-ground-truth-plus-inst3`: the same loss plus `0.10 * Inst 3` soft-target
  loss.

Both variants retain the existing vocals neural core and define the output as
`mixture_gt_spec - neural_vocal_spec`.

## Controls

Runner: [run_inst3_distill_stability_sweep.py](../tools/run_inst3_distill_stability_sweep.py)

The frozen manifest and teacher are the same as the preceding pilot:

- manifest: `musdb18-inst3-oracle-split@1`;
- songs: two `train`, one `calibration`, and one `internal-test`;
- official `final-test`: not extracted or evaluated;
- input: `mixture-gt = vocals + drums + bass + other`;
- teacher contract: `uvr_mdxnet_inst_3@2`;
- teacher ONNX SHA-256:
  `2b7834e2972158d8c9864e7376e3a7d084079c80a23f38dc31c4b0a4e901a1cb`.

Each 30-second song segment was evaluated across all 12 contiguous windows,
including the short final tail. Training used 8 activity-stratified windows
per training song, selected using both vocal RMS and mixture RMS. Three
non-tail windows per training song were held out. Thus the run used 16 train,
6 train-held-out, and 24 full evaluation windows.

The student model was kept in evaluation mode during updates, so BatchNorm
running statistics remained frozen. Gradients to trainable parameters were
still enabled. Other controls were fixed: AdamW with zero weight decay,
gradient norm clip 1.0, seed 891, 32 updates, and milestone evaluation at
`0/1/4/8/16/32` updates.

The run required CUDA and used the RTX 4060 environment. The teacher reported
`CUDAExecutionProvider, CPUExecutionProvider`; the student ran on `cuda`.
Machine-readable evidence is at:

`data/musdb18-inst3-stability-sweep/reports/inst3-distill-stability-sweep-report.json`

## Baseline

Metrics below are energy-SNR proxies. Vocal projection is the projection of
instrumental error onto the reference vocal stem; a higher value is worse
(less negative means more vocal-correlated error).

| Evaluation | Instrumental energy-SNR | Vocal energy-SNR | Vocal projection | Low-vocal energy-SNR |
| --- | ---: | ---: | ---: | ---: |
| Full calibration + internal-test | 15.476 dB | 11.602 dB | -23.320 dB | 43.028 dB |
| Training held-out windows | 14.327 dB | 7.249 dB | -14.656 dB | 29.068 dB |

## Full evaluation at 32 updates

| Variant | LR | Instrumental energy-SNR | Vocal energy-SNR | Vocal projection | Low-vocal energy-SNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 1e-6 | 15.483 dB | 11.609 dB | -23.166 dB | 43.039 dB |
| S1 | 1e-6 | 15.484 dB | 11.609 dB | -23.166 dB | 43.039 dB |
| S0 | 3e-6 | 15.490 dB | 11.616 dB | -22.906 dB | 43.044 dB |
| S1 | 3e-6 | 15.491 dB | 11.617 dB | -22.909 dB | 43.044 dB |
| S0 | 1e-5 | 15.484 dB | 11.609 dB | -22.477 dB | 43.041 dB |
| S1 | 1e-5 | 15.487 dB | 11.612 dB | -22.505 dB | 43.054 dB |

Relative to the zero-update full baseline, all configurations retain the
instrumental and vocal energy-SNR within about `0.02 dB`. However, vocal
projection becomes worse by approximately `0.15 dB`, `0.41 dB`, and `0.84 dB`
for S0 at `1e-6`, `3e-6`, and `1e-5` respectively. The held-out windows show
the same direction. This is a small but consistent movement toward more
vocal-correlated accompaniment error, not the desired outcome.

## Distillation effect

At 32 updates, S1 minus S0 on the full evaluation was:

| LR | Instrumental energy-SNR | Vocal projection | Low-vocal energy-SNR |
| ---: | ---: | ---: | ---: |
| 1e-6 | `+0.00025 dB` | `-0.00008 dB` | `+0.00475 dB` |
| 3e-6 | `+0.00077 dB` | `-0.00266 dB` | `+0.01094 dB` |
| 1e-5 | `+0.00294 dB` | `-0.02839 dB` | `+0.01274 dB` |

The negative projection differences technically favor S1, but they are far
below an audible or useful quality threshold. The teacher loss therefore did
not demonstrate a meaningful benefit in this setup.

Parameter drift stayed small: at 32 updates it ranged from approximately
`6.45e-5` (`1e-6`, S0) to `5.27e-4` (`1e-5`, S0). The corresponding held-out
metrics did not show catastrophic overfitting. This confirms that the earlier
pilot's large collapse was primarily related to its tiny/unstable fine-tuning
setup, but it does not validate the current distillation objective.

## Gate decision

The engineering controls pass:

- CUDA teacher execution and frozen source identity;
- activity-stratified training selection;
- disjoint held-out windows;
- full calibration/internal-test segment evaluation;
- BatchNorm running-statistics freeze;
- milestone, parameter-drift, and output-change logging;
- final-test isolation.

The quality gate does not pass. There is no meaningful S1 improvement, and
fine-tuning increases vocal-correlated accompaniment error as learning rate
and update count grow. Do not publish these checkpoints, use them in Booming
SS, export a 24-frame candidate, or expand to the full 80-song training split.

## Next experiment

The next useful test is objective alignment rather than more learning-rate
search: add a small anchor loss to preserve the initial checkpoint output and
an explicit vocal-projection penalty on active-vocal windows. Compare S0 and
S1 with teacher weights `0.01/0.03/0.10`, keeping the same frozen windows and
milestones. If that still cannot improve projection without harming
low-vocal instrumental preservation, Inst 3 should be dropped as a teacher
for this compact student.
