# Inst 3 instrumental distillation pilot

Status: complete as a local, non-commercial pilot. The data and checkpoints
remain in the ignored `data` tree and are not suitable for product release.

## Purpose

This stage compares two instrumental-output variants initialized from the
public `vocals_epoch=891.ckpt` neural core:

- `S0-ground-truth`: MUSDB18 instrumental ground-truth loss only.
- `S1-ground-truth-plus-inst3`: the same loss plus `0.10 * Inst 3` soft-target
  loss.

The neural core still estimates vocals. Both variants define their student
output as:

```text
instrumental_spec = mixture_gt_spec - neural_vocal_spec
```

The teacher input and student input are the exact stem sum
`mixture_gt = vocals + drums + bass + other`. The encoded mixture stream is
decoded and retained only as a codec-error control; it is not used as the
training input.

## Frozen scope

The runner is
`tools/run_inst3_distill_pilot.py` at the SHA-256 recorded in the machine
report. It reads the frozen manifest
`data/musdb18-inst3-oracle/musdb18-inst3-oracle-manifest.json` and selected
exactly:

| Role | Song |
| --- | --- |
| train | The So So Glos - Emergency |
| train | Dark Ride - Burning Bridges |
| calibration | Lushlife - Toynbee Suite |
| internal-test | Triviul - Dorothy |

The official `final-test` split was not extracted, rendered, or evaluated.
Each song used a 30-second segment beginning at 15 seconds. Four training
windows per training song and five evaluation windows per evaluation song
were selected on the highest mixture-RMS candidates. This selection policy is
deliberately recorded because it is too small and potentially unrepresentative
for a quality claim.

The teacher identity is `uvr_mdxnet_inst_3@2`, with ONNX SHA-256:

```text
2b7834e2972158d8c9864e7376e3a7d084079c80a23f38dc31c4b0a4e901a1cb
```

The frozen student checkpoint has SHA-256:

```text
101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d
```

The complete machine-readable evidence is:
`data/musdb18-inst3-student-pilot/reports/inst3-distill-pilot-report.json`.

## Reproduction

The run used the local CUDA environment and required the ONNX Runtime CUDA
provider explicitly:

```powershell
C:\Users\User\AppData\Local\MusicSourceSeparation\musdb18-inst3-oracle-venv\Scripts\python.exe `
  tools/run_inst3_distill_pilot.py `
  --require-teacher-cuda
```

Environment observed in the report:

- PyTorch `2.11.0+cu128`
- ONNX Runtime `1.26.0`
- NVIDIA GeForce RTX 4060 Laptop GPU
- Teacher providers: `CUDAExecutionProvider`, `CPUExecutionProvider`
- Student device: `cuda`

The run rendered all four 30-second teacher segments without failed or
non-finite windows. The residual reconstruction maximum error was about
`3.0e-8` for the teacher and both students.

## Aggregate evaluation

The reported SDR values are energy-SNR proxies, not SI-SDR. Vocal projection
is the projection of instrumental error onto the reference vocal stem; a more
negative value is preferable. The evaluation contains 1,198,080 stereo sample
frames from the selected calibration and internal-test windows.

| Variant | Instrumental energy-SNR | Vocal energy-SNR | Vocal projection | Low-vocal energy-SNR | Error RMS (dBFS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial checkpoint | 14.995 dB | 11.940 dB | -24.018 dB | 44.264 dB | -38.628 dBFS |
| Inst 3 teacher | 4.745 dB | 1.691 dB | -29.194 dB | 0.482 dB | -28.379 dBFS |
| S0 ground truth | 1.188 dB | -1.866 dB | -15.348 dB | -2.860 dB | -24.821 dBFS |
| S1 ground truth + Inst 3 | 1.178 dB | -1.876 dB | -15.340 dB | -2.869 dB | -24.812 dBFS |

Relative to S0, S1 changed the aggregate metrics by:

- instrumental energy-SNR: `-0.0097 dB`;
- vocal energy-SNR: `-0.0097 dB`;
- vocal projection: `+0.0075 dB`, slightly worse;
- low-vocal energy-SNR: `-0.0091 dB`.

These differences are far below a meaningful listening or quality threshold.
The S0/S1 checkpoints also degraded substantially from the initial checkpoint
after only 64 updates over eight selected training windows. The calibration
song has almost no vocal energy in this segment, and the highest-mixture-RMS
selection is therefore not a sufficient training/evaluation design for the
intended vocal-removal use case. This is an experimental limitation, not
evidence that Inst 3 distillation is intrinsically harmful.

## Gate decision

The engineering/data path passes:

- frozen manifest selection and source hashes;
- exact `mixture-gt` construction;
- Inst 3 CUDA execution and residual identity;
- instrumental target semantics through training and evaluation;
- checkpoint and report identity capture;
- final-test isolation.

The quality gate does not pass. Do not use either pilot checkpoint in Booming
SS, publish it to `bss-tflite`, start 24-frame export, or expand teacher
rendering to the full 80-song training split.

## Recommended next experiment

Before another distillation comparison, run a stability sweep using the same
frozen songs but a representative, vocal-activity-stratified window set:

1. Include a zero-update baseline and evaluate after 1, 4, 8, 16, and 32
   updates.
2. Compare learning rates `1e-6`, `3e-6`, and `1e-5`, with gradient clipping
   and a held-out window set from each training song.
3. Stratify windows by vocal RMS and mixture RMS so silent-vocal and active-
   vocal material are both present.
4. Keep S0 and S1 identical in every respect except the teacher loss, then
   require a repeatable improvement on accompaniment vocal projection without
   an increase in low-vocal instrumental error.

Only a stable result in that controlled sweep justifies a larger song-level
training run. All MUSDB18-derived audio, teacher outputs, and derived weights
must remain local under the existing non-commercial restrictions.
