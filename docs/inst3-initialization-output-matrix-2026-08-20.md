# Inst 3 initialization and output-semantics matrix

Status: complete as a local, non-commercial research experiment. The matrix
does not establish a releasable model. MUSDB18-derived audio, Inst 3 teacher
outputs, checkpoints, and all generated files remain under the ignored
`data` tree and must not be uploaded to `bss-tflite`.

Runner: [run_inst3_initialization_output_matrix.py](../tools/run_inst3_initialization_output_matrix.py)

Machine-readable report:

`data/musdb18-inst3-init-output-matrix/reports/inst3-init-output-matrix-report.json`

## Question

The preceding experiments used the pretrained vocals-residual TFC-TDF model
and changed its training target. This matrix tests whether the observed gap is
caused mainly by the initialization, by the output semantic, or by both. It
keeps the direct Inst 3 instrumental target fixed and compares four practical
cells:

| Cell | Initialization | Neural output semantic | Instrumental prediction |
| --- | --- | --- | --- |
| `W-R` | pretrained `vocals_epoch=891.ckpt` | residual vocals | mixture spectrum minus neural output |
| `W-D` | pretrained body, output head reset | direct instrumental | neural output |
| `R-R` | seeded random | residual vocals | mixture spectrum minus neural output |
| `R-D` | seeded random | direct instrumental | neural output |

`W-D` resets the final output convolution because a pretrained residual head
cannot be reinterpreted as a direct instrumental head without changing its
meaning. Therefore this is a practical initialization/output-cell comparison,
not a claim that output semantic alone is the only changed variable in `W-D`.
The `R-R` versus `R-D` pair provides the cleanest output-semantic comparison
under the same random neural initialization.

## Frozen contract

- Manifest: `musdb18-inst3-oracle-split@1`.
- Training selection: 10 fixed songs from the 80-song training split.
- Training windows: 32 per song, 320 total.
- Training holdout: 16 windows per selected training song.
- Evaluation: 10 calibration and 10 internal-test songs, 16 windows per song.
- Official MUSDB18 `final-test` songs were not extracted, loaded, or evaluated.
- Target: direct `uvr_mdxnet_inst_3@2` instrumental output in the audio-domain
  teacher contract, converted to the student STFT contract.
- Student: 128-frame TFC-TDF, FP32.
- Seeds: `891`, `1891`, and `2891`.
- Batch size: 4.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0`.
- Gradient clipping: norm `1.0`.
- BatchNorm: running statistics were updated in train mode for every cell.
- Schedule: every pass is a complete seeded permutation of all 320 windows.
- Milestones: 0, 25, 50, and 100 complete passes.
- Updates: 80 per pass, 8,000 total per cell.
- Execution: CUDA on an NVIDIA GeForce RTX 4060 Laptop GPU.
- Determinism: deterministic cuDNN settings, persisted optimizer/model/RNG
  state, and exact resume support.

The report records the manifest and selection hashes, initial checkpoint hash,
runner hash, per-cell schedule identity, BatchNorm summaries, and checkpoint
metadata. All 12 runs completed and all four cells have all four milestones.

## Metric definitions

The tables below report the aggregate calibration plus internal-test result.
Values are `mean +/- sample standard deviation` across the three seeds.

- `Aggressive target SDR`: energy-SNR against the Inst 3 instrumental target;
  higher means closer to the aggressive teacher behavior.
- `Teacher-residual SDR`: energy-SNR between the candidate removed residual
  and the residual removed by Inst 3; higher means closer to the teacher.
- `Instrumental SDR`: conventional energy-SNR against the MUSDB18 instrumental
  stem; this is a safety diagnostic, not the sole selection criterion.
- `Low-vocal instrumental SDR`: conventional instrumental score on low-vocal
  windows; higher is better for preserving accompaniment outside vocal events.
- `Vocal projection`: vocal-correlated accompaniment error proxy; more negative
  is better.
- `Teacher spectrum L1`: mean absolute student/teacher STFT error; lower is
  closer to the teacher.

These are diagnostic proxies. They do not replace level-matched blind
listening, and they do not predict Android performance.

## Pass-wise results

### `W-R`

| Pass | Aggressive target SDR (dB) | Teacher-residual SDR (dB) | Instrumental SDR (dB) | Low-vocal SDR (dB) | Vocal projection (dB) | Teacher spectrum L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16.522 +/- 0.000 | 10.520 +/- 0.000 | 13.913 +/- 0.000 | 35.431 +/- 0.000 | -17.103 +/- 0.000 | 0.04159 +/- 0.00000 |
| 25 | 16.637 +/- 0.052 | 10.635 +/- 0.052 | 13.795 +/- 0.029 | 32.747 +/- 0.450 | -14.931 +/- 0.741 | 0.03841 +/- 0.00054 |
| 50 | 16.621 +/- 0.031 | 10.619 +/- 0.031 | 13.741 +/- 0.004 | 33.685 +/- 1.224 | -14.278 +/- 0.242 | 0.03824 +/- 0.00023 |
| 100 | 16.542 +/- 0.042 | 10.539 +/- 0.042 | 13.687 +/- 0.027 | 34.672 +/- 0.656 | -14.385 +/- 0.417 | 0.03786 +/- 0.00035 |

### `W-D`

| Pass | Aggressive target SDR (dB) | Teacher-residual SDR (dB) | Instrumental SDR (dB) | Low-vocal SDR (dB) | Vocal projection (dB) | Teacher spectrum L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.025 +/- 0.014 | -6.028 +/- 0.014 | -0.025 +/- 0.013 | -0.000 +/- 0.000 | -34.470 +/- 8.041 | 0.40229 +/- 0.03126 |
| 25 | 6.403 +/- 1.348 | 0.401 +/- 1.348 | 6.039 +/- 1.207 | 7.344 +/- 1.544 | -15.424 +/- 1.318 | 0.13093 +/- 0.01249 |
| 50 | 12.422 +/- 0.453 | 6.420 +/- 0.453 | 10.852 +/- 0.310 | 16.617 +/- 1.450 | -12.471 +/- 0.172 | 0.07948 +/- 0.00529 |
| 100 | 13.362 +/- 0.229 | 7.359 +/- 0.229 | 11.449 +/- 0.126 | 20.747 +/- 1.274 | -12.457 +/- 0.549 | 0.06485 +/- 0.00328 |

### `R-R`

| Pass | Aggressive target SDR (dB) | Teacher-residual SDR (dB) | Instrumental SDR (dB) | Low-vocal SDR (dB) | Vocal projection (dB) | Teacher spectrum L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.005 +/- 0.003 | 0.003 +/- 0.003 | 5.660 +/- 0.003 | 46.464 +/- 2.033 | -0.003 +/- 0.003 | 0.16333 +/- 0.01069 |
| 25 | 6.219 +/- 0.208 | 0.217 +/- 0.208 | 5.850 +/- 0.185 | 36.528 +/- 0.745 | -0.199 +/- 0.191 | 0.09793 +/- 0.00135 |
| 50 | 7.823 +/- 2.010 | 1.820 +/- 2.010 | 7.200 +/- 1.666 | 25.454 +/- 5.048 | -2.396 +/- 3.036 | 0.09134 +/- 0.01353 |
| 100 | 9.958 +/- 0.153 | 3.956 +/- 0.153 | 8.965 +/- 0.110 | 20.187 +/- 3.175 | -6.069 +/- 0.362 | 0.07910 +/- 0.01208 |

### `R-D`

| Pass | Aggressive target SDR (dB) | Teacher-residual SDR (dB) | Instrumental SDR (dB) | Low-vocal SDR (dB) | Vocal projection (dB) | Teacher spectrum L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.003 +/- 0.003 | -6.000 +/- 0.003 | 0.002 +/- 0.003 | 0.003 +/- 0.003 | -53.472 +/- 1.211 | 0.36321 +/- 0.00821 |
| 25 | 8.796 +/- 0.583 | 2.794 +/- 0.583 | 8.055 +/- 0.471 | 11.758 +/- 1.540 | -7.799 +/- 1.038 | 0.11207 +/- 0.00233 |
| 50 | 9.904 +/- 0.388 | 3.902 +/- 0.388 | 8.920 +/- 0.280 | 13.986 +/- 2.169 | -9.106 +/- 2.553 | 0.09810 +/- 0.00084 |
| 100 | 10.453 +/- 0.081 | 4.451 +/- 0.081 | 9.318 +/- 0.062 | 16.279 +/- 0.975 | -8.694 +/- 0.785 | 0.08399 +/- 0.00067 |

## Interpretation

At pass 100, `W-R` remains the strongest cell:

- `W-R`: 16.542 dB aggressive-target SDR and 13.687 dB conventional
  instrumental SDR;
- `W-D`: 13.362 dB and 11.449 dB respectively;
- `R-R`: 9.958 dB and 8.965 dB respectively;
- `R-D`: 10.453 dB and 9.318 dB respectively.

The practical findings are:

1. The pretrained neural representation is a major advantage. Both random
   cells remain far below `W-R` after 100 passes. `R-D` is only about 0.49 dB
   ahead of `R-R` on the aggressive target, so changing the output semantic
   does not compensate for losing the pretrained body.
2. A direct instrumental head can learn from the pretrained body, but `W-D`
   starts near an uninformative output because its final convolution is reset.
   It is still about 3.18 dB behind `W-R` on the aggressive target and about
   2.24 dB behind on conventional instrumental SDR at pass 100.
3. `W-R` reaches its best aggressive-target score around pass 25 and then
   drifts slightly. More passes do not produce a monotonic improvement.
4. The random cells show larger seed variation during early learning,
   especially `R-R` at pass 50. This is consistent with optimization
   instability and does not support selecting a random-init student for the
   next experiment.
5. The final gate fails: the three-seed `R-D` mean is 6.089 dB below `W-R` for
   both aggressive-target and teacher-residual SDR. The runner therefore did
   not render a 12-song listening set.

The matrix supports continuing from a pretrained TFC-TDF representation. It
does not show that direct instrumental output is intrinsically worse; the
direct cell also has to learn a newly reset output head. A useful follow-up,
if direct output is still desired, would be a longer or staged `W-D` run with
an explicitly documented head warm-up, but that is a new experiment and not a
reason to reinterpret this failed gate as a product result.

## Engineering and licensing disposition

All 12 cells completed without non-finite loss or OOM on the CUDA runner.
The report status is `completed`, but the quality gate is `passed: false`.
There was no Android LiteRT, ONNX, TFLite, listening, or Booming SS product
qualification in this run.

The experiment uses educational/non-commercial MUSDB18 material and a teacher
whose redistribution permission is not established. Derived checkpoints and
audio remain local research artifacts. They are not eligible for `bss-tflite`,
GitHub release assets, or Booming SS integration.

## Next step

Do not spend more compute on random initialization or immediately export a
candidate. Keep `W-R` as the current pretrained reference. If the goal remains
an aggressive compact instrumental model, the next controlled experiment
should focus on short vocal events: retain the pretrained body, rank training
windows by 50-200 ms Inst 3 residual activity and student/teacher error, and
replace a measured fraction of uniform updates with those hard windows. Keep
the external listening timestamps evaluation-only. Compare against the
four-song alpha-1.00 step-2048 and ten-song scale-10 step-8192 references.

