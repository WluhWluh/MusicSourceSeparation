# TFC-TDF 24-Frame Host Experiment

Status: complete for static ONNX/TFLite export and one full-song host
evaluation. This is an experiment only; it does not change the 128-frame
product candidate or establish Android real-time performance.

## Objective

Evaluate whether the existing public TFC-TDF checkpoint can be exported with
a shorter static time dimension, then choose a less wasteful overlap-save
contract before any Android integration work.

The checkpoint remains unchanged. Only the exported neural-core shape and the
host window assembly change.

## Frozen identities

| Item | Identity |
| --- | --- |
| Checkpoint | `models/tfc-tdf/source/vocals_epoch=891.ckpt` |
| Checkpoint SHA-256 | `101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d` |
| Candidate shape | `[1, 4, 1025, 24]` NCHW / `[1, 1025, 24, 4]` NHWC |
| Candidate ONNX | `tfc_tdf_default_vocals_core_fp32_f24.onnx` |
| Candidate TFLite | `tfc_tdf_default_vocals_core_fp32_f24.tflite` |
| Candidate ONNX SHA-256 | `ab71803120d709c3498635dcada1aeed925dc113531bd764a3552615e24bb7b1` |
| Candidate TFLite SHA-256 | `f4b9178347662e6700b67abd6ff0d58fed5ca36ed55d3b0294eae8ce47a09f4a` |
| Sample rate | 44,100 Hz |
| FFT / hop | 2,048 / 1,024 |

The export reuses all checkpoint weights. The PyTorch graph is temporally
shape-compatible for 24 frames because the three temporal downsamplings each
receive an even dimension. This is shape compatibility, not evidence that the
checkpoint was trained for short-context inference.

## Export validation

The reproducible commands are:

```text
python tools/export_tfc_tdf_default_candidate.py \
  --checkpoint models/tfc-tdf/source/vocals_epoch=891.ckpt \
  --num-frames 24 \
  --output-dir models/tfc-tdf/compact-24

python tools/convert_tfc_tdf_default_tflite.py \
  --checkpoint models/tfc-tdf/source/vocals_epoch=891.ckpt \
  --onnx models/tfc-tdf/compact-24/tfc_tdf_default_vocals_core_fp32_f24.onnx \
  --num-frames 24 \
  --output-dir models/tfc-tdf/compact-24
```

The export passed ONNX checking and PyTorch/ONNX parity. The TFLite conversion
passed the same three-way tensor gate used by the 128-frame candidate:

| Comparison | Minimum SNR | Maximum absolute error |
| --- | ---: | ---: |
| PyTorch vs ONNX | 103.10 dB | `1.559e-6` |
| PyTorch vs TFLite | 102.58 dB | `1.818e-6` |
| ONNX vs TFLite | 103.22 dB | `1.371e-6` |

## Window contracts tested

The 24-frame input contains 23 hops, or 23,552 samples (about 534 ms). Each
candidate uses equal left and right context and emits the center region:

| Context per side | Output stride | Output efficiency | Full-song windows |
| ---: | ---: | ---: | ---: |
| 1 hop | 21 hops / 488 ms | 91.3% | 536 |
| 2 hops | 19 hops / 441 ms | 82.6% | 592 |
| 3 hops | 17 hops / 395 ms | 73.9% | 662 |
| 4 hops | 15 hops / 348 ms | 65.2% | 750 |
| 5 hops | 13 hops / 302 ms | 56.5% | 865 |

Unlike the old isolated fixture path, the short-window evaluator supplies
continuous audio context across internal boundaries and zero pads only at the
song edges. The candidate implementation is in
`tools/tfc_tdf_short_window.py`.

## Full-song host evaluation

Input:

```text
C:/Users/User/Documents/MusicPlayer/BoomingMusic/.artifacts/listening/
  夜に駆ける-YOASOBI-83759543.flac
```

The 128-frame isolated render is used as a same-checkpoint reference, not as
ground truth. The 24-frame candidate is rendered with ONNX Runtime CPU for
each trim choice. The evaluator records vocals/instrumental waveform
differences, boundary derivative ratios, output efficiency, and a vocal
projection proxy. There is no isolated vocal reference for this song, so the
proxy is diagnostic only and is not SDR, SI-SDR, or a separation-quality
claim.

The most useful aggregate results were:

| Context | Vocals SNR vs 128-frame | Instrumental SNR vs 128-frame | Seam p95 ratio | Runtime | Output efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 hop | 13.40 dB | 21.47 dB | 3.14 | 32.61 s | 91.3% |
| 2 hops | 13.70 dB | 21.78 dB | 2.43 | 36.38 s | 82.6% |
| 3 hops | 13.89 dB | 21.97 dB | 2.67 | 41.95 s | 73.9% |
| 4 hops | 14.19 dB | 22.27 dB | 1.93 | 46.39 s | 65.2% |
| 5 hops | 14.21 dB | 22.29 dB | 2.06 | 54.28 s | 56.5% |

The 4-hop choice is the current experiment candidate: it removes 115 model
windows relative to 5-hop on this song, while retaining nearly the same
reference similarity and showing a slightly lower seam p95. The difference
between 4 and 5 hops is small enough that it must be confirmed with more
musical material and listening before adoption.

The vocal-residual proxy produced these instrumental projections onto the
128-frame vocal estimate:

| Context | Candidate instrumental projection | Delta from 128-frame baseline | Candidate vocal projection gain |
| ---: | ---: | ---: | ---: |
| 1 hop | -20.71 dB | +1.42 dB | 0.960 |
| 2 hops | -20.93 dB | +1.19 dB | 0.970 |
| 3 hops | -21.01 dB | +1.11 dB | 0.970 |
| 4 hops | -21.16 dB | +0.96 dB | 0.980 |
| 5 hops | -21.11 dB | +1.01 dB | 0.980 |

Lower projection is preferable in this diagnostic, and 4-hop is again the
best of the tested short contracts. These values only measure similarity to
the same checkpoint's 128-frame estimate; they do not measure true vocal
leakage.

The largest individual seam ratios are concentrated at musical transients or
low-level passages, so a raw derivative ratio is not by itself proof of an
audible click. The report retains the absolute boundary delta and local
derivative baseline for later listening correlation.

## Reproduction

```text
python tools/evaluate_tfc_tdf_short_window.py \
  --audio <stereo-audio> \
  --candidate-onnx models/tfc-tdf/compact-24/tfc_tdf_default_vocals_core_fp32_f24.onnx \
  --reference-onnx models/tfc-tdf/default-compact/tfc_tdf_default_vocals_core_fp32.onnx \
  --output-dir outputs/tfc-tdf-24-frame-host-evaluation-full \
  --trim-hops 1 2 3 4 5
```

The ignored report and FLAC renders are under
`outputs/tfc-tdf-24-frame-host-evaluation-full/`.

## Limitations and next gate

- The checkpoint was trained and originally evaluated at 128 frames; short
  shape export needs a dedicated quality review and likely fine-tuning.
- No isolated ground-truth stems are available for the listening sample.
- Host ONNX execution does not predict LiteRT CPU/GPU timing or delegate
  behavior at 24 frames.
- The 4-hop contract is not yet a Booming SS cache or streaming contract.
- The next gate is S25 CPU/GPU tensor and sustained-window benchmarking for
  24 frames, followed by a second full-song fixture before any product/UI
  exposure.
