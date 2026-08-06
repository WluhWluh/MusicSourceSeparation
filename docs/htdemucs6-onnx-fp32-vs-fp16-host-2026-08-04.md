# HTDemucs-6s FP32 versus FP16-storage ONNX host validation

Date: 2026-08-04

Branch: `experiment/model-matrix-demucs-multistem`

Source revision: `3510d5c263ab8236cd96cfeb07feef1c1a5c0dff` (dirty worktree)

## Result

The published FP32 ONNX materially reduces the error observed with the smaller
`fp16weights` ONNX, but the improvement is not caused by retaining more precise
learned weights. The official checkpoint payload is already FP16, and every ONNX
initializer value is identical after widening the FP16 artifact back to FP32.

The remaining FP32 residual is now localized. The complete patched Torch wrapper
is `120.5086 dB` from original Torch, while the published FP32 ONNX is only
`73.9913 dB` from that patched wrapper. Two large FP32 reductions in ORT CPU explain
the material gap:

1. The frequency input computes an unbiased standard deviation over
   `4 * 2048 * 336 = 2,752,512` values. An equivalent FP32 `ReduceSum` control is
   unchanged at `73.99 dB`; FP64 accumulation raises final parity to `81.69 dB`.
2. Ten transformer `MyGroupNorm(num_groups=1)` modules are exported through
   `InstanceNormalization` over `1,032,192` frequency or `516,096` time values.
   An explicit FP32 control is unchanged; FP64 statistics raise final parity to
   `92.58 dB`.

The FP64 graphs are diagnostic host artifacts. They are not LiteRT candidates and
do not establish a mobile CPU/GPU implementation route.

## Artifact identity

| Variant | Bytes | SHA-256 | Initializers | Nodes | Cast nodes |
|---|---:|---|---:|---:|---:|
| FP32 ONNX | 258,159,781 | `48f8e84945579f8ab340e083339e9221e03785dbe733a52c388200b6d3ca779a` | 541 FLOAT | 24,671 | 752 |
| FP16-storage ONNX | 136,428,532 | `7ce55792e2231c93fbf92de95f5fd5b3a5e6c89f7db690dfd693e8f1dce56869` | 397 FLOAT + 144 FLOAT16 | 24,819 | 900 |

Both files are pinned to `StemSplitio/htdemucs-6s-onnx` revision
`49df9b6989cf2150840ea65b0bef77a2e471b678`, ONNX opset 17, and the same
`mix [1,2,343980] -> stems [1,6,2,343980]` ABI.

Their complete non-Cast operator histograms are identical. The 144 renamed FP16
initializers contain 27,304,704 values; together with the 397 unchanged
initializers, all 27,414,996 values compare exactly after widening to FP32.
Maximum initializer difference is `0.0`.

## Checkpoint identity

The legacy exporter checkpoint and official safetensors reference were loaded with
the pinned Demucs checkout. The legacy pickle used `weights_only=True` with only
`HTDemucs` and `Fraction` allowlisted.

| Check | Result |
|---|---:|
| Serialized state tensors | 525 / 525 equal |
| Serialized values | 27,414,996 / 27,414,996 equal |
| Serialized dtype | FP16 for all 525 tensors |
| Instantiated FP32 model tensors | 525 / 525 equal |
| Maximum value difference | 0.0 |
| Model kwargs | all 62 equal |
| Training args and metrics | equal |

The source order is `drums,bass,other,vocals,guitar,piano`, and both configs use
the exact `39/5` second segment. Checkpoint format or source cannot explain the
ONNX-versus-Torch residual.

## Numerical comparison

All tests use ORT CPU with four threads. The frozen-window comparison is float32
before OLA or PCM quantization. The audio comparisons use identical global
normalization, centered EOF padding/crop, 25 percent overlap, triangle-prefix OLA,
and the same PCM16 writer as the completed batch experiment.

| Scope | FP32 vs Torch SNR | FP16 vs Torch SNR | FP32 gain | FP32 max error | FP16 max error |
|---|---:|---:|---:|---:|---:|
| Frozen 7.8 s float fixture | 73.9721 dB | 66.7188 dB | +7.2532 dB | 1.10559e-4 | 2.95237e-4 |
| Canonical 30 s, PCM16 | 78.0766 dB | 50.7398 dB | +27.3368 dB | 6 LSB | 254 LSB |
| John Lennon - Imagine, 185.208 s, PCM16 | 76.8669 dB | 68.2475 dB | +8.6194 dB | 15 LSB | 58 LSB |

On the frozen float fixture, FP32 reduces RMSE from `2.3879639e-5` to
`1.0360162e-5`: an RMSE ratio of `0.433849` and an MSE reduction of `81.1775%`.
FP16 versus FP32 itself measures `70.7184 dB` with maximum error `2.19315e-4`.

The low-signal per-stem SNR values remain content dependent. A small integer error
can produce a low SNR when a stem is nearly silent, so aggregate SNR does not imply
a uniform per-stem gate pass.

## Pre-export ablation

The pinned StemSplit exporter revision was applied to the official safetensors
model under Torch `2.11.0+cpu`. The waveform wrapper was compared before ONNX
serialization.

| Comparison | Aggregate SNR | Maximum error |
|---|---:|---:|
| All exporter patches vs original Torch | 120.5086 dB | 5.96046e-7 |
| FP32 ONNX vs all-patched Torch | 73.9913 dB | 1.10373e-4 |
| Real STFT vs original Torch STFT | 118.9050 dB | 1.00136e-5 |
| Real iSTFT vs original Torch iSTFT | 125.8844 dB | 4.17233e-7 |

The segment coercion, deterministic position embedding, manual MHA, and real
STFT/iSTFT patches therefore do not create the 73-78 dB discrepancy. The error is
introduced during ONNX execution.

## Reduction localization

The localization fixture uses WSL `Ubuntu-LiteRT`, Python `3.12.3`, Torch
`2.11.0+cpu`, ONNX `1.17.0`, ORT `1.21.1`, sequential CPU execution, and four
threads. Graph optimization disabled/basic/extended/all produced the same causal
boundaries.

| Variant | Frequency variance | Normalized frequency | Transformer frequency output | Final stems | Final max error |
|---|---:|---:|---:|---:|---:|
| Published FP32 ONNX | 57.8097 dB | 63.8240 dB | 70.8043 dB | 73.9865 dB | 1.10343e-4 |
| Corrected FP32 `ReduceSum` | 57.8097 dB | 63.8240 dB | 70.8043 dB | 73.9865 dB | 1.10343e-4 |
| Frequency FP64 accumulation | 136.6310 dB | 118.8314 dB | 73.4935 dB | 81.6905 dB | 4.49866e-5 |
| Frequency FP64 + explicit GroupNorm FP32 | 136.6310 dB | 118.8314 dB | 73.4945 dB | 81.6883 dB | 4.50164e-5 |
| Frequency FP64 + GroupNorm FP64 statistics | 136.6310 dB | 118.8314 dB | 91.5036 dB | 92.5845 dB | 1.40518e-5 |

The frequency tensor is `[1,4,2048,336]`. The correct reduction count is
`2,752,512`, with unbiased denominator `2,752,511`. The earlier exploratory count
`1,376,256` was invalid; its manifest has been removed. Correcting the count while
retaining FP32 accumulation does not change any measured boundary, which rules out
the Bessel correction as the source.

The first transformer layer was also extracted as an isolated ONNX subgraph and
fed its exact patched-Torch input. LayerNorm, Q/K/V projections, attention logits,
Softmax, and feed-forward operations remain between `117.07` and `132.65 dB`.
The feed-forward residual is `128.13 dB`, then the exported group normalization
drops immediately to `91.16 dB`; the layer output is `91.62 dB`. This isolates the
second loss to the large `InstanceNormalization` statistics reduction rather than
attention or feed-forward kernels.

The final combined artifact remains stable across optimization levels:
`92.43 dB` disabled, `92.45 dB` basic, `92.50 dB` extended, and `92.58 dB` all.

## Remaining floor

After both diagnostic rewrites, the first remaining low-SNR boundary is the time
encoder: normalized input is `106.76 dB`, `tencoder.0` is `72.97 dB`, and
`tencoder.3` is `80.05 dB`. The sinusoidal position embedding itself is
`137.69 dB`, so it is not the source. Cross-attention carries the time-branch
residual into the frequency branch (`119.15 dB` after frequency layer 0,
`98.59 dB` after cross layer 1, and `91.50 dB` after layer 4). The frequency iSTFT
then measures `91.03 dB`; combined stems finish at `92.58 dB`.

These remaining boundaries are reported, not assigned to a specific kernel. An
exact-input time-encoder subgraph experiment would be required to separate ORT
kernel error from amplification of its `106.76 dB` input perturbation. This is no
longer necessary to explain the original 73-78 dB failure or to pass the 80 dB
aggregate host gate.

## Runtime observation

The full 185.208 second run used ORT 1.26.0 for both variants. FP32 separation RTF
was `0.3878`, versus `0.4087` for FP16; peak observed RSS was respectively
4,885,225,472 and 4,885,905,408 bytes. These were not interleaved benchmark trials,
so the timing difference is observational. The smaller artifact did not reduce the
observed runtime memory footprint.

## Evidence

- Float fixture report:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/fixture/report.json`
- Checkpoint equivalence report:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/checkpoint-equivalence.json`
- 30 second comparison:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/validation-30s/comparison.json`
- Full-track comparison:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/tracks/john-lennon-imagine/comparison.json`
- Downloaded FP32 artifact:
  `models/demucs/onnx/htdemucs_6s.onnx`
- Patched-Torch ablation:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/error-localization-torch211.json`
- Original intermediate probes:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/onnx-intermediate-probes/report.json`
- Isolated transformer layer-zero probes:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/onnx-transformer-layer0-isolated/report.json`
- FP32 GroupNorm control:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/onnx-transformer-groupnorm-fp32-probes/report.json`
- Final FP64 diagnostic boundary probes:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/onnx-final-boundary-probes/report.json`
- Final diagnostic manifest and model SHA:
  `outputs/htdemucs6-onnx-fp32-vs-fp16-host-20260804/rewrite/manifest-frequency-fp64-groupnorm-fp64.json`,
  `c18a8b927ba105da032769697986f57615e06e69f21e99dd89c603cb340db766`
