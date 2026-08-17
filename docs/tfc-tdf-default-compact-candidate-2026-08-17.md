# TFC-TDF default compact candidate

Status: host export, tensor parity, and full-song audio parity passed; device
execution and separation quality are not established

This experiment evaluates the public TFC-TDF default vocal checkpoint as an
extremely small two-stem candidate. The generated graph contains only the
spectrogram-to-spectrogram neural core. STFT, iSTFT, chunk assembly, and the
residual instrumental stem remain host DSP responsibilities.

## Frozen source identity

| Item | Frozen identity |
| --- | --- |
| Repository | `https://github.com/ws-choi/ISMIR2020_U_Nets_SVS` |
| Upstream commit | `aafcb69c43675713b86cd4f96eb659cf7eb55d16` |
| Checkpoint path | `etc/checkpoints/tfc_tdf_net/debug/vocals_epoch=891.ckpt` |
| Checkpoint bytes | `12002436` |
| Checkpoint SHA-256 | `101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d` |
| Upstream license | MIT, copyright (c) 2020 ws-choi |
| Training corpus | MUSDB18, Zenodo record `1117372`, DOI `10.5281/zenodo.1117372` |

The checkpoint is publicly committed in the upstream MIT-licensed repository;
no separate checkpoint license was found. This project preserves the upstream
MIT notice and does not claim rights in MUSDB18 audio. Zenodo records the
dataset license as `other-nc` and states that MUSDB18 is provided for
educational purposes only; commercial use requires express permission from
the relevant copyright holders. This experiment is therefore suitable for the
current free research build, but the checkpoint must receive a fresh legal and
provenance review before any commercial or store edition uses or redistributes
it.

## Frozen neural-core contract

The checkpoint is the README's pretrained TFC-TDF vocal model and reports
median MUSDB18 SDR `6.954695` for vocals and `13.561705` for accompaniment.
The export must strictly load every `spec2spec.*` tensor after removing that
prefix. `stft.*` tensors and Lightning optimizer/trainer state are not part of
the neural-core artifact.

| Field | Value |
| --- | --- |
| Target | `vocals` |
| Sample rate | `44100` Hz |
| STFT | `n_fft=2048`, `hop_length=1024`, periodic Hann, centered, reflect padding |
| Input/output | FP32 `[1,4,1025,128]` NCHW at the ONNX boundary |
| Feature order | `left.real, right.real, left.imag, right.imag` |
| TFLite boundary | FP32 `[1,1025,128,4]` NHWC |
| Estimation mode | direct complex mapping, not a mask |
| Stem reconstruction | vocals from model; instrumental is `mixture - vocals` |
| Network | 7-block dense U-Net, 24 internal channels, 5 TFC-TDF layers per block |
| Kernels | temporal `3`, frequency `3`; TDF bottleneck factor `16`, minimum `16` |
| Activations | ReLU internally, identity output |

The static neural input includes all 1025 real-FFT bins. Unlike the older MDX
contracts, no Nyquist bin is removed and no frequency truncation is applied.

## Frozen whole-track DSP

The upstream evaluation path defines these exact assembly rules:

- model input length: `hop_length * (num_frame - 1) = 130048` samples;
- trim length: `ceil(5000 / 1024) * 1024 = 5120` samples on each side;
- useful samples per chunk: `119808`;
- each chunk is zero-padded by 5120 samples on both sides;
- the final chunk receives additional right zero padding to 130048 samples;
- every reconstructed chunk is cropped to `[5120:-5120]`, concatenated in
  ascending offset order, and finally truncated to the source length.

The upstream complex packing is frequency/frame/complex/channel before being
flattened and transposed. The complex axis precedes the channel axis before
flattening, so each neural input frequency point is ordered exactly as
`left.real, right.real, left.imag, right.imag`.

## Required evidence

The candidate is not complete until one deterministic command records:

1. strict checkpoint loading and a state-tensor inventory;
2. a static FP32 ONNX artifact and PyTorch-versus-ONNX tensor metrics;
3. a static FP32 TFLite artifact and PyTorch/ONNX/TFLite tensor metrics;
4. a short real-audio fixture comparison across all three backends;
5. a full-song comparison, including vocals and residual instrumental stems;
6. artifact, script, environment, fixture, and output SHA-256 identities.

Generated checkpoints, ONNX/TFLite files, tensors, reports, and audio outputs
must remain under ignored `models/`, `data/`, or `outputs/` directories. Only
the reproducible scripts, dependency lock, contract, and summarized results
belong in Git.

## Reproducible tools

The Python 3.12 environment is pinned by
`requirements-tfc-tdf-export.txt`. The three stages are:

```text
python tools/export_tfc_tdf_default_candidate.py --checkpoint <checkpoint>
python tools/convert_tfc_tdf_default_tflite.py --checkpoint <checkpoint>
python tools/validate_tfc_tdf_default_audio.py \
  --checkpoint <checkpoint> --audio <stereo-audio>
```

The export tool verifies the checkpoint byte count and SHA-256 before using a
minimal pickle compatibility shim for its historical Lightning
`AttributeDict`. It rejects any hyperparameter, state key, or tensor shape that
does not match this exact checkpoint. The converter operates on an ONNX work
copy because onnx2tf may simplify its input in place. It appends an explicit
NHWC output adapter to that work copy, leaving the canonical NCHW ONNX source
unchanged.

## Frozen generated artifacts

The following artifacts were generated with Python 3.12.10, PyTorch
2.11.0+cpu, ONNX 1.20.1, ONNX Runtime 1.26.0, onnx2tf 2.6.6, TensorFlow
2.20.0, and ai-edge-litert 2.1.2:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `tfc_tdf_default_vocals_core_fp32.onnx` | 3,991,252 | `ad1c5dcdf59630a6b130fb76aa39561b79c4896d36c9cba1074065435d7acc07` |
| `tfc_tdf_default_vocals_core_fp32.tflite` | 3,989,856 | `0ee7bbc0bd5a1194745ebf4df1753ba6ef32a256cbb55fca8098a43912591f5e` |

The Lightning checkpoint has 353 state tensors. The export consumes all 352
`spec2spec.*` tensors with strict loading and excludes only
`stft.stft.window`. The neural core contains 990,164 FP32 state values
(3,960,656 bytes) and 56 scalar int64 BatchNorm counters. The ONNX graph has
192 nodes and no STFT/iSTFT operators.

The FP32 FlatBuffer has one subgraph, 150 operators, 373 tensors, and no custom
operators. Both public tensors are `[1,1025,128,4]` NHWC FP32. The observed
operator inventory is 40 `CONV_2D`, 31 `CONCATENATION`, 24 `ADD`, 16
`TRANSPOSE`, 14 `BATCH_MATMUL`, 14 `MUL`, 7 `RELU`, 3 `TRANSPOSE_CONV`, and 1
`PAD`.

Generated reports and fixtures are local at:

```text
models/tfc-tdf/default-compact/onnx-export-report.json
models/tfc-tdf/default-compact/tflite-conversion-report.json
outputs/tfc-tdf-default-validation/audio-validation-report.json
```

## Tensor parity

Three deterministic random tensors passed a 95 dB / `1e-5` gate:

| Comparison | Minimum SNR | Maximum absolute error |
| --- | ---: | ---: |
| PyTorch vs ONNX | 101.233 dB | `2.801419e-6` |
| PyTorch vs TFLite | 100.983 dB | `3.099442e-6` |
| ONNX vs TFLite | 101.602 dB | `3.308058e-6` |

The real-audio DSP oracle independently compared the NumPy host path with
PyTorch. STFT parity was 135.921 dB with `1.525879e-5` maximum error; iSTFT
parity was 136.024 dB with `2.384186e-7` maximum error. The NumPy and PyTorch
round trips measured 139.298 dB and 137.477 dB respectively. This also
executable-proves the less obvious feature order
`left.real, right.real, left.imag, right.imag`.

## Audio parity

The source was a local stereo 44.1 kHz, 261.013-second listening sample. It is
not part of Git. Its identities are:

```text
source bytes:              35,351,663
source SHA-256:            8ff6b5f84deade9ddc9060ff3702dca80cb3aa304bb06eb99e479fa2d92a424b
decoded samples:           11,510,688
decoded FP32 PCM SHA-256:  3fbefbfd19fec935b58697588f98dbc6cc8271dc09e13f74b691332095aa14be
```

The first 30 seconds used 12 windows; the full song used 97. All pairwise
vocals and residual-instrumental comparisons passed the 90 dB / `1e-4` audio
gate. The most relevant PyTorch-versus-TFLite results are:

| Fixture | Stem | SNR | Maximum absolute error |
| --- | --- | ---: | ---: |
| 30 seconds | vocals | 128.108 dB | `3.278256e-7` |
| 30 seconds | instrumental | 137.171 dB | `3.576279e-7` |
| full song | vocals | 127.492 dB | `7.450581e-7` |
| full song | instrumental | 135.406 dB | `7.450581e-7` |

For every backend, `vocals + instrumental` reconstructed the input within
`5.960465e-8`. The ignored output directory contains PCM16 FLAC renders for
all three backends; the report binds both each FLAC and its pre-encoding FP32
audio.

Host inference time for the full song was 41.84 seconds for PyTorch, 36.84
seconds for ONNX Runtime, and 123.32 seconds for LiteRT, plus 5.05 seconds of
shared/reconstruction DSP. These desktop observations only establish that the
small graph executes; they are not Android latency, power, or delegate
evidence.

## Decision boundary

The compact candidate is admitted to an Android CPU/GPU feasibility and
listening experiment. It is not yet a product model. Host parity does not
establish vocal-removal quality, device real-time behavior, LiteRT 2.2.0
compatibility, GPU delegation, or thermal behavior. Redistribution and any
commercial/store use also remain blocked on a fresh review of the MUSDB18
training-data restriction.

## Listening assessment

The local full-song renders were reviewed before device benchmarking. The
instrumental sounded slightly cleaner than Flamingo's causal-model output when
that player was adjusted for complete vocal suppression. It remained much
worse than Booming SS's current default MDX 9662 model, with clearly audible
vocal residue. The result is nevertheless suitable for an experimental,
lightweight model intended for casual everyday sing-along use; it is not a
replacement for the default quality route.

No audible difference was found among the PyTorch, ONNX, and TFLite renders.
This listening observation is consistent with the measured conversion parity,
but it remains one listener's assessment on the local sample rather than an
objective separation-quality score.
