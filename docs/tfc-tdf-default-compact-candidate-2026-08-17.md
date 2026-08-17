# TFC-TDF default compact candidate

Status: provenance and executable contract frozen; export and validation pending

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
| Feature order | `left.real, left.imag, right.real, right.imag` |
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
flattened and transposed. Consequently each neural input frequency point is
ordered exactly as `left.real, left.imag, right.real, right.imag`.

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
