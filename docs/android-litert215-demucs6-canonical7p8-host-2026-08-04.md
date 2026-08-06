# HTDemucs 6-stem canonical 7.8-second host freeze

Date: 2026-08-04

Status: complete host export and parity; S25 follow-up is recorded separately

This record freezes the first project-owned FP32 neural-core artifact generated
from the official HTDemucs-6s safetensors checkpoint. It is a prerequisite for
later device work, not a device result and not a product-support decision.

## Result

The canonical candidate is accepted for a later device experiment after a
complete Torch/LiteRT host pipeline check. The raw frequency latent has a
separate warning: its first OLA window does not satisfy the old uniform
`maxAbs <= 1e-3` rule, even though its SNR is 94.6801 dB. The reconstructed
waveform, branch combination, and two-window OLA all pass the layered host gate.

```text
modelId:    htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0
candidate:  htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0@host-1
profile:    canonical_7p8s
scope:      host-only-not-a-device-result
sampleRate: 44100 Hz
samples:    343980 (39/5 seconds)
stems:      drums, bass, other, vocals, guitar, piano
```

The authoritative host contract is
`models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/candidate-manifest.json`.
The generated files remain in ignored local model storage and are not copied
into the Android APK.

## Frozen identities

| Item | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| LiteRT FlatBuffer `htdemucs_6s.core.canonical_7p8s.fp32.tflite` | 117,624,880 | `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7` |
| `export-report.json` | 69,484 | `05b67b091a11b2105d07a2c8f608f2825fa35a1fa0d4cfd95bd54deb6cfebb88` |
| `flatbuffer-inspection.json` | 3,844 | `6b4763cfe62844983dd3d183a53c9ecad6ee5865f2cf1f795363e7b698b0475e` |
| `candidate-manifest.json` | 47,505 | `e22708ecbb1e43f528a3f1ff2ab33a8062c42fc36ffed1134f837426865f33e2` |
| exporter `tools/export_htdemucs_litert_candidate.py` | 55,693 | `532f5e1b7c9c30aa53963c891c94e3379f5aa18d0297d6ae21ce1e4d82e1b9ba` |
| requirements lock `requirements-demucs-litert-export.txt` | 334 | `cb6ef5b77262c7d8594735413ecef5e841f3be1a710a48602b0620d229bf1706` |

The exporter row records its historical execution path. The exact source is at
commit `ee68956` and under
`tools/frozen-exporters/532f5e1b7c9c30aa53963c891c94e3379f5aa18d0297d6ae21ce1e4d82e1b9ba/`;
the current exporter at `tools/export_htdemucs_litert_candidate.py` is a later
revision. The freezer resolves the archived snapshot by its declared SHA while
preserving the immutable report's original path.

The canonical run deliberately does not persist an ONNX diagnostic. The
LiteRT conversion input is the project-owned PyTorch neural-core module.

## Source and reproducibility

| Source | Identity |
| --- | --- |
| official weight | `models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors`, 54,885,744 bytes, SHA-256 `d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411` |
| official metadata | `5c90dfd2.json`, 10,398 bytes, SHA-256 `72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e` |
| bag manifest | `htdemucs_6s.yaml`, 21 bytes, SHA-256 `207405151270af8fd81c2373c25d27950916682ac91dca7884a11ce13dad6f58` |
| Demucs loader | revision `eeac1d15891af95b1288d2884b95baa3e5baa96c` |
| neural-core boundary reference | `JBNU-CILAB/demucs-lite`, revision `9a2a17c7a81843c2ae49674986f9e1e8b5f6915f`, reference-only |

The source checkouts were dirty only in line-ending metadata; the exporter
records this as `workingTreeState=line-endings-only`. The public neural-core
reference is not treated as the producer of this FlatBuffer, and no third-party
node IDs were copied.

The frozen environment is:

```text
Python 3.12.3
torch 2.11.0+cpu
NumPy 2.5.1
safetensors 0.8.0
litert-torch 0.9.1
ai-edge-litert 2.1.5
```

Canonical conversion flags:

```text
strictExport=true
lightweightConversion=false
runtimeConstantFolding=false
enableX64=false
deterministicPositionalEmbedding=true
```

The exporter was run with the explicit `canonical_7p8s` profile. It validates
the checkpoint's original segment as `39/5` rather than silently changing the
model segment to match an arbitrary sample count. Two independent conversions
from a clean output directory produced the same 117,624,880-byte artifact and
the same SHA-256, `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7`.

## FlatBuffer contract

Inspection reports TFL3/schema 3, one subgraph, 3,432 operators, 4,176 tensors,
4,179 buffers, 26 operator codes, and zero custom operators. The signature is
`serving_default`:

| Binding | Tensor index | Name | Dtype and shape |
| --- | ---: | --- | --- |
| `args_0` | 0 | `serving_default_args_0` | float32 `[1,2,343980]` |
| `args_1` | 1 | `serving_default_args_1` | float32 `[1,4,2048,336]` |
| `output_0` | 4170 | `serving_default_output_0_output` | float32 `[1,6,4,2048,336]` |
| `output_1` | 4175 | `serving_default_output_1_output` | float32 `[1,6,2,343980]` |

Axes are, respectively, `batch/channel/sample`,
`batch/feature/frequency/frame`, `batch/stem/feature/frequency/frame`, and
`batch/stem/channel/sample`. Feature order is `L.real, L.imag, R.real, R.imag`.
The output stem order is `drums, bass, other, vocals, guitar, piano`.

The FlatBuffer metadata contains `min_runtime_version=2.16.0` and
`keep_stablehlo_constant=true`. These are converter metadata values; they do
not establish that any Android runtime or delegate can execute this graph.
The graph still contains 132 `GATHER_ND`, 940 `RESHAPE`, 688 `MUL`, 414 `ADD`,
88 `CONV_2D`, 44 `FULLY_CONNECTED`, and 5 `BROADCAST_TO` operators. These
counts are useful for planning the next backend probe and must not be merged
with counts from the 2-second artifact.

## Host DSP contract

The neural core accepts both waveform and host-generated spectrum inputs. The
host preprocessing and reconstruction are part of the frozen contract:

```text
STFT:
  n_fft=4096, hop_length=1024, periodic Hann, normalized=true, center=true
  pad_mode=reflect, outer pad left/right=(1536,1620)
  drop the Nyquist bin, crop two frames from both ends

iSTFT:
  restore a zero Nyquist row and two frame pads on both ends
  normalized=true, center=true, reconstruction length=347136
  crop [1536,345516)

branch combination: frequency iSTFT + time waveform
global normalization: stereo-channel mean; unbiased std (correction=1)
  with epsilon=1e-8 before OLA, then inverse normalization after OLA

OLA:
  window=343980, stride=257985, overlap=85995 (0.25)
  track=515970, shifts=0, transition_power=1
  ascending-offset FP32 accumulation, EOF weight=triangle-prefix
```

The first OLA window uses source `[0,343980)`. The EOF window has offset
`257985`, copies source `[214988,515970)`, pads 42,998 samples on the right,
and center-crops 42,997 samples on the left and 42,998 on the right. The Torch
implementation's padded chunks are bitwise equal to the official Demucs
`apply_model` split/overlap-add oracle, including this EOF rule.

## Host parity

The single-window fixture uses seed `20260803`. It freezes waveform input,
spectrum input, raw frequency output, raw time output, frequency-branch iSTFT,
and combined six-stem output. LiteRT versus Torch aggregate metrics are:

| Layer | SNR (dB) | Max absolute error | RMS error |
| --- | ---: | ---: | ---: |
| raw frequency output | 99.7049 | `1.158714e-04` | `2.969984e-07` |
| frequency iSTFT waveform | 98.0434 | `5.152076e-06` | `4.732068e-07` |
| raw time waveform | 124.6445 | `5.736947e-07` | `1.884214e-08` |
| combined waveform | 100.7594 | `5.237758e-06` | `4.742397e-07` |

The two-window fixture has seed `20260804`. Torch core OLA versus official
`apply_model` is bitwise equal. LiteRT versus Torch OLA is:

| Region | SNR (dB) | Max absolute error |
| --- | ---: | ---: |
| complete 515,970-sample track | 98.4269 | `5.550683e-06` |
| overlap `[257985,343980)` | 105.4076 | `1.929700e-06` |
| EOF `[429975,515970)` | 117.7663 | `8.195639e-07` |

The lowest full-track per-stem SNR is 85.3562 dB. In the low-energy EOF
region, bass, vocals, piano, and other tracks can have SNR below 80 dB despite
very small absolute errors; the report therefore applies the `1e-6` absolute
gate when reference RMS is at or below `1e-3`.

Host timing is informational only and was not collected as a device benchmark:

| Operation | Wall time (seconds) |
| --- | ---: |
| Torch reference and single-window core | 3.7778 |
| LiteRT single-window core | 1.6438 |
| official Torch OLA | 2.6413 |
| project Torch OLA | 2.6444 |
| LiteRT OLA | 3.8301 |

## Gate and scope

The report records the following independent decisions:

```text
uniformTensorGatePassed: false
hostPipelineGatePassed:  true
acceptedForDeviceTesting: true
```

`uniformTensorGatePassed=false` is intentional. The first OLA window's raw
frequency latent has SNR 94.6801 dB but max absolute error 0.002628326, so it
does not satisfy the obsolete uniform `1e-3` tensor rule. The downstream
frequency iSTFT, time branch, combined waveform, and OLA checks pass the
80 dB plus `1e-3` policy; low-energy waveform references use the separately
declared `1e-6` absolute rule. The warning must remain visible in future
reports rather than being hidden by the aggregate result.

`acceptedForDeviceTesting` only authorizes a subsequent experiment. This host
run contains no Android allocation, delegate coverage, latency, memory, thermal,
CPU, GPU, or NPU evidence. In particular, LiteRT 2.1.5 is the conversion
environment, not a claim that S25 can execute the 7.8-second graph.

## Fixtures

All files are little-endian float32 and are bound by shape, role, byte size, and
SHA-256 in the manifest:

| Role / file | Shape | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `waveform_input.f32le.raw` | `[1,2,343980]` | 2,751,840 | `9515d42e72e96036e34f0193d1302e8cccbf80c4e9e7084d1fcba26e17fea40e` |
| `spectrum_input.f32le.raw` | `[1,4,2048,336]` | 11,010,048 | `9d01b6cf6943c61724f6bb7edefc8f6156f9749f1ae9bc4f84508a62738ee5e0` |
| `frequency_golden.f32le.raw` | `[1,6,4,2048,336]` | 66,060,288 | `86e916da6041bf0038adcd770ed986ddbe54e52e264d934f11d8186b7616dcbb` |
| `waveform_golden.f32le.raw` | `[1,6,2,343980]` | 16,511,040 | `be0934e5be603d5fd17a69cf92d3786e21053128cdfb4fa0f4f78166aebea474` |
| `frequency_waveform_golden.f32le.raw` | `[1,6,2,343980]` | 16,511,040 | `5bb82c18f2f778b6cf66c28c663d76921f37c4bb69128351e3ce0b4229498ba8` |
| `combined_golden.f32le.raw` | `[1,6,2,343980]` | 16,511,040 | `7d5fd3585cbdbb46551e9e05883da9aa880702a72efaa71402072957d037264a` |
| `ola_mix_input.f32le.raw` | `[1,2,515970]` | 4,127,760 | `dc28161d585169a4ce4661a5e142fe2668ed76e32f6b89a71d24c24522ee1790` |
| `ola_combined_golden.f32le.raw` | `[1,6,2,515970]` | 24,766,560 | `65f9cba1bb9b16213f5e3cc0dda5ec919de640a65088eadb69f0d079ec79d48a` |

## Device follow-up

The identity-indexed Android registry, digest-pinned host-manifest adapter, full
host DSP contract parser, and bounded S25 probes have now been implemented.
CPU and explicit GPU+CPU FP32 device results, including the failed numerical
gates and the 30/180-second CPU E2E experiments, are recorded in
[`android-litert215-demucs6-canonical7p8-s25-2026-08-04.md`](android-litert215-demucs6-canonical7p8-s25-2026-08-04.md).
This does not change the host-only scope of the evidence in this file, and no
QNN claim is implied by this host candidate.

For a byte-identical rerun, check out commit `ee68956`; the reproduction entry
point at that revision is:

```text
tools/export_htdemucs_litert_candidate.py --profile canonical_7p8s \
  --demucs-root .tmp/demucs-adefossez \
  --demucs-lite-root .tmp/demucs-lite-reference \
  --weights models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors \
  --metadata models/demucs/official-hf/htdemucs_6s/5c90dfd2.json \
  --samples 343980 --seed 20260803 \
  --output models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/htdemucs_6s.core.canonical_7p8s.fp32.tflite \
  --fixtures models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/fixtures \
  --report models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/export-report.json
```
