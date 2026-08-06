# Model Contracts

This document records model-specific assumptions that the Android implementation must eventually reproduce. It is intentionally separate from the general progress log because tensor contracts and DSP parameters need to stay precise.

## Executable MDX benchmark contracts (schema v2)

The executable Android benchmark contract is currently schema v2. It is
strictly scoped to one-input/one-output MDX spectrogram graphs, NHWC float32
tensors, and target-plus-residual stem reconstruction. Canonical sidecars live
under `app/src/main/assets/benchmark-contracts/` and are parsed by
`BenchmarkModelContractLoader`.

Do not generalize v2 by making its MDX DSP fields nullable. The Android runner,
host scripts, reference generation, and QNN audio path all rely on those fields
and intentionally reject other graph families.

## Tensor-only multi-stem candidates (schema v3)

Schema v3 is a candidate identity and tensor-shape contract, not an executable
LiteRT runtime contract. It is parsed by `TensorOnlyCandidateContractLoader`
and requires `contractKind: tensor-only-candidate`.

It supports:

- static float32 tensors with explicit index, name, shape, and semantic axes;
- multiple inputs and outputs for a future external-DSP neural core;
- packed 4-stem/6-stem outputs or one whole output tensor per stem;
- neural-core time and spectral output branches that bind the same ordered
  logical stem set without duplicating the public stem list;
- exact sample rate and rational segment length;
- maintainer model artifacts, the exact exporter input, and the ONNX conversion
  source as separate pinned identities;
- conversion-source license/model-card files and exact mixed storage dtype
  inventories (float32 I/O, no quantization, runtime precision not established);
- ordered direct stem semantics, including guitar and piano.

The current v3 sidecars describe 7.8-second `htdemucs` and `htdemucs_6s`
community ONNX conversion sources. They cannot be passed to the Android v2
runner. The later project-owned canonical artifacts use separately hashed host
manifests and an identity-indexed Android experimental registry; they do not
turn these community-ONNX sidecars into executable contracts. Any future
portable executable tensor-only schema must remain a separate type that binds
its source candidate, conversion recipe, FlatBuffer, and actual signature.

See `android-litert-demucs-multistem-feasibility-2026-08-03.md` for the source
freeze, conversion blockers, historical batch plan, and measured closure.

## External LiteRT runtime candidates (schema v4)

Schema v4 freezes an externally published LiteRT artifact and its observed
runtime plan without promoting it into the executable Android benchmark path.
It is parsed by `ExternalLiteRtCandidateContractLoader` and requires
`contractKind: external-litert-runtime-candidate`.

The initial sidecar,
`bandbuddy_htdemucs_6s_core_v1_0_0.json`, records the fixed BandBuddy neural
core as third-party evidence. It references the schema v3 six-stem candidate
context, but explicitly does not claim official-safetensors parity or a
complete public conversion recipe. It freezes:

- the ModelScope artifact URL, byte size, SHA-256, and exact TFL3 ABI;
- two static float32 inputs, two static float32 outputs, and the serving
  signature tensor indices;
- the official six-stem semantic order and complex-feature order;
- the declared TensorFlow Lite coordinate separately from its resolved LiteRT
  relocation artifact;
- the QNN runtime/delegate AAR identities and exact 11-node, seven-partition
  HTP plan; and
- `throughput-only`, `not-established` official parity, and an explicitly
  unpassed 80 dB project gate.

The current Android runner does not consume schema v4. Device results for this
contract come from an isolated TFLite/QNN APK because combining its LiteRT
1.0.1 runtime with the app's LiteRT 2.1.5 runtime would no longer reproduce the
reference environment. See
`android-litert-demucs6-bandbuddy-s25-2026-08-03.md` for the S25 evidence.

## Generated LiteRT runtime candidates (schema v1)

Generated LiteRT runtime candidates use an independent schema namespace. A
`contractSchemaVersion` of `1` together with
`contractKind: generated-litert-runtime-candidate` is not an older form of the
external schema v4 contract. It identifies a project-generated FlatBuffer with
reproducible source, conversion, ABI, fixture, host-parity, and runtime-target
evidence. These sidecars are parsed by
`GeneratedLiteRtCandidateContractLoader`; the loader requires exact object
keys and performs cross-field validation in addition to JSON type checks.

The initial sidecar is
`htdemucs_6s_core_smoke_2s_fp32_v1_0_0.json`, with contract ID
`htdemucs_6s_core_smoke_2s_fp32_v1_0_0@1`. Its responsibility is limited to one
fixed two-second, six-stem HTDemucs neural-core candidate. It freezes the
following facts:

- `provenance.officialModel` binds the official
  `adefossez/HTDemucs-6s` revision, the Demucs loader revision, and the exact
  safetensors weight, metadata, and bag-manifest identities. The artifact is
  generated from this official safetensors source.
- `provenance.neuralCoreReference` records the pinned MIT-licensed
  `JBNU-CILAB/demucs-lite` boundary as `boundary-reference-only`.
  `provenance.exportRecipe` separately binds the project-owned exporter and
  its executed conversion flags. The public reference is not represented as
  the code that produced this FlatBuffer.
- `sourceCandidateContract.role` is
  `upstream-model-identity-only`, and `conversion.onnxDiagnostic.role` is
  `diagnostic-parity-intermediate-not-litert-input`. Neither field claims that
  the generated LiteRT artifact was converted from the schema v3 community
  ONNX candidate or from the diagnostic ONNX export.
- `artifact` binds the 112,924,120-byte TFLite file and its SHA-256.
  `flatBuffer` and `flatBufferInspection` bind TFL3/schema 3, one subgraph,
  3,504 operators, 4,301 tensors, zero custom operators, the inspected
  operator inventory, and the exact `serving_default` signature.
- The fixed ABI is `args_0 [1,2,88200]` and
  `args_1 [1,4,2048,87]`, producing
  `output_0 [1,6,4,2048,87]` and
  `output_1 [1,6,2,88200]`. `modelSemantics` fixes the
  `L.real/L.imag/R.real/R.imag` feature order and
  `drums/bass/other/vocals/guitar/piano` stem order.
- `fixtures.inputs` and `fixtures.outputs` bind coherent raw float32-LE files
  to signature `bindingName` values, with exact shape, byte size, and SHA-256.
  `fixtures.combinedGolden` separately freezes the reconstructed six-stem
  Torch reference because it is not a LiteRT signature output.
- `hostValidation` records the completed Torch-core reconstruction and
  LiteRT-versus-Torch frequency, waveform, reconstructed-combined, and
  per-stem checks. `qualityGate` requires finite output, at least 80 dB SNR,
  and no more than `0.001` maximum absolute error. Its
  `acceptedForDeviceTesting: true` value admits a device experiment; it is not
  a device pass.
- `runtimeTarget` pins LiteRT 2.1.5 and its resolved AAR, the S25/SM8750
  target, the QNN v79 bundle manifest, and the exact CPU, GPU, and NPU plans.
  `runtimeTarget.targetDevice.status` and every plan `status` must remain
  `not-run` in this immutable candidate contract.

The official weights, generated ONNX/TFLite files, reports, and raw fixtures
remain under ignored local `models/` paths. They are not Git payloads, and the
paths recorded by this sidecar are identities rather than Android asset
packaging instructions. This contract does not add those external model files
to an APK. Only the small sidecar and its loader/test code are repository
metadata.

Device observations must be written to a separately identified result artifact
and summarized in the corresponding experiment document. Do not update this
candidate sidecar from `not-run` to a measured status, and do not treat host
parity as CPU, GPU, or NPU execution evidence. In particular, the
`conversion.profile` sample count of 88,200 and segment `2/1` describe only the
two-second smoke window. Latency, memory, delegate coverage, numerical parity,
or success from this contract must not be extrapolated to the original `39/5`
canonical window. A canonical-window artifact requires a distinct identity,
inspection, fixtures, host parity, and device validation.

### Canonical experiment closure (2026-08-05)

The canonical 7.8-second research experiment is closed; its results are not a
product-support approval.
The following results belong to digest-pinned model artifacts and dated reports;
none changes the schema-v1 smoke sidecar or grants product admission:

| Artifact | Host status | Device result | Research decision |
| --- | --- | --- | --- |
| official HTDemucs-6s | layered host pipeline passed | S25 CPU 180-second E2E RTF `0.555`; CPU and GPU+CPU neural-core strict per-stem device gates failed | CPU offline/producer-ahead feasibility only |
| HTDemucs-6s guitar-ft | deterministic EOF gate miss: `79.245 dB` versus `80 dB` | S25 CPU three-song mean E2E RTF `0.6783`; same-weight PCM16 comparison differs by at most 1-2 LSB | diagnostic-only, research-only, not admitted |
| official HTDemucs four-stem base | layered host pipeline passed | S25 CPU 180-second E2E RTF `0.617`; CPU and GPU+CPU neural-core strict per-stem device gates failed | CPU offline/producer-ahead feasibility only |

Batch 4A selected official four-stem base as that batch's research export
baseline. Psytrance ONNX sounded clearly worse with more cross-stem leakage.
Official base, the complete fine-tuned bag, and the two base/specialist hybrids
were not stably distinguishable in the multi-song blind review; this is no
demonstrated advantage, not proof of perceptual equivalence. This selection is
not product qualification or a general-purpose support decision.

The four-stem S25 report has an explicit provenance limitation: its APK/source
revision was not fully self-attested and the recorded Maven runtime digest was
unresolved. Its device numbers remain valid bounded observations tied to the
artifact, input, and report evidence, but they are not a fully self-contained
rebuild identity. The official six-stem S25 report records `sourceDirty=true`,
and the guitar-ft S25 diagnostic does not bind a resolved source identity.

All three 7.8-second artifacts also executed on the S10 LiteRT 2.1.5 CPU
(guitar-ft remains diagnostic-only). With the original serial iSTFT, mean
30-second E2E RTF was `2.960`, `2.401`, and `2.905` for official six-stem,
official four-stem, and guitar-ft. The four-worker `parallel-lanes` experiment
preserved the official six-stem and official four-stem raw-FP32 fixtures
bit-for-bit; all three variants' final output WAV SHA values matched their
serial counterparts.
The resulting RTFs were `1.292`, `1.382`, and `1.310` in that same order. Serial
and parallel runs used different APK/runner sessions, and a same-APK control
showed scheduling variance, so these are measured research comparisons rather
than a universal speedup guarantee. All remain slower than real time and are
limited to offline research on S10.

Authoritative result reports are:

- `android-litert215-demucs6-canonical7p8-host-2026-08-04.md`;
- `android-litert215-demucs6-canonical7p8-s25-2026-08-04.md`;
- `htdemucs6-guitar-ft-host-experiment-2026-08-04.md`;
- `htdemucs6-guitar-ft-litert-s25-diagnostic-2026-08-05.md`;
- `htdemucs4-batch4a-host-quality-2026-08-05.md`;
- `android-litert215-demucs4-official-s25-2026-08-05.md`;
- `android-litert215-demucs3-s10-cpu-2026-08-05.md`; and
- `android-litert215-demucs3-s10-parallel-istft-2026-08-05.md`.

### Canonical 7.8-second host candidate

The first official-safetensors canonical export is frozen as a host-only
manifest, not as an Android schema-v1 sidecar:

```text
candidateId: htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0@host-1
modelId:    htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0
profile:    canonical_7p8s
scope:      host-only-not-a-device-result
manifest:   models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/candidate-manifest.json
```

The manifest is `manifestSchemaVersion: 1`, has status
`host-pipeline-passed`, and is independently hashed. Its artifact, report,
inspection report, and manifest identities are:

| Item | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `htdemucs_6s.core.canonical_7p8s.fp32.tflite` | 117,624,880 | `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7` |
| `export-report.json` | 69,484 | `05b67b091a11b2105d07a2c8f608f2825fa35a1fa0d4cfd95bd54deb6cfebb88` |
| `flatbuffer-inspection.json` | 3,844 | `6b4763cfe62844983dd3d183a53c9ecad6ee5865f2cf1f795363e7b698b0475e` |
| `candidate-manifest.json` | 47,505 | `e22708ecbb1e43f528a3f1ff2ab33a8062c42fc36ffed1134f837426865f33e2` |

The host manifest is intentionally separate from
`htdemucs_6s_core_smoke_2s_fp32_v1_0_0@1`. The Android experimental registry,
manifest adapter, DSP harness, and device probes now resolve canonical
candidates by identity and digest. This manifest remains the authoritative
host contract, while device observations remain separate result artifacts. The
117 MB FlatBuffer is ignored local model storage, not a committed Android
asset or a product catalog entry.

#### Source and conversion freeze

The graph is regenerated from the official `adefossez/HTDemucs-6s` safetensors
source. The pinned source identities are:

| Source | Size (bytes) | SHA-256 / revision |
| --- | ---: | --- |
| `5c90dfd2.safetensors` | 54,885,744 | `d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411` |
| `5c90dfd2.json` | 10,398 | `72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e` |
| `htdemucs_6s.yaml` | 21 | `207405151270af8fd81c2373c25d27950916682ac91dca7884a11ce13dad6f58` |
| Demucs loader | - | `eeac1d15891af95b1288d2884b95baa3e5baa96c` |
| neural-core boundary reference | - | `9a2a17c7a81843c2ae49674986f9e1e8b5f6915f` |

The boundary reference is MIT-licensed and is recorded as reference-only. The
FlatBuffer is produced by the project-owned
`tools/export_htdemucs_litert_candidate.py`, whose frozen SHA-256 is
`532f5e1b7c9c30aa53963c891c94e3379f5aa18d0297d6ae21ce1e4d82e1b9ba`.
The lock file is `requirements-demucs-litert-export.txt`, SHA-256
`cb6ef5b77262c7d8594735413ecef5e841f3be1a710a48602b0620d229bf1706`.

The export environment is Python `3.12.3`, torch `2.11.0+cpu`, NumPy `2.5.1`,
`safetensors 0.8.0`, `litert-torch 0.9.1`, and `ai-edge-litert 2.1.5`. The
canonical conversion is direct from the project-owned PyTorch neural-core
module; no ONNX input is used. The flags are fixed to
`strictExport=true`, `lightweightConversion=false`,
`runtimeConstantFolding=false`, `enableX64=false`, and
`deterministicPositionalEmbedding=true`.

#### ABI and semantics

The inspected FlatBuffer is TFL3/schema 3, one subgraph, 3,432 operators,
4,176 tensors, 4,179 buffers, 26 operator codes, and zero custom operators.
The minimum-runtime metadata says `2.16.0`; this is converter metadata and is
not a device compatibility result. The `serving_default` signature has these
actual tensor indices and names:

| Binding | Tensor index | Tensor name | Shape | Axes |
| --- | ---: | --- | --- | --- |
| `args_0` | 0 | `serving_default_args_0` | `[1,2,343980]` | batch, channel, sample |
| `args_1` | 1 | `serving_default_args_1` | `[1,4,2048,336]` | batch, feature, frequency, frame |
| `output_0` | 4170 | `serving_default_output_0_output` | `[1,6,4,2048,336]` | batch, stem, feature, frequency, frame |
| `output_1` | 4175 | `serving_default_output_1_output` | `[1,6,2,343980]` | batch, stem, channel, sample |

The sample rate is 44,100 Hz, the rational source segment is `39/5`, and the
canonical window is 343,980 samples. Feature order is
`L.real, L.imag, R.real, R.imag`; stem order is
`drums, bass, other, vocals, guitar, piano`. The operator histogram and all
indices are those of this FlatBuffer and must not be copied from a different
converter run.

#### Host DSP contract

The neural core is only one part of the contract. Host preprocessing and
reconstruction are frozen as follows:

```text
STFT:  n_fft=4096, hop=1024, periodic Hann, normalized=true, center=true,
       pad_mode=reflect, outer pad=(1536,1620), drop Nyquist, crop 2 frames
iSTFT: restore zero Nyquist, restore two frames on both ends, normalized=true,
       center=true, reconstruction length=347136, crop [1536,345516)
branch combination: frequency iSTFT + time waveform
global normalization: mean across stereo channels; unbiased std (correction=1)
                       plus epsilon=1e-8 before OLA, inverse after OLA
OLA: window=343980, stride=257985, overlap=85995 (25%), shifts=0,
     transition power=1, ascending-offset FP32 accumulation,
     EOF tail weight=triangle-prefix
```

The two-window fixture has `trackSamples=515970` and offsets `[0,257985]`.
The second window copies source `[214988,515970)`, pads 42,998 samples on the
right, then center-crops 42,997 samples on the left and 42,998 on the right.
The Torch implementation of this plan is bitwise equal to the official
`apply_model` split/overlap-add oracle. This exact tail behavior is part of the
contract; a center-sliced triangle weight or zeroing the left context is not
equivalent.

#### Host parity and gate interpretation

The frozen single-window fixture includes raw spectrum, raw frequency output,
frequency-branch iSTFT, raw time output, and combined six-stem waveform. Aggregate
LiteRT-versus-Torch results are:

| Layer | SNR (dB) | Max absolute error |
| --- | ---: | ---: |
| frequency output | 99.7049 | `1.158714e-04` |
| frequency iSTFT waveform | 98.0434 | `5.152076e-06` |
| time waveform output | 124.6445 | `5.736947e-07` |
| combined waveform | 100.7594 | `5.237758e-06` |

For the two-window OLA fixture, aggregate results are 98.4269 dB and
`5.550683e-06` max error over the full track, 105.4076 dB and
`1.929700e-06` over the overlap, and 117.7663 dB and `8.195639e-07` over the
EOF region. The Torch OLA is bitwise equal to the official oracle. The lowest
full-track per-stem SNR is 85.3562 dB; low-energy EOF stems are evaluated with
the absolute-error rule and pass.

The gate is deliberately layered:

- `uniformTensorGatePassed=false`: the first OLA window's raw frequency latent
  reaches max error `0.002628326` (SNR `94.6801 dB`), so a single uniform
  `1e-3` absolute gate would be misleading.
- `hostPipelineGatePassed=true`: waveform/iSTFT/combined and OLA checks are
  finite and pass 80 dB plus `1e-3` absolute error; references with RMS at or
  below `1e-3` use the stricter `1e-6` absolute-error rule.
- `acceptedForDeviceTesting=true` means only that this host manifest authorized
  the later separately recorded device experiment. It is not itself CPU, GPU,
  or NPU evidence and does not imply product support.

All eight raw fixture identities, including the 25% OLA input and golden output,
are in `candidate-manifest.json`; the report and manifest are the source of
truth rather than a hand-copied tensor list.

### Official four-stem canonical candidate

The official `955717e8.safetensors` base checkpoint was exported under the same
7.8-second external-DSP boundary as a separate host candidate:

```text
modelId:  htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0
artifact: htdemucs_4s.core.canonical_7p8s.fp32.tflite
bytes:    178042000
SHA-256:  9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81
manifest SHA-256:
          134642ea71cfbb8174cb8f27a6696f3620b03d6556afcfade1457a4245b8a879
```

Its inputs remain `[1,2,343980]` waveform and `[1,4,2048,336]` spectrum;
outputs are `[1,4,4,2048,336]` frequency and `[1,4,2,343980]` time branches in
`drums, bass, other, vocals` order. The FlatBuffer has 3,458 operators and zero
custom operators. Its layered host gate passed, including 121.256 dB
single-window and 90.781 dB full-OLA aggregate comparisons. S25 CPU and
GPU+CPU neural-core execution completed, but the strict per-stem device gate
failed for both. No GPU E2E audio batch ran. See
`android-litert215-demucs4-official-s25-2026-08-05.md`; host admission
authorizes research execution only.

### Guitar-ft diagnostic candidate

The guitar-ft artifact shares the official six-stem ABI and DSP contract but
has a distinct weight, artifact, and diagnostic-manifest identity:

```text
modelId: htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0
bytes:   117729544
SHA-256: ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5
kind:    generated-litert-diagnostic-candidate
status:  diagnosticOnly, researchOnly, hostAdmissionStatus=not-admitted
```

Same-weight Torch is the conversion oracle; similarity to official weights is
not a conversion gate. The full-OLA aggregate comparison passes, but the
canonical EOF region reproducibly measures 79.245 dB and fails the frozen
80 dB host threshold. S25 PCM16 renders closely match same-weight Torch, so the
observed listening changes are attributed to the fine-tuned weights rather
than LiteRT conversion within the measured proxies. Listening preferred the
guitar and piano separation and found substantially less vocal leakage in the
Imagine piano stem, while slight drum contamination was reported in vocals on
some material. MoisesDB training-data terms still require independent review;
the artifact is not a product-support candidate.

## Historical candidate: UVR-MDX-NET-Inst_Main

Status: early research record; not the canonical current benchmark contract

Local development path:

- `models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx`

Do not commit the model file. It is stored locally only for development and personal testing.

Source:

- Hugging Face mirror: `https://huggingface.co/seanghay/uvr_models`
- Reference implementation checked during research: `https://github.com/seanghay/uvr-mdx-infer`

License note:

- The app is currently for personal use only.
- Model redistribution rights are not confirmed.
- If this project is ever packaged for wider distribution, model licensing must be resolved before bundling weights.

ONNX graph:

- IR version: `6`
- Opset: `ai.onnx` version `13`
- Input name: `input`
- Input dtype: `float32`
- Input shape: `[batch_size, 4, 2048, 256]`
- Output name: `output`
- Output dtype: `float32`
- Output shape: `[batch_size, 4, 2048, 256]`

DSP parameters:

- Sample rate: `44100 Hz`
- Channels: stereo
- FFT size: `6144`
- Hop length: `1024`
- Frequency bins used by model: `2048`
- Full real FFT bins: `3073`
- Time frames per model window: `256`
- Window length in samples: `261120`
- Generation stride before trim: `254976`
- Trim per window: `3072` samples at each edge
- Window function: periodic Hann

Tensor layout:

- The model does not consume waveform samples directly.
- Each input item is a stereo STFT representation with four channels:
  - left real
  - left imaginary
  - right real
  - right imaginary
- Frequency bins are truncated from `3073` to `2048` before inference.
- Model output has the same layout and is reconstructed with inverse STFT.
- For `Inst_Main`, the reconstructed model output should be treated as the instrumental stem.
- The vocals stem should be computed as `mixture - instrumental`.

Android implication:

- Phase 4 cannot just pass PCM into ONNX Runtime.
- Android needs a matching STFT/ISTFT implementation or a model variant with DSP folded into the graph.
- For the first Android ONNX integration, using this exact model means implementing periodic Hann windows, reflect padding, chunking, frequency padding, and overlap-add behavior.
- The current Android app resamples decoded input audio to `44100 Hz` before MDX inference when the source file uses another sample rate.
- Current separated WAV outputs are written at `44100 Hz`.
- Personal-test APK builds include this ONNX file from the local ignored `models/uvr-mdx` directory as an Android asset, then copy it into app-private storage before creating an ONNX Runtime session.
