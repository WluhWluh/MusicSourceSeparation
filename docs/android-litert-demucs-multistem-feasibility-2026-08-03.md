# Android LiteRT Demucs multi-stem feasibility and source freeze

Date: 2026-08-04 (host local time; source freeze began 2026-08-03)

Closure update: 2026-08-05

Status: historical source-freeze and batch-planning baseline. Canonical
official 6-stem and 4-stem export/S25 work, guitar-ft diagnostics, Batch 4A,
and the S10 CPU serial/parallel matrix are complete in separate dated reports.

This branch evaluates larger 4-stem and 6-stem source-separation workloads only
to learn device performance limits. It is not a proposal to ship either model,
and it does not establish product support, separation quality, battery cost, or
full-song stability.

Forward-looking language below records the decision state at source freeze.
Where it conflicts with a dated result report, the dated report is
authoritative. Completion here means the bounded research batch ran; it does
not mean product qualification or admission into Booming SS.

The sibling BoomingMusic multi-stem contract/playback roadmap remains the
application authority for cache shape, N-stem playback, lifecycle, and catalog
admission. These model-execution experiments do not close those application
gates.

## Decision summary

Two official single-model HTDemucs candidates are now pinned and downloaded:
`htdemucs` (4 stems) and `htdemucs_6s` (6 stems). The official artifacts are
maintainer-published safetensors plus exact JSON/YAML metadata. Community ONNX
exports are also pinned as conversion sources, not as trusted runtime or quality
oracles.

### Result closure (2026-08-05)

The external-DSP neural-core route proposed by this source freeze was carried
through on LiteRT 2.1.5. The current bounded result is:

| Candidate / batch | Completed evidence | Decision |
| --- | --- | --- |
| official HTDemucs-6s canonical | host pipeline passed; S25 CPU 180-second E2E RTF `0.555`; hybrid GPU+CPU tested | strict per-stem device gate failed; CPU offline/producer-ahead research only |
| guitar-ft six-stem | host and S25 diagnostic completed; three-song S25 mean E2E RTF `0.6783` | EOF host gate failed at `79.245 dB`; research-only/not admitted; license review still required |
| Batch 4A four-stem quality | five host variants and blind review completed | Psytrance clearly worse/more cross-talk; other four not stably distinguishable; official base selected |
| official HTDemucs four-stem canonical | host pipeline passed; S25 CPU 180-second E2E RTF `0.617`; hybrid GPU+CPU tested | strict per-stem device gate failed; CPU offline/producer-ahead research only |
| S10 CPU serial matrix | all three artifacts completed three 30-second tracks | RTF `2.960` / `2.401` / `2.905`; serial iSTFT dominated |
| S10 four-lane iSTFT | raw FP32 and output WAVs preserved bit-for-bit | RTF `1.292` / `1.382` / `1.310`; still slower than real time and offline-only |

The S25 hybrid OpenCL profiles delegated only 158 neural-core nodes and did not
provide a useful product path. QNN remains outside this closure: the earlier
six-stem smoke produced non-empty IR but failed VTCM schedule allocation before
model creation, and no NPU inference occurred. S10 GPU/NPU were not exercised
by the CPU closure batches.

The guitar-ft listening result is material but does not waive its gates.
Guitar and piano were repeatedly preferred; the Imagine piano stem had much
less audible vocal leakage. Slight drum contamination in vocals was also
reported. Same-weight S25/Torch comparisons and fixed-mask proxies attribute
these changes primarily to the fine-tuned weights rather than LiteRT
conversion, within a no-ground-truth experiment.

Detailed authority:

- `android-litert215-demucs6-canonical7p8-host-2026-08-04.md`;
- `android-litert215-demucs6-canonical7p8-s25-2026-08-04.md`;
- `htdemucs6-guitar-ft-host-experiment-2026-08-04.md`;
- `htdemucs6-guitar-ft-litert-s25-diagnostic-2026-08-05.md`;
- `htdemucs4-batch4a-host-quality-2026-08-05.md`;
- `android-litert215-demucs4-official-s25-2026-08-05.md`;
- `android-litert215-demucs3-s10-cpu-2026-08-05.md`; and
- `android-litert215-demucs3-s10-parallel-istft-2026-08-05.md`.

### Historical smoke checkpoint (2026-08-04)

The first project-generated executable candidate is now frozen separately as
`htdemucs_6s_core_smoke_2s_fp32_v1_0_0@1`. It was exported from the official
`5c90dfd2.safetensors` and passed the host parity gate. The ignored local
FlatBuffer is 112,924,120 bytes with ABI:

```text
args_0   [1,2,88200]       args_1   [1,4,2048,87]
output_0 [1,6,4,2048,87]   output_1 [1,6,2,88200]
```

On `SM-S9310 / SM8750` with LiteRT `2.1.5`, one frozen two-second window gave:

| Plan | Prepare | Inference | RTF | Delegation / decision |
| --- | ---: | ---: | ---: | --- |
| Compiled CPU + XNNPACK | 597.32 ms | 1,148.98 ms | 0.574 | 3,308/3,504 nodes, 103 partitions; finite and output-level parity pass |
| Strict GPU OpenCL BUFFER FP32 | failed | not run | n/a | 158/3,504 nodes, 3 partitions; `CompiledModel.create` failed |
| GPU + CPU OpenCL BUFFER FP32 | 1,696.92 ms | 637.11 ms | 0.319 | real `LITERT_CL` plus XNNPACK remainder; output-level parity pass |
| NPU QNN v79, prepare optimization | incomplete | not run | n/a | non-empty IR, VTCM allocation failure; finalize/model creation not complete |

The hybrid row is a valid mixed-backend measurement and is about 44.5% faster
than this CPU smoke run, but it is not a strict GPU result: only 158 nodes ran
on OpenCL and 3,152 of the 3,347 remaining delegate-eligible nodes ran in 105
XNNPACK partitions. The device parity JSON passes the frequency, waveform, and
combined 80 dB gates; low-energy per-stem diagnostics are below 80 dB for some
stems and must not be summarized as a strict per-stem pass.

The NPU log records allocation within a physical VTCM pool of `0x800000` bytes,
185 allocator passes, and `Cannot allocate for schedule due to vtcm limits`,
followed by a retry. The probe was stopped before `QnnGraph_finalize` completed;
there is no `CompiledModel` and no NPU inference result. The generated QNN IR is
therefore evidence of attempted compilation only. These results are archived
under `outputs/htdemucs-smoke2s-s25-20260804/` and detailed in
`android-litert215-demucs6-smoke2s-s25-2026-08-04.md`.

This smoke result does not satisfy, estimate, or replace the canonical
`[1,2,343980]` / 336-frame contract. Attention and activation costs do not scale
linearly with window duration.

The complete 7.8-second waveform ONNX graphs execute on desktop ORT and return
the expected finite outputs, but sampled process RSS peaks at about 2.70 GiB for
4 stems and 2.55 GiB for 6 stems. A LiteRT conversion coverage probe on the
6-stem graph did not produce TFLite and found 683 blocking nodes. The main
blocker is 680 FLOAT64 `ScatterND` nodes expanded from the embedded iSTFT
envelope loop.

At this checkpoint, the recommended mobile experiment was:

1. keep STFT/iSTFT and overlap-add on the host;
2. export a static tensor-only HTDemucs neural core;
3. start with a 2-second allocation and op-coverage profile;
4. run one canonical 7.8-second window on the S25 CPU first; and
5. try the canonical S10 window only after the S25 completes within an explicit
   memory bound (a 2-second smoke may still be run independently).

At this checkpoint, GPU and QNN evidence existed only for the generated
two-second smoke profile. Later canonical reports added S25 GPU+CPU evidence,
but no strict canonical GPU qualification and no canonical QNN inference. QNN
remains S25-only with the recorded runtime matrix; there is no SM8150 pack for
the S10.

## Branch and scope

The source-freeze work was isolated on
`experiment/model-matrix-demucs-multistem`, branched from
`experiment/model-matrix-batch0-9662` at `3510d5c`. The sibling HQ4 smoke test
remains on `experiment/model-matrix-hq4-smoke` at `1e59b3a`.

The source-freeze batch completed:

- official candidate identity and license/source review;
- pinned downloads with full SHA-256 verification;
- a separate, fail-closed tensor-only candidate contract schema;
- ONNX checker, graph inventory, and three zero-input ORT windows per candidate;
- one bounded LiteRT conversion coverage probe; and
- a device feasibility decision and next-batch order.

The follow-up smoke batch additionally completed the generated LiteRT artifact,
host parity, S25 CPU run, strict GPU prepare attempt, hybrid GPU/CPU run, and
bounded QNN prepare attempt.

At the source-freeze checkpoint, the batch had not performed:

- official PyTorch-to-ONNX numerical parity on this host;
- canonical-window ONNX-to-TFLite conversion;
- any canonical S10 or S25 Demucs execution; or
- full-song, sustained, energy, or thermal performance testing; or
- application integration.

At the smoke checkpoint, the batch had not performed canonical 7.8-second
execution, NPU inference, or product-quality testing. The dated closure reports
later completed canonical CPU/hybrid-GPU research execution; NPU inference and
product qualification remain unclaimed.

## Frozen official candidates

The preferred source artifacts are the maintainer's Hugging Face safetensors,
not the legacy pickle checkpoints. Source code and loader behavior are pinned to
`adefossez/demucs@eeac1d15891af95b1288d2884b95baa3e5baa96c`.

| Property | `htdemucs` 4-stem | `htdemucs_6s` 6-stem |
| --- | --- | --- |
| HF repository | `adefossez/HTDemucs` | `adefossez/HTDemucs-6s` |
| HF revision | `bf35a81b663819a8255c8fefee17f9d812b786b5` | `053e1404489b3dc58bf718224fac4b7316de8c93` |
| Weight | `955717e8.safetensors` | `5c90dfd2.safetensors` |
| Bytes | 84,025,440 | 54,885,744 |
| SHA-256 | `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd` | `d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411` |
| Stored state | 41,984,456 FP16 values | 27,414,996 FP16 values |
| Bag members | 1 (`955717e8`) | 1 (`5c90dfd2`) |
| License metadata | MIT | MIT |

The matching metadata identities are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `955717e8.json` | 12,087 | `12540373de858920b60002ebfbe17738860dbf2e89df625dc3b7875af2d28491` |
| `htdemucs.yaml` | 21 | `239c445d0b14454d541ad8bd9bb271c9e536d267e8a4625208744cbb2e7bb66c` |
| `5c90dfd2.json` | 10,398 | `72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e` |
| `htdemucs_6s.yaml` | 21 | `207405151270af8fd81c2373c25d27950916682ac91dca7884a11ce13dad6f58` |

Legacy official `.th` files were also downloaded for provenance comparison:

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| `955717e8-8726e21a.th` | 84,141,911 | `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4` |
| `5c90dfd2-34c22ccb.th` | 54,996,327 | `34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd` |

The safetensors plus their JSON sidecars are preferable because class, kwargs,
and tensor metadata can be inspected without unpickling code. Loading them into
a normal PyTorch
module creates FP32 parameters unless the module is explicitly kept in FP16;
the file size is not the runtime parameter-memory size.

## Exact model contract

Both official bags contain one HTDemucs model and use the same native waveform
window:

| Property | Value |
| --- | --- |
| Sample rate / channels | 44,100 Hz / stereo |
| Segment | exact `39/5` seconds (7.8 seconds) |
| Segment samples | 343,980 |
| Input | float32 `[1, 2, 343980]` (`batch, channel, sample`) |
| FFT / hop | 4,096 / 1,024 |
| Base channels / depth | 48 / 4 |
| Transformer layers / heads | 5 / 8 |
| 4-stem output | float32 `[1, 4, 2, 343980]` |
| 6-stem output | float32 `[1, 6, 2, 343980]` |

The output order is frozen from the official model kwargs, not inferred from a
README:

```text
4-stem: drums, bass, other, vocals
6-stem: drums, bass, other, vocals, guitar, piano
```

The 6-stem model must not be described as the larger parameter model merely
because it has more outputs. Its `bottom_channels=0` and 27.4 million stored
state values make it materially smaller than the 4-stem model, whose
`bottom_channels=512` and stored state contain about 42.0 million values. The
6-stem workload still has larger decoder/output buffers and more output data.

| Tensor | 4-stem bytes | 6-stem bytes |
| --- | ---: | ---: |
| Waveform input | 2,751,840 | 2,751,840 |
| Packed waveform output | 11,007,360 | 16,511,040 |

`htdemucs_ft` is deliberately excluded from this first batch. It is a bag of
four approximately 84 MB model files with one-hot per-stem weights and requires
four forwards. It can be retained as a later extreme stress workload, but it is
not one joint 4-output forward.

## ONNX conversion sources

The official repositories currently publish no ONNX or TFLite model. The
following community exports are pinned to the open-source
`StemSplit/demucs-onnx` exporter at
`85db5c80aba33f0f2bdf88034a4be6539feec85b`:

| Model | HF revision | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `htdemucs_fp16weights.onnx` | `d54ed9eb60e258ea82131c6ee14578628816456a` | 165,612,636 | `d05c269d0178d2a72ad484b10b11dd370193fc923201c3b27a99f848745db70a` |
| `htdemucs_6s_fp16weights.onnx` | `49df9b6989cf2150840ea65b0bef77a2e471b678` | 136,428,532 | `7ce55792e2231c93fbf92de95f5fd5b3a5e6c89f7db690dfd693e8f1dce56869` |

| Conversion provenance file | Bytes | SHA-256 |
| --- | ---: | --- |
| `demucs-onnx-LICENSE.txt` (shared) | 1,066 | `002e90589e1030d13e3708342c89faf67b520273793b6a2c77c4d9bd9a5ec7e7` |
| `htdemucs-onnx-README.md` | 8,846 | `671c5f3f0bcc8f28f637365996004b3f71a065673ff1833ff7548d7835cdf086` |
| `htdemucs-6s-onnx-README.md` | 8,513 | `42c63ffad79357ccd93f9b93a8ed7935a742bd2e08035321f7e6623c0635661c` |

The exporter predates the maintainer's safetensors publication and loads the
legacy official `.th` checkpoints. Each v3 sidecar therefore records that
exact legacy file as `conversion.exporterInput`; the safetensors are recorded
separately as the maintainer-published source used by the later canonical
Torch/LiteRT oracle.
The exporter repository license and each fixed ONNX model-card file are pinned
as separate conversion artifacts. The ONNX source contract states float32 I/O,
no quantization, and `runtimePrecisionStatus=not-established`; the internal
storage inventory is not a claim about a future delegate's activation type.

Both are opset 17 graphs produced by PyTorch 2.4.1. The exporter replaces
complex STFT/iSTFT with large convolutional DFT kernels, freezes positional
randomness, and replaces fused multi-head attention. The source-freeze report
recorded a single synthetic parity check before the later FP32/FP16 and
checkpoint-equivalence diagnostics. Those later diagnostics do not turn these
graphs into the canonical mobile producer; they remain conversion and
allocation probes only. Their graph I/O is float32,
but storage is mixed: most neural weights are float16, while embedded DFT/iSTFT
constants include float16, float32, float64, and int64. The sidecars record the
exact dtype inventories; `fp16weights` must not be interpreted as an FP16
activation or LiteRT runtime contract.

The model protobuf leaves output dimensions symbolic. ORT resolves them to the
contracted static shapes. Any TFLite conversion staging copy must explicitly
freeze the complete input and output shapes before conversion.

## Tensor-only contract schema v3

Existing schema-v2 sidecars and runner behavior remain unchanged. A separate
strict schema v3 now describes tensor-only multi-stem experiments:

- one or more static float32 inputs and outputs;
- explicit tensor indices, names, shapes, and semantic axes;
- ordered 4-stem or 6-stem bindings;
- packed stems along an axis or one whole output tensor per stem;
- exact waveform sample rate and rational segment duration;
- separate maintainer safetensors, exact legacy exporter input, and community
  ONNX conversion-source identities, including pinned exporter license and
  model-card files;
- conversion-source precision inventory: float32 I/O, no quantization, mixed
  float16/float32/float64/int64 storage, and runtime precision not established;
- a mandatory `contractKind: tensor-only-candidate` discriminator.

Schema v3 is deliberately candidate-only and cannot describe a TFLite runtime.
This prevents the downloaded ONNX files from being passed off as LiteRT device
artifacts. For any future imported ONNX candidate, an executable tensor-only
runtime contract must be a separate type that references the candidate contract
ID and sidecar SHA, then freezes the conversion, actual FlatBuffer signature,
and TFLite identity. The completed official 4-stem and 6-stem experiments use
their own generated host manifests and identity-indexed harnesses; they do not
promote these v3 ONNX sidecars.

Current sidecars:

| Contract | Bytes | SHA-256 |
| --- | ---: | --- |
| `htdemucs_4s_waveform_7p8s_onnx@3` | 6,360 | `f53a3cdad1d162541b8b353257d716e25b4178a433d8629a4804b3178484277d` |
| `htdemucs_6s_waveform_7p8s_onnx@3` | 6,604 | `cc8dafb665602b6b8410ca87c6450e29ff4986990f22613e6451cf8da9f4c513` |

The general Android inference service remains schema-v2-only and must not be
used for these sidecars. The canonical device experiments used a separate
identity-indexed benchmark harness, not an implicit v3 service upgrade.
Likewise, the MDX QNN audio runner remains intentionally out of scope because
it hard-codes target plus residual audio reconstruction.

## Desktop execution probe

Host:

- Windows 11 Enterprise Evaluation 10.0.22621;
- AMD Ryzen AI 9 HX 370, 12 cores / 24 logical processors;
- 32 GiB physical memory;
- ONNX Runtime 1.26.0 CPUExecutionProvider; and
- runtime-default thread count.

Each observation loaded one model in a fresh process, then ran once with a zero
float32 waveform. RSS was sampled every 10 ms. This is an
allocation/executability probe, not a representative audio latency benchmark.

| Model | Observation | ORT setup | One window | Peak RSS | Output | Finite |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 4-stem | 1 | 5.241 s | 2.286 s | 2,700.7 MiB | `[1,4,2,343980]` | yes |
| 4-stem | 2 | 6.291 s | 2.401 s | 2,702.1 MiB | `[1,4,2,343980]` | yes |
| 4-stem | 3 | 4.908 s | 2.256 s | 2,708.7 MiB | `[1,4,2,343980]` | yes |
| 6-stem | 1 | 4.491 s | 2.521 s | 2,552.2 MiB | `[1,6,2,343980]` | yes |
| 6-stem | 2 | 5.504 s | 2.364 s | 2,549.5 MiB | `[1,6,2,343980]` | yes |
| 6-stem | 3 | 4.891 s | 2.237 s | 2,552.2 MiB | `[1,6,2,343980]` | yes |

The 6-stem model's lower peak is consistent with its smaller transformer/core,
despite its larger output. These desktop numbers cannot be projected directly
onto Android ORT or LiteRT. Numerically, the desktop RSS is higher than the
roughly 1.90 GiB Android ORT peak PSS measured for HQ4 on the S10 and S25, but
RSS/PSS and host/device environments are different memory measures and are not
a shared admission limit.

## Graph and conversion probe

| Graph property | 4-stem | 6-stem |
| --- | ---: | ---: |
| Nodes | 24,917 | 24,819 |
| Initializers | 549 | 541 |
| `Constant` | 11,684 | 11,654 |
| `Range` | 697 | 697 |
| `ScatterND` | 684 | 684 |
| `LayerNormalization` | 26 | 26 |
| `ConvTranspose` | 10 | 10 |
| `Softmax` | 10 | 10 |

Only the 6-stem model was probed with the related `bss-tflite` conversion environment
at revision `28d9a076c8a44980085a059e6224768ae77f9c8a`, using onnx2tf 2.6.6,
TensorFlow 2.20.0, and ai-edge-litert 2.1.2. The probe was stopped after about
seven minutes without a TFLite artifact. It produced an op-coverage report for
24,595 preprocessed nodes:

| Result | Count | Reason |
| --- | ---: | --- |
| Supported by the converter inventory | 23,912 | not proof of end-to-end conversion |
| Unsupported `ScatterND` | 680 | iSTFT data inferred as FLOAT64 |
| Unsupported `LayerNormalization` | 2 | converter inferred `[1,1,1]` input against scale `[384]` |
| Invalid `Concat` | 1 | converter saw only one input |

The 97.2% supported node share is not an acceptance result. Any one of these
nodes blocks a builtin-only graph, and 680 expanded iSTFT updates would also be
poor delegate input even if represented as custom or CPU fallback ops.

The direct probe also demonstrated that onnx2tf may rewrite its input file.
The original 6-stem ONNX was restored from the pinned URL and reverified after
the probe. Future attempts must always operate on a staging copy.

A narrow full-waveform retry is possible: precompute the FP32 iSTFT envelope
before export, freeze every public shape, and decompose or correct the two layer
normalizations. It is worth one desktop conversion check, but it should not
displace the external-DSP core path because the convolutional DFT constants and
dynamic indexing remain poor GPU/QNN workload characteristics.

## Device feasibility: historical estimate and measured closure

The closest measured device boundary is the HQ4 short smoke test. HQ4 is only a
59 MB TFLite graph with a 10 MiB input, yet its one-warmup/five-measured profile
reached the following P50 wall times and sampled process PSS peaks (QNN also
includes its setup time):

| Device | CPU (P50 / peak PSS) | Bounded GPU (P50 / peak PSS) | QNN (P50 / peak PSS) |
| --- | --- | --- | --- |
| S10 / SM8150 | 8.145 s / 1,182.9 MiB | 3.788 s / 926.9 MiB | unavailable in current runtime matrix |
| S25 / SM8750 | 2.480 s / 1,178.5 MiB | 0.739 s / 909.8 MiB | 0.218 s / 1,561.9 MiB plus 17.23 s setup |

Those values were not Demucs estimates. They established that activation and
delegate memory already dominated on a much smaller graph. The planned S25-first
ordering was subsequently executed and resolved as follows:

| Device/backend | Measured closure | Remaining boundary |
| --- | --- | --- |
| S25 CPU | official 6-stem and 4-stem canonical artifacts completed 30/180-second E2E batches; guitar-ft completed three 30-second diagnostics | both official artifacts failed the strict per-stem device gate; guitar-ft is not host-admitted |
| S10 CPU | three 7.8-second artifacts (guitar-ft diagnostic-only) completed serial and four-lane 30-second matrices | four-lane RTF remains `1.292-1.382`; offline-only |
| S25 GPU | canonical official profiles completed only as GPU+CPU FP32 hybrids with 158 OpenCL nodes; latency/memory did not justify them | no strict GPU product route; the tested hybrids are rejected |
| S10 GPU | not run in these closure batches | no GPU claim |
| S25 QNN | smoke IR generation reached VTCM schedule failure before model creation | no NPU inference; unchanged graph closed to option-only retries |
| S10 NPU | unavailable in the recorded runtime matrix | do not label CPU fallback as NPU execution |

The successful S25 CPU allocation result justified the later S10 CPU matrix,
but neither device's execution result overrides the numerical and product
admission gates.

## Historical recommended batch sequence

The sequence below is retained as the source-freeze planning baseline. It
explains why artifacts and backends were tested in this order; the dated result
reports above supersede its then-future status statements.

### Batch M0A: source freeze and structural probe - complete

- Pin official model, metadata, bag, loader, exporter, and ONNX revisions.
- Verify every byte length and SHA-256.
- Freeze source order, tensor axes, and exact 7.8-second window.
- Run ONNX checker and three independent finite-output allocation probes per
  candidate (one is the minimum structural gate).
- Record converter coverage and the no-TFLite boundary.

### Batch M0B: historical ONNX-oracle plan

The canonical project-owned route ultimately converted the official-weight
PyTorch neural core directly and froze Torch/LiteRT DSP and OLA parity. The
community waveform ONNX remained diagnostic/conversion-source evidence rather
than the producer of the canonical mobile artifact.

- Build an isolated, version-pinned PyTorch/Demucs environment.
- Load the official safetensors through their exact serialized kwargs.
- Compare official PyTorch and community ONNX on zero, seeded random, impulse,
  low-amplitude, and one real single-window tensor.
- Record per-output max absolute error, RMSE, SNR, cosine, and non-finite count.
- Freeze each fixture as raw little-endian float32 bytes with a SHA-256 and
  shape sidecar: all-zero; NumPy `PCG64` with a recorded seed and amplitude for
  random; a documented channel/sample/amplitude impulse; a documented
  low-amplitude scale; and a real single-window crop identified by source-file
  SHA-256 and crop offset. Do not regenerate these implicitly at test time.
- Do not use the community full-song overlap-add helper as the official oracle;
  its linear edge fade differs from the official triangular weighting.
- Gate progression on exact shapes and all-finite outputs. For impulse and
  silent fixtures use maximum absolute error at most `1e-3` (and `1e-6` for a
  truly silent reference); for random, low-amplitude, and real fixtures also
  require at least 80 dB SNR per non-silent output. A failed parity gate blocks
  device performance claims for that core, while structural/allocation results
  remain explicitly marked as unverified.

### Batch M0C: export the neural core (canonical follow-up complete)

Create two distinct static profiles for each model and never combine their
results:

- `smoke_2s`: waveform `[1,2,88200]`, 87 spectral frames, for allocation and
  op coverage only; and
- `canonical_7p8s`: waveform `[1,2,343980]`, 336 spectral frames, for the
  actual performance level.

At this checkpoint, the six-stem `smoke_2s` profile had been generated and
device-tested under
the independent generated-runtime contract. Its exporter, official weight
identities, FlatBuffer inspection, fixtures, and host metrics are frozen by
`app/src/main/assets/benchmark-contracts/htdemucs_6s_core_smoke_2s_fp32_v1_0_0.json`.
Later batches exported and host-validated distinct canonical official 6-stem
and 4-stem artifacts, then ran both on S25 and S10. A separate four-stem
two-second smoke was no longer needed once its canonical host/S25 gates ran.

The neural-core tensor contract is:

| Tensor | Axes | Static shape |
| --- | --- | --- |
| waveform input | batch, channel, sample | `[1,2,N]` |
| host-STFT input | batch, feature, frequency, frame | `[1,4,2048,F]` |
| time-branch output | batch, stem, channel, sample | `[1,S,2,N]` |
| spectral-branch output | batch, stem, feature, frequency, frame | `[1,S,4,2048,F]` |

Here `S` is 4 or 6. The canonical export must be generated from the official
safetensors and exact serialized kwargs; the legacy `.th` files are provenance
only and must not be used as a new core source. For `smoke_2s`, keep the
official `use_train_segment=true` behavior but explicitly set the deserialized
model's `segment=Fraction(2,1)` before tracing/export. Changing only the dummy
input shape is insufficient because the default forward pads back to 7.8 s.
The sidecar must record that override and the observed FlatBuffer frame count.
`smoke_2s` then fixes `N=88200, F=87`; `canonical_7p8s` fixes `N=343980,
F=336`. Both output branches bind the same ordered logical stem set; they are
not eight or twelve distinct stems. The four-element `feature` axis is ordered
`left-real, left-imag, right-real, right-imag` for both host-STFT input and
spectral output. At 7.8 seconds, the
expected combined input is about 13.1 MiB. Combined outputs are about 52.5 MiB
for 4 stems and 78.75 MiB for 6 stems. Dense attention also creates individual
score tensors on the order of 110-220 MiB before accounting for convolution
activations and skip connections.

The source-freeze plan originally required both a static ONNX core and a static
TFLite FlatBuffer. The completed canonical route superseded the ONNX
intermediate: it converted the project-owned official-weight PyTorch neural
core directly, and froze the exact FlatBuffer signatures, converter revision
and flags, artifact size/SHA-256, host execution, zero undeclared custom/Flex
ops, and the layered waveform/OLA numerical gate. Freeze
FP32 baseline and any FP16/quantized candidate as separate runtime profiles;
their converter flags, accumulator type, delegate-allowed type, and
quantization scheme belong in the future runtime contract. The eventual
executable runtime contract must reference the candidate sidecar ID and
SHA-256. Passing
`smoke_2s` does not satisfy any `canonical_7p8s` gate.

The 2-second result cannot be linearly scaled to 7.8 seconds because attention
cost grows approximately quadratically with token length.

### Batch M1: CPU allocation boundary (historical smoke protocol; canonical follow-up complete)

At the smoke checkpoint, the generated `smoke_2s` S25 CPU allocation/inference
probe
completed with finite outputs, XNNPACK coverage `3308/3504`, prepare
`597.32 ms`, inference `1148.98 ms`, and retained end-of-run PSS `88,925 KiB`.
That PSS value is a final snapshot, not a process peak. It remains only a smoke
result. The later canonical official 6-stem and 4-stem S25 CPU runs completed,
followed by the S10 CPU matrix; their dated reports replace the historical
unknown status without retroactively changing this smoke measurement.

- Run the 2-second allocation smoke independently on each available device; it
  is not evidence for the canonical window. For the canonical window, use S25
  first with fresh-process setup, zero warmups, one measured inference, and a
  180-second watchdog.
- Record setup, wall/CPU time, output shapes, finite counts, peak PSS, swap,
  thermal state, and LMK/abort evidence.
- Classify `PASS` only when canonical allocation and inference complete with
  exact finite outputs, no unexpected runtime/provider fallback or kill,
  recorded XNNPACK delegation count, minimum observed `MemAvailable` of at
  least 512 MiB, and swap growth no greater than 256 MiB. Classify `RISK` when
  execution completes but crosses either memory line. Allocation failure, OOM,
  LMK, timeout, or non-finite output is `FAIL`. A builtin op executing on the
  LiteRT CPU is not itself a failure; required-op no-fallback is an M3 GPU/QNN
  gate.
- Admit the S10 canonical window only after an S25 `PASS` and when the idle S10
  reports at least `1.25 * S25 peak PSS + 512 MiB` as `MemAvailable`; use a
  300-second watchdog and the same PASS/RISK/FAIL rules. This is a resource
  safety ordering, not a claim that an S10 2-second smoke cannot be informative.
- At the source-freeze checkpoint, do not reconstruct audio or run a full song
  in this smoke batch. Later canonical CPU and GPU+CPU experiments intentionally
  used separate full DSP and long-duration contracts; their results are in the
  dated reports above.

### Batch M2: historical protocol; expanded CPU batches complete

- At most one warmup and three measured windows per accepted device/backend.
- Test 2-second and 7.8-second profiles as separate workloads.
- Keep model-only time, output readback, and host DSP time separate.
- Require 3/3 exact-shape finite outputs. Sample retained PSS immediately after
  each measured output readback and keep a separate session peak PSS. Stop and
  mark unstable if the final retained PSS is more than both 10% and 128 MiB
  above the first measured window, thermal status reaches `SEVERE` (3), swap
  crosses the M1 line, or an unexpected fallback occurs under that backend's
  contract. Builtin CPU execution remains subject to the M1 CPU rule.
- Report median, maximum, peak PSS, and model-only real-time factor. Define
  `RTF = inference_seconds / represented_audio_seconds`; RTF below 1.0 is
  faster than real time. With three measurements, do not label the maximum as
  a statistical P95.

### Batch M3: GPU/NPU boundary (historical smoke protocol)

The generated smoke profile has exercised the M3 ordering on S25. Strict GPU
prepare failed after reporting only `158/3504` GPU nodes. Explicit GPU+CPU
OpenCL BUFFER FP32 completed and emitted the `LITERT_CL` evidence line, but its
CPU remainder and output-level parity must be reported with the mixed-backend
label. QNN created a non-empty IR and then failed VTCM schedule allocation;
because graph finalize and `CompiledModel` creation did not complete, no NPU
inference was attempted. The unchanged smoke graph is closed to further NPU
option retries.

- Run only artifacts that pass static conversion and M0B numerical gates with
  the same DSP, padding, and precision revision.
- GPU requires successful allocation, an actual dispatch/event count greater
  than zero, and explicit accounting of CPU fallback; a non-empty partition
  alone is not execution evidence.
- QNN is S25-only and requires provider/library identity, a non-empty IR
  partition, graph execution count greater than zero, finite output, and
  explicit proof that no required op fell back.
- Apply the FP32 `1e-3`/80 dB numerical gate independently. Lower-precision GPU
  or NPU results outside that gate may be retained as throughput-only evidence,
  but must not be described as numerically usable.
- The source-freeze rule was not to run sustained GPU/NPU work until
  single-window memory and delegation were understood. That rule applies only
  to this historical smoke batch. Later canonical reports did run bounded
  S25 GPU+CPU profiles, while no canonical NPU inference was completed; those
  results supersede the then-future wording without changing the smoke result.

For the external-DSP route, freeze and independently test the host reference
before M1: periodic Hann (`n_fft=4096`, `hop=1024`), Demucs reflect padding,
normalized STFT, Nyquist-bin drop/restore, frame crop, iSTFT target length, and
the iSTFT `window^2` envelope normalization. Treat full-song chunk
triangular-edge weighting as a separate overlap-add contract with separate
fixtures. Record separate STFT, core, iSTFT, and overlap-add timings.
Host-vs-PyTorch reference vectors use the same fixture gates as M0B; a failed
DSP gate blocks end-to-end device conclusions.

## Reproduction of the historical source-freeze probes

These commands reproduce the pinned download, community-ONNX structure, and
schema checks from the original baseline. They do not reproduce the later
project-owned canonical exports or device batches; use the dated reports in
the closure index for those identities and runners.

Pinned artifacts are ignored under `models/demucs/`. Download or reverify them:

```powershell
.\tools\prepare_demucs_candidates.ps1
```

Validate structure and run one desktop ORT allocation window per candidate:

```powershell
.\.venv\Scripts\python.exe tools\validate_tensor_only_candidate.py `
  app\src\main\assets\benchmark-contracts\htdemucs_4s_waveform_7p8s_onnx.json `
  --run-zero

.\.venv\Scripts\python.exe tools\validate_tensor_only_candidate.py `
  app\src\main\assets\benchmark-contracts\htdemucs_6s_waveform_7p8s_onnx.json `
  --run-zero
```

Validate both v2 and v3 contract loaders:

```powershell
.\gradlew.bat testStandardDebugUnitTest --no-daemon
```

The interrupted conversion coverage remains ignored under
`.tmp/onnx2tf-demucs6-full/`. It is diagnostic evidence only and must not be
treated as a converted model. The original probe was exploratory and its
one-off shell invocation is not yet packaged as a reproducible script. Before
any retry, copy the pinned input to a new staging directory, record the exact
onnx2tf flags and environment lock there, run with a seven-
minute wall timeout, and verify the original file's SHA-256 is unchanged after
the process. The related `bss-tflite` helper used this pinned invocation after
static-shape preparation:

```powershell
$staging = Resolve-Path .tmp/onnx2tf-demucs6-staging
$output = Resolve-Path .tmp/onnx2tf-demucs6-full
python -m onnx2tf -i (Join-Path $staging 'htdemucs_6s_static.onnx') `
  -o $output -tb flatbuffer_direct -coion -roc -n
```

For Demucs, freeze every input and output dimension in the staging copy before
that command; the helper's batch-only shape rewrite is insufficient. Never
pass `models/demucs/onnx/*.onnx` directly to a converter.
