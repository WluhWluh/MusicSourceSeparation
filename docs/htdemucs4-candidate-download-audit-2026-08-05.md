# Four-stem HTDemucs candidate download and initial audit

Date: 2026-08-05

## Scope

This batch searched for and downloaded HTDemucs weights or graphs that could
plausibly produce the ordered four-stem ABI:

```text
drums, bass, other, vocals
```

It performs source pinning, artifact hashing, static architecture inspection,
weight-identity comparison, and one fixed-window ONNX CPU smoke. It does not
perform S25 execution, full-song separation, ground-truth quality evaluation,
or product admission.

## Outcome

Only two downloaded groups retain a four-stem test path:

1. **P0: official `adefossez/HTDemucs-ft` bag.** This is the authoritative
   quality candidate. It consists of four full HTDemucs models and requires
   four forwards for the published bag semantics.
2. **P2 research-only: `Kani95/htdemucs-psy-ft`.** Its ONNX ABI really exposes
   four stems and its weights are materially different, but it has no published
   training configuration, dataset, quality metric, raw checkpoint, exporter,
   or Torch-to-ONNX parity evidence.

Two apparent candidates were downloaded and then rejected from the four-stem
matrix:

- `pablebe/htdemucs` is a 48 kHz vocals-only HTDemucs.
- Both `raddhuha/HT-Demucs` files are undocumented one-stem HTDemucs states.

Conversion-only mirrors of official weights were not counted as new quality
candidates. Examples include GGUF/MLX/CoreML/WebGPU variants and the single
`04573f0d` repack described as `htdemucs_ft.safetensors` by AEmotionStudio.

## Downloaded artifacts

The new candidate and conversion-reference directories occupy 2,535,825,593
bytes (2.362 GiB), including small source cards and download metadata.

| Group | Pinned revision | Main artifact(s) | Bytes | SHA-256 | Decision |
| --- | --- | --- | ---: | --- | --- |
| Official FT drums | `adefossez/HTDemucs-ft@478be8a68f85418addd6f7baefd4be76522a4034` | `f7e0c4bc.safetensors` | 84,025,440 | `2c85ab3c62dd6edd8e0b965e38b16fd1cdde357cc25de6b6bc9ce7c83f60925f` | P0 bag member |
| Official FT bass | same | `d12395a8.safetensors` | 84,025,440 | `5b01a97567ae9a3178a6236fb520251045c03eb8834bc8c24a4eec11d6c8fb56` | P0 bag member |
| Official FT other | same | `92cfc3b6.safetensors` | 84,025,440 | `a241863551f30d01c42bd7b97da40839922ead3acb0f1fcab25682f55b4eeb59` | P0 bag member |
| Official FT vocals | same | `04573f0d.safetensors` | 84,025,440 | `68854b0d7c2b3274723b5761f6fd9f5aec5f1bcd3f0de7c1669546fdb7871b7c` | P0 bag member |
| Psytrance | `Kani95/htdemucs-psy-ft@e725e7eb9204188de4731658e9923dcf049273c4` | `psytrance_ft.onnx` | 174,266,467 | `d7cfcfaf41dc611dd14d35a206fe14a667bdaa6a1910f4c09a8d8ed43606cc78` | P2 research-only |
| Kani conversion control | `Kani95/htdemucs-ft-ort@12cc9a49c3b5f3badc1b0821ccc26f1a32c1779a` | `htdemucs_ft.onnx` | 174,266,467 | `21cdabc8246f5052397647399e48292dc7394475e92335c65c19eb7e90bde6e0` | Conversion control only |
| WASPAA 2025 | `pablebe/htdemucs@6a6d4df0e334c263ff9d820005db808239e70974` | `htdemucs_epoch=570-sdr=6.38.ckpt` | 1,635,710,219 | `30d165a32b0a2238210f818c06b4b9f9da0198f2c734a95075ee21077cd596b8` | Reject: one stem |
| Undocumented state A | `raddhuha/HT-Demucs@ebb50b725c6b2a3b408b75e5430ff9a056ebeb2c` | `htdemucs_finetuned.pt` | 107,715,779 | `1bfc71644adba7d2065ec8ab1bb8d71d4e68d2fb15ed2c4d284e816b704dc698` | Reject: one stem |
| Undocumented state B | same | `htdemucs_finetuned (1).pt` | 107,715,779 | `f9150cd79333a8e8539defbd74073793e9ac85d0851de090371cbf243e4e4028` | Reject: one stem |

Local roots:

- `models/demucs/candidates/htdemucs-ft/478be8a68f85418addd6f7baefd4be76522a4034/`
- `models/demucs/candidates/htdemucs-psy-ft/e725e7eb9204188de4731658e9923dcf049273c4/`
- `models/demucs/candidates/htdemucs-waspaa2025/6a6d4df0e334c263ff9d820005db808239e70974/`
- `models/demucs/candidates/htdemucs-raddhuha-ft/ebb50b725c6b2a3b408b75e5430ff9a056ebeb2c/`
- `models/demucs/conversion/htdemucs-ft-ort/12cc9a49c3b5f3badc1b0821ccc26f1a32c1779a/`

## Official HTDemucs-ft bag

The official bag manifest fixes this assembly:

| Output stem | Model | Training loss weights | Stored state |
| --- | --- | --- | ---: |
| drums | `f7e0c4bc` | `[1,0,0,0]` | 533 tensors / 41,984,456 FP16 values |
| bass | `d12395a8` | `[0,1,0,0]` | 533 tensors / 41,984,456 FP16 values |
| other | `92cfc3b6` | `[0,0,1,0]` | 533 tensors / 41,984,456 FP16 values |
| vocals | `04573f0d` | `[0,0,0,1]` | 533 tensors / 41,984,456 FP16 values |

All four have the same key set and tensor shapes as the existing official
`955717e8` four-stem base model. Every member therefore fits the same 44.1 kHz,
stereo, exact 7.8-second neural-core architecture. All 533 tensors and more than
41.89 million values differ from the base for each specialist, so none is a
renamed duplicate.

The one-hot weights are a semantic constraint, not just an ensemble preference.
Only the target stem from each specialist is part of the published bag. The
other three outputs from a specialist must not be treated as a validated cheap
four-stem model.

The official paper reports MUSDB-HQ SDR of 9.00 dB overall, with 10.08 drums,
10.39 bass, 6.32 other, and 9.20 vocals for the published per-source fine-tuned
bag. Its reported 9.20 dB overall sparse/per-source system is a separate,
unpublished configuration and must not be attributed to these artifacts.

The repository declares MIT. Training metadata records MUSDB-HQ and additional
non-public internal material. The artifact provenance is authoritative, while
normal product training-data due diligence remains separate from this device
experiment.

Sources:

- <https://huggingface.co/adefossez/HTDemucs-ft/tree/478be8a68f85418addd6f7baefd4be76522a4034>
- <https://github.com/adefossez/demucs/blob/eeac1d15891af95b1288d2884b95baa3e5baa96c/docs/training.md#model-zoo>
- <https://ar5iv.labs.arxiv.org/html/2211.08553>

## Psytrance ONNX

The candidate has a useful neural-core boundary:

- opset 18, 3,291 nodes, 537 float32 initializers;
- 41,984,456 initializer values;
- inputs `input [1,2,T]` and `x [1,4,2048,336]`;
- outputs `output [1,4,4,2048,T_frames]` and
  `add_67 [1,4,2,T]`;
- manifest-fixed `T=343980`, 44.1 kHz, four ordered stems.

The graph uses a dynamic waveform axis even though the publisher manifest fixes
7.8 seconds. A LiteRT conversion must freeze both waveform and frame axes in
the derived contract.

Against the author's `htdemucs_ft.onnx` control:

- every graph node is byte-identical;
- initializer names, shapes, and dtypes are identical;
- 393/537 initializers and 41,473,591/41,984,456 values changed;
- parameter RMS delta is 0.026055 and parameter SNR is 16.833 dB.

The control is not the complete official FT bag. All 322 directly named tensors
(14,653,352 values) shared with official safetensors are bitwise equal to the
`f7e0c4bc` drums specialist. This makes the Psytrance artifact a real weight
mutation of a drums-specialist-shaped reference, but does not establish that
all four emitted stems were optimized or quality-tested.

The publisher declares Apache-2.0 but provides no training data, training code,
configuration, original Torch weights, export code, metrics, or parity report.
The candidate remains `research-only / unverified-provenance` until those gaps
are resolved.

Source:

- <https://huggingface.co/Kani95/htdemucs-psy-ft/tree/e725e7eb9204188de4731658e9923dcf049273c4>

## ONNX host smoke

ONNX Runtime 1.21.1 CPU, four threads, one warmup and two timed executions used
the existing canonical HTDemucs neural-core fixture. This is a load/ABI/finite
smoke only; two timings are not a performance benchmark.

| Model | Prepare | Timed runs | Mean | Finite frequency | Finite time |
| --- | ---: | --- | ---: | ---: | ---: |
| Kani FT control | 1,203.7 ms | 2,716.8 / 2,047.4 ms | 2,382.1 ms | 11,010,048 / 11,010,048 | 2,751,840 / 2,751,840 |
| Psytrance | 1,039.2 ms | 2,568.6 / 1,894.1 ms | 2,231.4 ms | 11,010,048 / 11,010,048 | 2,751,840 / 2,751,840 |

Both outputs have the expected shape and contain no NaN or infinity. The output
difference is intentionally large because the weights differ; it is not a
conversion parity measurement.

## Rejected downloads

### WASPAA 2025

The checkpoint was inspected with `torch.load(weights_only=True, mmap=True)`
under `FakeTensorMode`. Its one custom pickle global was replaced by an inert,
explicitly allowlisted metadata stand-in; no publisher code was imported or
executed.

It contains a 533-tensor, 81,739,446-value FP32 inference state. Checkpoint
hyperparameters explicitly state:

```text
backbone=htdemucs
sources=['vocals']
target_str=vocals
sample_rate=48000
bottom_channels=768
```

The frequency and time output heads contain 4 and 2 channels, respectively,
which is exactly one stereo CaC stem. It is a credible large vocals-only
research workload, but it is not compatible with the four-stem contract. The
paper uses MUSDB18-HQ and MoisesDB; MoisesDB terms keep this path research-only.

Sources:

- <https://huggingface.co/pablebe/htdemucs/tree/6a6d4df0e334c263ff9d820005db808239e70974>
- <https://arxiv.org/abs/2507.11427>
- <https://github.com/pablebe/gensvs_eval/tree/cdcef6502a7ea802f37397450845e3ae1150a769>

### raddhuha

Both artifacts are tensor-only `OrderedDict` ZIPs, safely readable with
`weights_only=True`. Each has 381 FP32 tensors and 26,892,606 values. Their
4-channel frequency and 2-channel time heads again prove a single stereo stem.
The associated application constructs `HTDemucs(sources=['vocals'])`.

The two files are not duplicates: 380/381 tensors and 26,891,801 values differ,
with parameter SNR -0.815 dB. However, there is no training description,
dataset, metric, stem metadata, sample-rate contract, or usable license file.
They are excluded rather than assigned a speculative stem contract.

Sources:

- <https://huggingface.co/raddhuha/HT-Demucs/tree/ebb50b725c6b2a3b408b75e5430ff9a056ebeb2c>
- <https://github.com/raddhuha/model/blob/6dff60294811db201f3f7ba6d3b48474caf31cba/app.py>

## Batch closure

The plan below is retained as the decision record from candidate discovery. Its
execution status is now:

| Batch | Status | Outcome |
| --- | --- | --- |
| 4A host semantics | complete | Official base, FT bag, and two hybrids were not reliably distinguishable; Psytrance was clearly worse with more cross-stem leakage. See [`htdemucs4-batch4a-host-quality-2026-08-05.md`](htdemucs4-batch4a-host-quality-2026-08-05.md). |
| 4B LiteRT export | base complete; specialists cancelled | Official base passed the same-weight host gate. No specialist conversion was justified after 4A showed no repeatable listening benefit. |
| 4C S25 CPU | base-only complete | Three 30-second official-base runs completed; the unused multi-pass specialist workloads were not run. See [`android-litert215-demucs4-official-s25-2026-08-05.md`](android-litert215-demucs4-official-s25-2026-08-05.md). |
| 4D Psytrance | rejected before conversion | General-purpose blind listening found obvious quality and leakage regressions. A future in-domain study would require a separate licensed Psytrance corpus and the publisher's 50% hop. |

## Historical recommended batches

### Batch 4A: host quality semantics

Use the official `955717e8` base as the control and retain the exact 7.8-second
44.1 kHz DSP/OLA contract already used by this repository.

Run short, fixed excerpts through:

1. official base, one forward;
2. each official FT specialist, scoring/listening only to its target stem;
3. the official full FT bag, four forwards;
4. experimental two-pass hybrids: base plus vocals specialist, and base plus
   drums specialist;
5. Psytrance ONNX as a separate research arm, including actual psytrance and
   non-psytrance controls.

Record per-stem waveform delta, mixture residual, finite/clipping counts, OLA
boundary error, and blind-listening outputs. A hybrid must report its mixture
residual before any residual redistribution is introduced.

Do not convert Psytrance first. Its four output stems must first sound coherent
and its mixture residual must be acceptable; the model card provides no evidence
for either property.

### Batch 4B: official LiteRT export and host parity

Generalize the existing official-safetensors neural-core exporter so model
metadata, stem count, and weight identity are parameters rather than six-stem
constants. Freeze a new four-stem contract with:

- float32 waveform and spectrum inputs;
- float32 frequency and time outputs;
- exact 343,980-sample window;
- ordered stems `drums,bass,other,vocals`;
- source Torch, converter, LiteRT 2.1.5, artifact, fixture, and tool SHA values.

Export the base first, then one specialist. The vocals specialist is the best
first quality experiment; the drums specialist is additionally useful because
it is the identified source of the Kani ONNX control. Compare each LiteRT model
against Torch with the same weights. Preserve the current strict numerical
gate; do not compare a specialist artifact to base-model output.

Only after one specialist passes should the other three be exported. All four
share topology, so four early conversions add little performance information.

### Batch 4C: S25 CPU short batch

After host parity, run three 30-second excerpts and measure these workloads
separately:

- base: one forward;
- single specialist: one forward;
- base plus one specialist: two forwards;
- official full bag: four sequential forwards.

Measure prepare, STFT, inference, iSTFT, OLA, peak PSS, native memory, thermal
status, and output finiteness. Load specialist models sequentially for the full
bag rather than retaining four interpreters unless memory testing explicitly
requires the latter. This batch is a performance-bound experiment, not a
product-support commitment.

### Batch 4D: Psytrance conversion only after quality evidence

If Batch 4A finds a repeatable benefit, freeze its dynamic axes and try the
two-input neural-core ONNX path. The graph has only four `ScatterND` operators
and no STFT/iSTFT, so it is a materially better conversion starting point than
a complete waveform graph. Its numerical oracle is the pinned ONNX artifact,
not an unavailable Torch checkpoint. Keep all resulting artifacts
`research-only / unverified-provenance`.

GPU testing can follow CPU parity. QNN/NPU is outside this batch.

## Reproducible evidence

- Full static audit:
  `models/demucs/candidates/htdemucs4-candidate-audit-20260805.json`
- ONNX smoke:
  `models/demucs/candidates/htdemucs4-onnx-smoke-20260805.json`
- Static audit tool:
  `tools/analyze_htdemucs_4s_candidates.py`
- ONNX smoke tool:
  `tools/smoke_htdemucs_4s_onnx_candidates.py`
