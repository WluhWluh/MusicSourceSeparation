# HTDemucs-6s guitar-ft LiteRT 2.1.5 S25 diagnostic

Date: 2026-08-05

## Decision

The guitar-ft neural core was exported successfully and is usable for bounded
research diagnostics on S25 CPU. It is not admitted as a normal host or device
candidate.

The strict host gate remains failed. Single-window parity and full canonical
OLA parity pass, but the canonical EOF region reaches only 79.245 dB against
the required 80 dB. The S25 runs below were therefore executed under an
explicit `diagnosticOnly`, `researchOnly`, `hostAdmissionStatus=not-admitted`
variant. They do not relax or override the frozen gate.

## Artifact identity

```text
modelId: htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0
file:    htdemucs_6s_guitar_ft.core.canonical_7p8s.fp32.tflite
bytes:   117,729,544
SHA-256: ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5
```

Export environment:

```text
Python:          3.12.3
Torch:           2.11.0+cpu
litert-torch:    0.9.1
AI Edge LiteRT:  2.1.5
NumPy:           2.5.1
safetensors:     0.8.0
```

The FlatBuffer is TFL3 schema 3 with one subgraph, 3,432 operators, 4,176
tensors, two static FP32 inputs, two static FP32 outputs, and no custom
operators. Its declared minimum runtime is 2.16.0. The ABI and host DSP
contract are inherited from the official canonical 343,980-sample candidate.

Frozen evidence:

| Evidence | Bytes | SHA-256 |
|---|---:|---|
| Export report | 71,910 | `482a710f640db368bf335966842f8c026b0c262d2061605b9d4098bd67ffa1ec` |
| FlatBuffer inspection | 3,864 | `5d38e71bdd10337ee14cfe2fa9975ecdfa8924e38e65e10513db81c4dda95c22` |
| Diagnostic manifest | 79,911 | `b8b0026e8f416c080019035d3b2a18ddbc5ca9e0f9aa4611f8c42c9cfac4012a` |

The diagnostic manifest kind is `generated-litert-diagnostic-candidate`, not
`generated-litert-host-candidate`. Product use remains unapproved because the
MoisesDB training-data terms require an independent license review.

## Host parity

The parity reference is the same guitar-ft Torch FP32 model. The official
HTDemucs-6s output is intentionally not an admission reference for this
fine-tuned weight set.

| Scope | SNR | Maximum absolute error | Result |
|---|---:|---:|---|
| Torch neural core vs guitar-ft full model | bitwise equal | 0 | pass |
| LiteRT vs Torch, deterministic single-window combined output | 108.591 dB | `2.593e-6` | pass |
| LiteRT vs Torch, complete canonical OLA | 85.416 dB | `2.135e-5` | pass |
| LiteRT vs Torch, overlap region | 91.823 dB | `9.999e-6` | pass |
| LiteRT vs Torch, EOF region | 79.245 dB | `2.135e-5` | **fail** |
| LiteRT vs Torch, EOF guitar stem | 79.170 dB | `2.135e-5` | **fail** |

The artifact and metrics reproduced exactly with `--reuse-litert`. This makes
the EOF miss deterministic rather than a one-off conversion result.

## S25 CPU batch

Device and runtime:

```text
device:       Samsung SM-S9310, Snapdragon SM8750, Android API 35
runtime:      com.google.ai.edge.litert:litert:2.1.5
accelerator:  CPU
threads:      4
window:       343,980 samples / 7.8 seconds
overlap:      25 percent
selection:    first 30.0 seconds / 1,323,000 frames
```

Each run verified the source MP3 SHA, the selected PCM SHA against the existing
S25 canonical WAV, the model size and SHA on device, six output WAV contracts,
and all output statistics before removing the remote per-track evidence.

| Role | Track | E2E RTF | Mean core inference | Peak PSS | Thermal |
|---|---|---:|---:|---:|---:|
| guitar-heavy | Athletics - II | 0.6745 | 2,119.8 ms | 1,128,795 KiB | 0 |
| piano-heavy | John Lennon - Imagine | 0.6955 | 2,160.4 ms | 1,128,369 KiB | 0 |
| vocals/drums-sensitive | Josiah James - Chasing The Wind | 0.6648 | 2,149.2 ms | 1,128,438 KiB | 0 |

Aggregate results across 18 windows:

```text
mean 30-second E2E RTF:                  0.6783
mean prepare per session:               483.7 ms
  model compile:                         192.9 ms
  tensor buffer allocation:             253.1 ms
mean first-window total:               4,096.7 ms
mean first-window core inference:      2,205.7 ms
mean warm core inference:              2,130.6 ms
warm core inference P50 / P95:         2,121.8 / 2,201.9 ms
mean warm complete window:             3,071.2 ms
warm complete window P50 / P95:        3,113.1 / 3,521.8 ms
mean warm STFT / iSTFT:                   50.7 / 456.3 ms
mean warm PCM conversion / OLA:          335.4 / 23.7 ms
peak native heap allocated:            971,717,696 bytes
non-finite output samples:                        0
clipped output samples:                           0
maximum Android thermal status:                   0
```

The matched official model reports for the first six windows of the same three
tracks average 2,130.7 ms core inference. Guitar-ft averages 2,143.1 ms, a
0.59 percent difference that is within run-to-run noise for the identical
architecture.

## S25 vs same-weight Torch

These comparisons use PCM16 outputs generated from the same selected 30-second
PCM. The intentional guitar-ft weight change is also shown to establish scale.

| Track | S25 guitar-ft vs guitar-ft Torch | Guitar-ft Torch vs official Torch | Maximum S25 error |
|---|---:|---:|---:|
| Athletics - II | 87.690 dB | 28.339 dB | 1 LSB |
| John Lennon - Imagine | 82.853 dB | 7.320 dB | 1 LSB |
| Josiah James - Chasing The Wind | 84.360 dB | 9.078 dB | 2 LSB |

The S25/Torch correlations are respectively `0.9999999992`, `0.9999999974`,
and `0.9999999982`. Low-energy stems can report lower SNR despite a 1 LSB
maximum difference. For example, the conservative Imagine vocals/drums screen
fails because the drum RMS lies just above the low-signal threshold and its
63.42 dB SNR is below 80 dB; the same stem still has maximum error 1 LSB and
correlation `0.999999742`.

The device PCM16 result is strong evidence that the S25 CPU path preserves the
guitar-ft output on these songs. It does not replace the failed host float EOF
gate because the two gates exercise different representations and regions.

## Leakage attribution

There is no isolated ground truth for these songs. The following results are
fixed-mask cross-stem proxies, not SDR or a direct perceptual quality score.

### Piano vocals in Imagine

The user's observation that guitar-ft removes most audible vocals from the
Imagine piano stem is directionally supported by the fixed official-vocal
comparison:

```text
piano vs fixed vocal, active spectral cosine:  0.235986 -> 0.173614  (-26.4 percent)
piano vs fixed vocal, active waveform cosine:  0.103783 -> 0.050331  (-51.5 percent)
piano / fixed-vocal active power:              3.417 dB -> -0.164 dB (-3.581 dB)
removed piano content vs official vocal,
  active waveform cosine:                      0.197913
```

The S25 piano stem matches guitar-ft Torch at 84.011 dB, correlation
`0.9999999980`, and maximum error 1 LSB. The S25 conversion piano delta on
vocal-active frames is 77.507 dB below the fine-tune weight delta. Attribution
is therefore `weight-change-dominant`, and the proxy is
`directionally-consistent-with-reduced-vocal-leakage`.

### Drums in vocals

The proxy is content-dependent. Guitar-ft raises the vocal-to-drum transient
2-8 kHz power ratio by 3.38 dB on Athletics and 3.17 dB on Josiah, while it
lowers the ratio by 13.79 dB on Imagine. The increases on Athletics and Josiah
are directionally consistent with the reported slight drum contamination in
vocals, but musical coincidence remains a confounder.

S25 reproduces each guitar-ft Torch ratio within 0.00016 dB. The S25 conversion
vocal-delta power is 47.9 to 65.7 dB below the weight-change vocal delta on the
fixed drum-transient frames. The audible vocal change is therefore attributable
to the fine-tuned weights, not to the S25 LiteRT conversion, within the limits
of these no-ground-truth proxies.

## Blind listening material

The actual S25 LiteRT CPU output and same-weight guitar-ft Torch output are
available as a separate three-track lossless A/B set:

```text
outputs/htdemucs6-guitar-ft-s25-20260805/blind-litert-vs-torch/
```

Each track has randomized `A` and `B` folders containing all six stems. The key
is intentionally outside the blind directory:

```text
docs/htdemucs6-guitar-ft-litert-s25-blind-map-2026-08-05.json
docs/htdemucs6-guitar-ft-litert-s25-blind-map-2026-08-05.md
```

The lossless reference manifest SHA-256 is
`cef3918c8fba5c9f9939e0614259fc519ada3e7511c55ed8a098eed9a2a6baf1`.

## Evidence layout

```text
models/demucs/generated/htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0/
  diagnostic-manifest.json
  export-report.json
  flatbuffer-inspection.json
  fixtures/
  htdemucs_6s_guitar_ft.core.canonical_7p8s.fp32.tflite

outputs/htdemucs6-guitar-ft-s25-20260805/
  batch-progress.json
  device.json
  host-reference-progress.json
  blind-reference-manifest.json
  tracks/<slug>/
    official-torch-30s/
    guitar-ft-torch-30s/
    s25-cpu-30s/
    three-way-waveform-comparison.json
    leakage-analysis.json
    leakage-analysis.md
```

## Reproduction

The export entry point is
`tools/export_htdemucs_guitar_ft_litert_candidate.py`. The explicit failed-gate
manifest is reproduced with:

```text
python tools/freeze_htdemucs_guitar_ft_litert_candidate.py \
  --allow-narrow-host-failure --check
```

The bounded device and reference batches are reproduced with:

```text
python tools/run_htdemucs_guitar_ft_s25_batch.py \
  --serial "$DEVICE_SERIAL" --source-root "$MP3_CORPUS"
python tools/run_htdemucs_guitar_ft_30s_reference_batch.py
```

The Android benchmark defaults to the official model. Guitar-ft is reachable
only through the whitelisted instrumentation argument:

```text
-e canonicalE2eModelVariant guitar-ft
```
