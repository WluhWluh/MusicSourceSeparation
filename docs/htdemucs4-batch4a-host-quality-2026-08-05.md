# HTDemucs four-stem Batch 4A host quality short test

Date: 2026-08-05

Status: **complete**

## Result

Batch 4A completed on four 30.000-second excerpts and produced the five planned
four-stem renderings. All pre-quantization outputs are finite, all variants use
the exact same selected PCM per track, the official base fresh-load rerun is
bit-exact, and the Kani neural-core conversion-control gate passed.

The completed multi-track blind review found no reliable distinction or
repeatable preference among official base, the official FT bag,
base-plus-vocals, and base-plus-drums. Official base therefore advances as the
only general-purpose four-stem mobile export candidate from this batch. No
isolated stem ground truth is available, so waveform delta and mixture residual
remain diagnostics rather than quality scores.

The Psytrance ONNX graph executes correctly, but listening found clearly worse
quality and substantially more cross-stem leakage. It does not advance:

- no real Psytrance excerpt exists in this corpus;
- the model produces large non-domain changes, including strong drum energy on
  the near-instrumental Athletics control;
- it clips 4,319 samples on Josiah and 4,120 samples on Kygo under the common
  25% OLA contract, with pre-quantization peaks of 3.83 and 3.37;
- its mixture reconstruction SNR is only 8.19-11.21 dB on the four excerpts;
- the publisher-native 50% overlap changes the waveform materially and does not
  remove the clipping problem.

## Scope and contract

| Field | Primary five-way comparison |
|---|---|
| Input | first 30.000 s of the existing fully decoded canonical WAV |
| Sample format | 44.1 kHz, stereo, signed PCM16 input |
| Selected frames | 1,323,000 per track |
| Global normalization | selected-prefix mono reference mean and correction-1 standard deviation |
| Model window | 343,980 samples, 7.8 s |
| Stride | 257,985 samples |
| Overlap | 25%, triangular OLA, transition power 1 |
| Windows | 6 per 30-second excerpt |
| Stem order | drums, bass, other, vocals |
| Post-processing | no residual redistribution, limiting, or per-variant loudness matching |

This selected-prefix normalization is valid for the Batch 4A comparison. These
outputs must not be spliced into the existing full-song runs, whose normalization
was calculated over a different selection.

The Psytrance publisher manifest declares a 171,990-sample hop, or 50% overlap.
The primary comparison deliberately uses the shared 25% contract. A separate
Kygo sensitivity run uses the publisher hop and is excluded from the A-E set.

## Inputs

| Track | Role | Selected PCM SHA-256 |
|---|---|---|
| Athletics - II | near-instrumental/guitar control | `c35318b7d17823bedebe755a65fe636ab3527ef964cbc85eb151717c087869f3` |
| John Lennon - Imagine | piano, vocals, and other instruments | `83ee55a04439ca44035f47599277c1f1a0bdfffebf2fff31777c13daadd22529` |
| Josiah James - Chasing The Wind | dense vocals/drums/bass control | `10f8760bc48dc0a78ba6f595107191a38d24d768206f959dddec5f3794d0a374` |
| Kygo & Ed Sheeran - I See Fire (Kygo Remix) | electronic, explicitly not Psytrance | `64f5dff9968754c09417ec156c69fd16f1b827b3ef93867dd2fb046077a9ea64` |

Kygo can reveal a non-domain regression but cannot establish Psytrance-domain
quality.

## Variants

| Variant | Rendered stem sources | Forwards/window |
|---|---|---:|
| `official-base` | all stems from `955717e8` | 1 |
| `official-ft-bag` | `f7e0c4bc.drums`, `d12395a8.bass`, `92cfc3b6.other`, `04573f0d.vocals` | 4 |
| `base-plus-vocals` | base drums/bass/other plus `04573f0d.vocals` | 2 |
| `base-plus-drums` | `f7e0c4bc.drums` plus base bass/other/vocals | 2 |
| `psytrance-onnx` | all four outputs from the Kani Psytrance neural core | 1 |

Each official specialist is a complete four-output HTDemucs model, but only its
documented target stem has official bag semantics. The other three outputs were
checked for finiteness during inference and discarded before OLA.

## Provenance

| Artifact | Revision | SHA-256 |
|---|---|---|
| official base `955717e8.safetensors` | `adefossez/HTDemucs@bf35a81b663819a8255c8fefee17f9d812b786b5` | `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd` |
| FT drums `f7e0c4bc.safetensors` | `adefossez/HTDemucs-ft@478be8a68f85418addd6f7baefd4be76522a4034` | `2c85ab3c62dd6edd8e0b965e38b16fd1cdde357cc25de6b6bc9ce7c83f60925f` |
| FT bass `d12395a8.safetensors` | same | `5b01a97567ae9a3178a6236fb520251045c03eb8834bc8c24a4eec11d6c8fb56` |
| FT other `92cfc3b6.safetensors` | same | `a241863551f30d01c42bd7b97da40839922ead3acb0f1fcab25682f55b4eeb59` |
| FT vocals `04573f0d.safetensors` | same | `68854b0d7c2b3274723b5761f6fd9f5aec5f1bcd3f0de7c1669546fdb7871b7c` |
| Kani conversion control `htdemucs_ft.onnx` | `Kani95/htdemucs-ft-ort@12cc9a49c3b5f3badc1b0821ccc26f1a32c1779a` | `21cdabc8246f5052397647399e48292dc7394475e92335c65c19eb7e90bde6e0` |
| Kani Psytrance `psytrance_ft.onnx` | `Kani95/htdemucs-psy-ft@e725e7eb9204188de4731658e9923dcf049273c4` | `d7cfcfaf41dc611dd14d35a206fe14a667bdaa6a1910f4c09a8d8ed43606cc78` |

The pinned Demucs loader is revision
`eeac1d15891af95b1288d2884b95baa3e5baa96c`. Every Torch model has 533
tensors and 41,984,456 FP16-stored parameters, loaded as CPU FP32. Embedded
`klass/args/kwargs` metadata matches each pinned JSON sidecar.

Runtime: Python 3.12.3, Torch 2.11.0+cpu, ONNX Runtime 1.21.1, NumPy 2.5.1,
four CPU threads under WSL.

## Conversion-control gate

Before allowing the Psytrance graph into the render set, the author's
`htdemucs_ft.onnx` control was run through the exact same host boundary:

```text
normalized waveform -> official _spec -> ONNX core -> official _ispec
                    -> frequency branch + time branch
```

The control drums output was compared with official `f7e0c4bc` Torch drums on
the first Josiah window.

| Metric | Gate | Result |
|---|---:|---:|
| waveform delta SNR | >= 60 dB | 66.3521 dB |
| maximum absolute delta | <= 1e-4 | 2.4341e-5 |
| cosine similarity | >= 0.99999 | 0.999999886 |

This passes the DSP/layout/axis conversion boundary. It does not prove that the
Psytrance weights are high quality or correctly licensed.

## Host performance

Timings exclude artifact loading and WAV rendering. A variant with two or four
models reports the sum of its independently measured constituent passes.

| Variant | Mean RTF | Seconds per 30 s | Forwards/window |
|---|---:|---:|---:|
| official base | 0.43 | 12.96 | 1 |
| official FT bag | 1.77 | 53.10 | 4 |
| base plus vocals | 0.86 | 25.81 | 2 |
| base plus drums | 0.90 | 26.88 | 2 |
| Psytrance ONNX plus host DSP | 0.40 | 12.10 | 1 |

Single Torch model track RTF ranged from 0.416 to 0.489. Model load time ranged
from 4.80 to 6.27 seconds. Psytrance neural-core time averaged 1,942.2 ms per
window; host STFT and iSTFT/combine averaged 5.3 and 45.4 ms.

Maximum observed resident memory was about 728-784 MiB during the sequential
Torch arms and 4,772 MiB during the Psytrance arm. The process high-water mark
was 4,900 MiB. These are shared-process observations after the conversion
control and earlier arms, not clean cold-start per-model peaks.

## Mixture coherence

The values below are `mix` versus the raw sum of four float stems. Higher is a
closer reconstruction, but this is not a quality veto and is not SDR.

| Track | Base | FT bag | +vocals | +drums | Psy ONNX |
|---|---:|---:|---:|---:|---:|
| Athletics | 35.64 | 36.04 | 35.43 | 35.12 | 8.80 |
| Imagine | 38.66 | 29.20 | 28.44 | 38.30 | 10.06 |
| Josiah | 33.46 | 26.19 | 26.31 | 29.57 | 11.21 |
| Kygo | 32.08 | 16.54 | 31.93 | 23.95 | 8.19 |

The hybrid residual identities were checked exactly for every track:

```text
r(base+vocals) - r(base) = -(vocals_ft - vocals_base)
r(base+drums)  - r(base) = -(drums_ft  - drums_base)
```

All eight maximum identity errors are zero. Residual changes therefore do not
constitute independent leakage evidence.

## Waveform differences

These ranges compare each rendered stem with official base over the four
tracks. They describe change, not quality.

| Variant | Stem | Delta SNR range | RMS change range | Minimum correlation |
|---|---|---:|---:|---:|
| FT bag | drums | 0.94 to 19.87 dB | -4.50 to -0.08 dB | 0.41 |
| FT bag | bass | 1.44 to 22.67 dB | -5.65 to -0.02 dB | 0.42 |
| FT bag | other | 12.83 to 39.10 dB | -0.30 to +0.05 dB | 0.97 |
| FT bag | vocals | -0.21 to 22.80 dB | -10.80 to +0.17 dB | 0.06 |
| Psy ONNX | drums | -50.21 to -2.97 dB | +5.50 to +50.21 dB | 0.07 |
| Psy ONNX | bass | -34.76 to 11.75 dB | +0.38 to +34.74 dB | -0.07 |
| Psy ONNX | other | 3.46 to 10.65 dB | +0.08 to +1.22 dB | 0.81 |
| Psy ONNX | vocals | -35.82 to 13.76 dB | +0.17 to +35.82 dB | 0.02 |

The hybrid rows are identical to base for their three unchanged stems and to
the corresponding specialist row for the changed stem. Very low-signal base
stems can produce extreme delta ratios; that alone is not a quality failure.
However, the Psytrance output also moves material energy: on Athletics, drums
rise from effectively zero in base to 27.83% of rendered-stem energy. Because
the conversion-control gate passed, this is evidence about the candidate
weights/behavior rather than a host ABI explanation.

## Finiteness, clipping, and seams

- Every raw and assembled float output is finite.
- Athletics has no pre-quantization clipping in any primary variant.
- Imagine has six clipped Psytrance drum samples; all official variants have none.
- Josiah base has 82 clipped drum samples, FT drums has 18, and Psytrance has
  4,300 clipped drum plus 19 clipped other samples. The Psytrance drum peak is 3.83.
- Kygo Psytrance has 4,095 clipped drum plus 25 clipped other samples. Its drum
  peak is 3.37. Official variants have no clipping on this excerpt.
- Maximum OLA-boundary jump divided by local derivative P95 is 1.67 across all
  primary outputs. This is recorded for listening review, not treated as an
  automatic seam failure.

## Psytrance native-hop sensitivity

On the Kygo non-Psytrance control, changing only song-level OLA from 25% to the
publisher-declared 50% gives:

| Metric | 25% primary | 50% publisher hop |
|---|---:|---:|
| RTF | 0.410 | 0.587 |
| mixture reconstruction SNR | 8.19 dB | 8.82 dB |
| clipped samples | 4,120 | 3,542 |
| peak | 3.37 | 2.94 |
| maximum seam ratio | 0.89 | 0.82 |

The 50% output is materially different from 25%: per-stem delta SNR is 6.69 dB
for drums, 18.26 for bass, 9.51 for other, and 3.69 for vocals. Any future
Psytrance experiment should therefore retain its publisher-hop contract and
remain outside direct waveform ranking against the shared 25% group.

## Blind set

The primary blind set contains exactly 80 FLAC files:

```text
outputs/htdemucs4-batch4a-host-20260805/blind-ae/<track>/<A-E>/<stem>.flac
```

Each track has an independent deterministic HMAC-based A-E permutation; the
same letter identifies one coherent four-stem variant within that track. Every
FLAC was decoded before and after publication and matched to its source WAV PCM
SHA. Inventory, no-symlink, sample-rate, channel-count, PCM16, and frame-count
gates passed.

Per the explicit storage decision for this batch, the 80 blind files are hard
links to private reference FLACs. `samefile` was verified for all 80. This saves
disk space, but a listener with filesystem-level hard-link enumeration can
discover the reference path and defeat blinding; the ordinary A-E directory
names themselves contain no variant identity.

Private key and evidence, outside the blind root:

- `docs/htdemucs4-batch4a-blind-map-2026-08-05.json`
- `docs/htdemucs4-batch4a-blind-map-2026-08-05.md`
- `docs/htdemucs4-batch4a-audio-evidence-2026-08-05.json`

Do not open the mapping documents before completing the listening sheet.

## Blinded listening outcome

After the multi-track blind review was completed and the assignments were
unblinded, the listener reported that `psytrance-onnx` was clearly worse than
the other four variants, with substantially more audible cross-stem leakage.
This agrees with its large non-domain energy transfers, clipping, and weak
mixture coherence, while remaining an independent perceptual observation.

Differences among official base, the full FT bag, base-plus-vocals, and
base-plus-drums were subtle. Across multiple tracks, the listener could not
reliably distinguish these four variants or establish a repeatable preference.
This is evidence of no demonstrated perceptual advantage on the present four
excerpts, not proof that the variants are perceptually equivalent.

## Evidence identities

| Evidence | SHA-256 |
|---|---|
| batch report | `f42a7a8b251b07f44e952a25fb0ef0aeecb292f10cf0da038e6046db864ec1e8` |
| conversion-control report | `9a24fe8320c76093078c86d4e1868b18ba17fd50f296cdb784c15acf2db7e3f8` |
| determinism report | `ac276d5a65ffe7b85e863bb894159d287039e15dadf2f646f8fe7d7ca69b54be` |
| private blind map JSON | `282199fdd15a74821634159cfbdc9aa6df4ff3cb605a5dc7807727ffb51eaf08` |
| private audio evidence JSON | `a0e27d1427739864d917e8365e360ff7db1bda822655a9f0f508656cbe1575bb` |
| Batch 4A runner | `3722c5e587e71ebc41b04e3a64585ac918fd512a3d8fc97468407b1f45a486d7` |
| blind finalizer | `cf03b0a66abf8291b9d2d0eab09144d227269d5e11f8b1582715e7354dee149e` |

The official-base fresh-load 30-second rerun changed zero samples and is
bit-exact.

## Batch 4A decision

1. Advance official base as the sole general-purpose four-stem mobile export
   candidate from this batch. No repeatable listening preference over it was
   demonstrated, and it requires only one inference pass.
2. Retain the full FT bag as a host quality-ceiling reference, but do not spend
   mobile conversion or S25 time on it now. Its host cost is about 4.1 times
   base, it is slower than real time, and no repeatable listening benefit was
   demonstrated.
3. Do not advance base-plus-vocals or base-plus-drums. Each requires two passes,
   yet neither produced a repeatable preference over official base. Reopen a
   hybrid only if a future targeted corpus exposes a consistent target-stem
   weakness in base.
4. Reject `psytrance-onnx` for the general-purpose four-stem route: the blind
   review found clearly worse quality and more cross-stem leakage, consistent
   with the objective diagnostics. A future domain-specific Psytrance study
   would be a separate research batch requiring a pinned, licensed real
   Psytrance excerpt and the publisher 50% hop; it must not inherit advancement
   from this experiment.

No S25 or LiteRT inference is claimed by this report. Batch 4A is the host
quality-semantics gate that precedes any four-stem mobile export contract.
