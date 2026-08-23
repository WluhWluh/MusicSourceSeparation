# MTG-Jamendo/FMA Expansion Evaluation

Status: completed locally on 2026-08-23. This is a non-commercial research
evaluation. The 16 source recordings, decoded audio, Inst 3 caches, model
outputs, and retained listening set remain local artifacts and must not be
uploaded to `bss-tflite`.

## Objective

The existing 12-song private listening set showed that the current
`H50-continuation+5` model can still leave short, conspicuous vocal-like
residue. This evaluation adds a more diverse set of public-license tracks
from MTG-Jamendo and FMA, compares the model with native Inst 3 full-song
output, and retains the 12 new tracks with the strongest local leak signal for
focused listening.

The additional tracks were selected for genre, language, vocal texture, and
artist diversity. They were not added to training in this run. No official
MUSDB18 test track was used.

## Frozen evaluation

- Candidate pool: 16 full songs, 8 MTG-Jamendo and 8 FMA records.
- Model: `H50-continuation+5`, checkpoint `step-1600.pt`.
- Checkpoint SHA-256: `cc0ce960e2d305cc0f03e69fc58941988be720d69fdd192f6b746b692959a674`.
- Initialization: `vocals_epoch=891.ckpt`, SHA-256
  `101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d`.
- Assembly: 128-frame continuous-context overlap-save; isolated zero-filled
  windows were not used.
- Teacher: `uvr_mdxnet_inst_3@2`, native full-song instrumental output.
- Device: CUDA, with CPU fallback provider available to the teacher.
- Outputs: model instrumental (`mixture - residual`), model residual, and
  native Inst 3 instrumental, all decoded/rendered at 44.1 kHz stereo.

The evaluation does not claim that Inst 3 is a ground-truth stem. It treats
Inst 3 as the practical aggressive-vocal-removal reference requested for this
experiment.

## Ranking metric

For each song, the model output is compared with the Inst 3 instrumental and
the removed-content proxy `mixture - Inst3`. Short blocks of 50, 100, and 200
ms are scanned across the full song. The primary local severity proxy is:

```text
severity = 0.65 * max(positive-projection-max at 50/100/200 ms)
         + 0.35 * mean(positive-projection-p95 at 50/100/200 ms)
```

Higher (less negative) dBFS means a stronger coherent component that remains
in the student accompaniment while Inst 3 removes it. Raw miss RMS and counts
above fixed dBFS thresholds are also retained in the JSON report, but are not
used alone for ranking because they can include content that is not clearly
vocal-like.

## New-pool ranking

| Rank | Artist - track | Source | Severity dBFS | 50 ms max | 100 ms max | 200 ms max | License |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Chico Correa Pocket Band - Rebuild Jampa Sessions | FMA | -19.13 | -14.90 | -15.72 | -17.46 | BY-NC-SA 3.0 International |
| 2 | Lee Maddeford, Roland Vouilloz - Free Me | FMA | -19.55 | -14.47 | -15.71 | -16.97 | BY-NC-SA 3.0 International |
| 3 | Hidden Tribe - Из дверей в двери | MTG-Jamendo | -22.18 | -18.73 | -19.89 | -20.72 | CC BY-SA |
| 4 | Stray - Akuma | FMA | -22.54 | -16.43 | -19.32 | -23.17 | BY-NC-SA |
| 5 | Ceilí Moss - The Praties Go Small | MTG-Jamendo | -22.62 | -19.14 | -21.28 | -21.35 | CC BY-SA |
| 6 | Löhstana David - La rupture | MTG-Jamendo | -25.07 | -21.80 | -23.80 | -25.66 | CC BY-SA |
| 7 | Los Cuatrocientos Golpes - La Horda | FMA | -25.48 | -21.86 | -22.74 | -23.43 | BY-NC-SA 3.0 International |
| 8 | Dr. Emiliyan Stankov - We Hymn Thee, We Bless Thee (anonymous) | MTG-Jamendo | -25.78 | -19.29 | -19.97 | -20.36 | CC BY |
| 9 | Cantonement Jazz Band - Bessemer | FMA | -26.12 | -20.30 | -21.68 | -24.48 | BY-NC-SA 3.0 International |
| 10 | The Imaginary Suitcase - o my | MTG-Jamendo | -27.59 | -23.96 | -25.29 | -26.40 | CC BY-SA |
| 11 | Mr. & Mrs. Smith - Cold Black Oil | FMA | -28.76 | -25.31 | -25.55 | -25.97 | BY-NC-SA |
| 12 | Burdeos - In (Feat Arrpa) | FMA | -29.77 | -25.46 | -27.86 | -29.99 | BY-NC-SA 3.0 International |
| 13 | Kate Orange - Нова Радість | MTG-Jamendo | -30.73 | -26.37 | -28.89 | -31.19 | CC BY-SA |
| 14 | Les Petits Chanteurs de Montigny - Soon and Very Soon (Negro Spiritual) | MTG-Jamendo | -31.36 | -28.40 | -29.09 | -29.86 | CC BY-SA |
| 15 | Welcome Wizard - Wilde Eye | FMA | -33.47 | -28.98 | -31.01 | -33.10 | BY-NC-SA |
| 16 | Zeffon - Earthquake | MTG-Jamendo | -45.59 | -41.69 | -42.52 | -44.25 | CC BY-NC-SA |

The ranking is a screening aid, not a listening verdict. In particular,
positive projection can identify vocal-like effects, spoken material, or
aggressively removed accompaniment. The retained set is intentionally biased
toward difficult local events so that listening effort is spent where the
model and Inst 3 differ most.

## Retained 12-song set

The following tracks were copied into the three retained output directories:

`data/inst3-mtg-fma-expansion-evaluation/retained-12/`

- `h50ContinuationPlus5`: current model instrumental output.
- `h50Residual`: current model residual output.
- `inst3Instrumental`: Inst 3 reference output.

Retained tracks are ranks 1-12 in the table above. The complete 16-song
candidate outputs remain under:

`data/inst3-mtg-fma-expansion-evaluation/all-candidates/`

The machine-readable report records all 16 songs, the retained ranking, the
combined ranking with the existing 12 private songs, exact output hashes, and
per-track attribution fields:

`data/inst3-mtg-fma-expansion-evaluation/expansion-evaluation-report.json`

Source/download and decoded provenance are recorded separately in:

`data/inst3-mtg-fma-expansion/source-manifest.json`

## Verification

- 16/16 songs completed with continuous-context model rendering.
- 48/48 FLAC outputs exist: model instrumental, model residual, and Inst 3
  instrumental for each candidate.
- All outputs are finite, stereo, 44.1 kHz, and have the frame counts recorded
  in the report.
- All 16 records contain license, license URL, download URL, raw SHA-256, and
  teacher-cache provenance fields.
- The report is marked `completed` and can be resumed safely if regenerated
  with the same manifest and checkpoint.

## Interpretation and next step

The new pool supplies substantially harder local events than the easiest
candidate tracks; the top two have approximately -15.7 dBFS 100-ms positive
projection maxima, while the least severe track is below -42 dBFS. This gives
a useful listening range without treating every residual as a confirmed vocal
error.

The next decision should be listening-based: compare the retained 12 model
outputs against their matching Inst 3 references, mark which events are truly
vocal/voice-like versus acceptable instruments, and use only the confirmed
subset for a future targeted experiment. Do not train on the expansion pool
or refresh the model from these tracks until that annotation step is complete.
